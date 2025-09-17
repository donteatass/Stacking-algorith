from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
import pandas as pd
import copy
from pathlib import Path
import math

@dataclass
class Sat:
    sat: str
    filled: bool
    interim: bool
    final: bool
    issue: bool
    @property
    def cleared(self) -> bool:
        return bool(self.filled and self.interim and self.final and (not self.issue))

@dataclass
class Action:
    step: int
    action: str
    location: str
    count: int
    items_top_to_bottom: List[str]
    note: str = ""

@dataclass
class StackState:
    name: str
    items: List[Sat] = field(default_factory=list)
    cap: int = 15
    def top(self) -> Optional[Sat]:
        return self.items[0] if self.items else None
    def len(self) -> int:
        return len(self.items)
    def space_left(self) -> int:
        return self.cap - len(self.items)
    def push_batch(self, sats_top_to_bottom: List[Sat]):
        self.items = list(sats_top_to_bottom) + self.items
    def pop_batch(self, k: int) -> List[Sat]:
        batch = self.items[:k]
        self.items = self.items[k:]
        return batch

    def pop_batch_bottom(self, k: int) -> List[Sat]:
        """
        Pop a batch of `k` items from the bottom of the stack.

        The returned list is ordered top-to-bottom relative to the popped segment.  For example, if
        the stack contains [top, ..., bottom] and k=3, this returns the bottom three items in the order
        [item_{-k}, item_{-k+1}, ..., item_{-1}], where item_{-1} is the original bottom of the stack.
        """
        assert k <= len(self.items), "pop_batch_bottom exceeds available items"
        batch = self.items[-k:]
        self.items = self.items[:-k]
        return batch

class LiftPlannerV1P5:
    def __init__(self, source_stacks: List[StackState], temp_stack: StackState, dest_stack: StackState,
                 hand_capacity: int = 7, temp_cap: int = 15, target_ids: Optional[Set[str]] = None,
                 lookahead_depth: int = 2, beam_width: int = 3, early_temp_threshold: int = 12):
        self.sources = source_stacks
        self.temp = temp_stack
        self.dest = dest_stack
        self.temp.cap = temp_cap
        self.hand: List[Sat] = []
        self.hand_capacity = hand_capacity
        self.log: List[Action] = []
        self.step_counter = 0
        self.target_ids: Set[str] = set(target_ids or set())
        self.lookahead_depth = lookahead_depth
        self.beam_width = beam_width
        self.early_temp_threshold = early_temp_threshold
        self._mode_B_active = False
        # Track the step number of the most recent drop to TEMP.  This is used to avoid
        # immediately offloading items that have just been stashed into TEMP in the same
        # step.  A value of -1 indicates no drops have occurred yet.
        self.last_temp_drop_step: int = -1
        # When a drop to TEMP occurs, set this flag to True.  The next call to
        # maybe_early_temp_return will clear the flag and skip any early offload from
        # TEMP.  This avoids the pathological case where items are returned to TEMP
        # and immediately offloaded back to a source in the same extraction sequence.
        self.just_dropped_to_temp: bool = False
        # Track the stack currently being excavated in Mode A so that the early
        # TEMP-return heuristic does not immediately re-bury the same stack.
        self._active_excavation_stack: Optional[StackState] = None
        # Maintain a set of stacks whose blockers are currently stashed in TEMP.
        # Early TEMP returns should avoid these stacks while they still contain
        # remaining targets, even if we have temporarily switched to excavating
        # other stacks.
        self._protected_temp_sources: Set[str] = set()

    def clone(self):
        return copy.deepcopy(self)

    # --- Logging helpers ---
    def _record(self, action: str, location: str, count: int, items: List[Sat], note: str = ""):
        self.step_counter += 1
        self.log.append(Action(self.step_counter, action, location, count, [x.sat for x in items], note))

    def get_log_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{"step":a.step,"action":a.action,"location":a.location,"count":a.count,
                              "items_top_to_bottom":a.items_top_to_bottom,"note":a.note} for a in self.log])

    def get_log_dataframe_compact(self) -> pd.DataFrame:
        rows = []
        carry = None  # type: Optional[Action]
        for a in self.log:
            if a.action == "pick":
                if carry and carry.action == "pick" and carry.location == a.location:
                    carry.count += a.count
                    carry.items_top_to_bottom.extend(a.items_top_to_bottom)
                    if a.note and a.note not in carry.note:
                        carry.note = (carry.note + " | " + a.note).strip(" |")
                else:
                    if carry:
                        rows.append(carry)
                    carry = Action(a.step, a.action, a.location, a.count, list(a.items_top_to_bottom), a.note)
            else:
                if carry:
                    rows.append(carry); carry = None
                rows.append(a)
        if carry:
            rows.append(carry)
        for i, a in enumerate(rows, 1):
            a.step = i
        return pd.DataFrame([{"step":a.step,"action":a.action,"location":a.location,"count":a.count,
                              "items_top_to_bottom":a.items_top_to_bottom,"note":a.note} for a in rows])

    
    def save_log_to_csv(self, path: Path, compact: bool = True) -> None:
        """Write the action log to CSV. Uses compact format by default."""
        df = self.get_log_dataframe_compact() if compact else self.get_log_dataframe()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def get_summary(self) -> Dict[str, Any]:
        """
        Produce a summary of the actions taken and the final state of all stacks.

        The summary includes counts of pick and drop actions, the number of satellites in each stack,
        and the final order of satellites in the destination, source, and temporary stacks.
        Satellite lists are ordered from top (index 0) to bottom.
        """
        picks = sum(1 for a in self.log if a.action == "pick")
        drops = sum(1 for a in self.log if a.action == "drop")
        summary: Dict[str, Any] = {
            "total_actions": picks + drops,
            "pick_actions": picks,
            "drop_actions": drops,
            "dest_count": len(self.dest.items),
            "temp_count": len(self.temp.items),
            "hand_count": len(self.hand),
            "source_lengths": {s.name: len(s.items) for s in self.sources},
        }
        # Add final stack orders for verification.
        summary["final_dest_top_to_bottom"] = [sat.sat for sat in self.dest.items]
        summary["final_source_stacks_top_to_bottom"] = {
            s.name: [sat.sat for sat in s.items] for s in self.sources
        }
        summary["final_temp_top_to_bottom"] = [sat.sat for sat in self.temp.items]
        return summary

    # --- Target-set helpers ---
    def _remaining_targets(self) -> Set[str]:
        delivered = {x.sat for x in self.dest.items}
        return {t for t in self.target_ids if t not in delivered}

    def _is_remaining_target(self, sat: Optional[Sat]) -> bool:
        return bool(sat and sat.sat in self._remaining_targets())

    def _stack_has_any_remaining_target(self, stack: 'StackState') -> bool:
        rem = self._remaining_targets()
        return any(s.sat in rem for s in stack.items)

    def _has_cleared_near_surface(self, stack: 'StackState', depth: int) -> bool:
        """Return True if any cleared sat occurs within the top `depth` items."""
        return any(s.cleared for s in stack.items[:depth])

    # --- Heuristics for temp returns ---
    def _score_offload_stack(self, stack: 'StackState', batch_size: int):
        """
        Compute a score for offloading a batch of blockers to a source stack.  Lower
        scores are better.  The score combines several heuristics:

        * Burying a remaining target or cleared satellite near the surface incurs a
          penalty proportional to how close it is to the top.
        * Stacks with any remaining targets receive a base penalty so we avoid hiding
          them when possible.
        * Preference is given to stacks with more free space and shorter overall length.
        """
        if len(stack.items) + batch_size > stack.cap:
            return (10**9, 0, 0)

        has_any = int(self._stack_has_any_remaining_target(stack))

        rem = self._remaining_targets()
        depth = None
        for i, sat in enumerate(stack.items):
            if sat.sat in rem:
                depth = i
                break

        threshold = min(batch_size, self.hand_capacity)
        margin = 2
        depth_pen = 0
        if depth is not None and depth < threshold + margin:
            depth_pen = 150 + (threshold + margin - depth) * 20
        elif depth is not None:
            depth_pen = max(0, 10 - depth)

        cleared_pen = sum(1 for sat in stack.items[:batch_size] if sat.cleared) * 100

        space = stack.space_left()
        score_primary = has_any * 50 + depth_pen + cleared_pen
        return (score_primary, -space, -len(stack.items))

    def _choose_offload_stack(self, batch_size: int, exclude: Optional['StackState']=None,
                               allow_temp: bool = True) -> Optional['StackState']:
        """
        Choose an offload destination for a batch of blockers.  Candidate stacks include
        all sources (subject to capacity and top-target checks) as well as TEMP when it
        has sufficient space (if ``allow_temp`` is True).  The stack with the lowest
        heuristic score is returned.

        Args:
            batch_size: Number of blockers to offload.
            exclude: A stack to exclude from consideration.
            allow_temp: When False, TEMP is not considered as a destination.

        Returns:
            The chosen StackState, or None if no valid offload site exists.
        """
        def allowed(s: 'StackState') -> bool:
            if exclude is not None and s is exclude:
                return False
            if len(s.items) + batch_size > s.cap:
                return False
            if self._is_remaining_target(s.top()):
                return False
            return True

        candidates: List[Tuple[Tuple[int, int, int], StackState]] = []
        for s in self.sources:
            if allowed(s):
                candidates.append((self._score_offload_stack(s, batch_size), s))

        if allow_temp and self.temp.space_left() >= batch_size:
            candidates.append(((120, 0, 0), self.temp))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _cleanup_protected_temp_sources(self) -> None:
        """Drop stacks from the protected set once their targets are exhausted."""
        if not self._protected_temp_sources:
            return
        active: Set[str] = set()
        for stack in self.sources:
            if stack.name in self._protected_temp_sources and self._stack_has_any_remaining_target(stack):
                active.add(stack.name)
        self._protected_temp_sources = active

    def _ensure_temp_space(self, needed: int):
        while self.temp.space_left() < needed:
            # Determine which source stacks can safely receive offloaded items.
            def safe_drop_target(s: 'StackState') -> bool:
                # Cannot offload to full stacks or onto a remaining target. Mode B additionally
                # avoids stacks that hide cleared satellites near the surface.
                if s.space_left() <= 0:
                    return False
                if self._is_remaining_target(s.top()):
                    return False
                if self._mode_B_active and self._has_cleared_near_surface(s, self.hand_capacity):
                    return False
                return True

            safe = [s for s in self.sources if safe_drop_target(s)]
            if not safe:
                # If no safe stack exists, try to clear top targets to create space
                tops = [s for s in self.sources if self._is_remaining_target(s.top())]
                if not tops:
                    # In Mode B, we can also move cleared tops directly to DEST
                    if self._mode_B_active:
                        cleared_tops = [s for s in self.sources if s.top() and s.top().cleared]
                        if cleared_tops:
                            self._pick(cleared_tops[0], 1, note="Free stack by moving top cleared to DEST")
                            self._drop(self.dest, 1, note="Drop cleared top to DEST (Mode B)")
                            continue
                    # If even that fails, deadlock; no space can be freed
                    raise RuntimeError("Deadlock: no safe stacks and no top targets to clear.")
                # Clear a top target to make a safe stack
                self._pick(tops[0], 1, note="Free stack top target to create safe offload site")
                self._drop(self.dest, 1, note="Clear top target to DEST")
                continue

            # Select the best stack to offload into based on scoring
            safe.sort(key=lambda s: self._score_offload_stack(s,
                                                             min(self.hand_capacity, len(self.temp.items), s.space_left())))
            target_stack = safe[0]
            move_n = min(self.hand_capacity, len(self.temp.items), target_stack.space_left())
            if move_n == 0:
                break
            # Free space by moving a batch from TEMP to a safe source.  Use the unified
            # _pick/_drop helpers so that the single-pick/single-drop contract is
            # respected even during this maintenance operation.
            self._pick(self.temp, move_n, note="Freeing TEMP space (early offload)")
            self._drop(target_stack, move_n, note="Return from TEMP to source (batched)")

    def maybe_early_temp_return(self):
        # Avoid immediately offloading items that were just dropped to TEMP.  When this flag
        # is set, clear it and return without performing any early offload.  This allows
        # freshly stashed blockers to remain in TEMP for at least one extraction cycle.
        if self.just_dropped_to_temp:
            # Reset the flag and skip early temp return
            self.just_dropped_to_temp = False
            return
        self._cleanup_protected_temp_sources()
        if len(self.temp.items) >= self.early_temp_threshold:
            best = None
            best_stack = None
            for s in self.sources:
                if s.space_left() <= 0: continue
                if self._is_remaining_target(s.top()): continue
                if self._mode_B_active and s.top() and s.top().cleared: continue
                if not self._mode_B_active:
                    has_remaining = self._stack_has_any_remaining_target(s)
                    if has_remaining and self._active_excavation_stack is s:
                        continue
                    if has_remaining and s.name in self._protected_temp_sources:
                        continue
                score = self._score_offload_stack(s, min(self.hand_capacity, len(self.temp.items), s.space_left()))
                if best is None or score < best:
                    best = score
                    best_stack = s
            if best_stack:
                move_n = min(self.hand_capacity, len(self.temp.items), best_stack.space_left())
                # Move a batch from TEMP back to a source stack using unified pick/drop
                # helpers.  This preserves ordering and satisfies the unified-mode
                # requirement of one pick followed by one drop.
                self._pick(self.temp, move_n, note="Proactive TEMP reduction")
                self._drop(best_stack, move_n, note="Early temp return (reduces future costs)")

    # --- Utility helpers ---
    def _first_remaining_target_in_stack(self, stack: 'StackState'):
        rem = self._remaining_targets()
        for idx, sat in enumerate(stack.items):
            if sat.sat in rem:
                return idx, sat
        return None

    def _candidate_stacks(self) -> List['StackState']:
        return [s for s in self.sources if self._first_remaining_target_in_stack(s) is not None]

    def _simulate_extract_from_stack(self, stack_name: str):
        sim = self.clone()
        s = next(x for x in sim.sources if x.name == stack_name)
        before = len(sim.log)
        sim._extract_one_from_stack(s)
        delta_actions = len(sim.log) - before
        return sim, delta_actions

    def _beam_choose_next_stack(self) -> 'StackState':
        cands = self._candidate_stacks()
        if not cands:
            raise RuntimeError("No candidates found.")
        # If only one candidate, no need for complex heuristics
        if len(cands) == 1:
            return cands[0]
        # --- Heuristic selection for Mode A with deeper run consideration ---
        # Build candidate information: for each stack, compute the index of the first remaining
        # target (first_idx), all contiguous segments of remaining targets (with their start
        # positions), the length of the largest contiguous run, and the starting index of that run.
        candidate_info = []  # tuples: (stack, first_idx, reachable, max_run_len, total_targets, segments_cnt, largest_start)
        rem = self._remaining_targets()
        for s in cands:
            # Find the index of the first remaining target in this stack
            first_idx = None
            for idx, sat in enumerate(s.items):
                if sat.sat in rem:
                    first_idx = idx
                    break
            if first_idx is None:
                continue
            reachable = first_idx <= self.hand_capacity
            # Compute contiguous segments of remaining targets along with their start offsets
            items_after = s.items[first_idx:]
            flags = [(item.sat in rem) for item in items_after]
            segments: List[Tuple[int, int]] = []  # (start_offset, length)
            i = 0
            while i < len(flags):
                if flags[i]:
                    start = i
                    while i < len(flags) and flags[i]:
                        i += 1
                    segments.append((start, i - start))
                else:
                    i += 1
            if segments:
                # Choose the largest contiguous run; tie-break on shallowest start
                segments.sort(key=lambda x: (-x[1], x[0]))
                largest_start, max_run_len = segments[0]
                total_targets = sum(l for _, l in segments)
                segments_cnt = len(segments)
            else:
                # Should not happen: no contiguous segments
                largest_start, max_run_len = 0, 1
                total_targets = 1
                segments_cnt = 1
            candidate_info.append((s, first_idx, reachable, max_run_len, total_targets, segments_cnt, largest_start))
        if not candidate_info:
            # Fallback: use first candidate if none computed
            return cands[0]
        # Evaluate each candidate by an approximate cost to reach and deliver its largest run
        # Compute depth to largest run: first_idx + largest_start
        scored: List[Tuple[float, int, int, Tuple]] = []  # (cost, -max_run_len, first_idx, candidate)
        for info in candidate_info:
            stack_c, first_idx, reachable_c, max_run_len, total_targets, segments_cnt, largest_start = info
            idx_large = first_idx + largest_start
            # Cost to peel blockers and deliver largest run: 2 * (ceil(idx_large/hand) + ceil(max_run_len/hand)) / max_run_len
            lifts_blockers = math.ceil(idx_large / self.hand_capacity)
            lifts_goods = math.ceil(max_run_len / self.hand_capacity)
            cost = (lifts_blockers + lifts_goods) * 2 / max_run_len
            scored.append((cost, -max_run_len, first_idx, info))
        # Sort candidates by cost, then by longer max_run_len, then by shallower first_idx
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        best_info = scored[0][3]
        best_stack = best_info[0]
        return best_stack

    def _contiguous_top_remaining_targets(self, stack: 'StackState') -> int:
        if not self._is_remaining_target(stack.top()):
            return 0
        rem = self._remaining_targets()
        cnt = 0
        for sat in stack.items:
            if sat.sat in rem:
                cnt += 1
            else:
                break
        return cnt

    def _contiguous_top_cleared(self, stack: 'StackState') -> int:
        cnt = 0
        for sat in stack.items:
            if sat.cleared:
                cnt += 1
            else:
                break
        return cnt

    def _contiguous_remaining_from_index(self, stack: 'StackState', start: int) -> int:
        rem = self._remaining_targets()
        cnt = 0
        for sat in stack.items[start:]:
            if sat.sat in rem:
                cnt += 1
            else:
                break
        return cnt

    # --- Core extraction step for Mode A ---
    # --- Modes ---
    def plan_mode_A(self):
        """
        Unified Mode A: deliver remaining target IDs using a deeper look‑ahead similar
        to Mode B.  When possible, deliver contiguous top runs of targets directly.
        Otherwise, peel enough blockers (including intermediate targets) to expose
        the largest contiguous run of remaining targets in any stack.  This
        approach avoids repeatedly peeling shallow, fragmented target runs when a
        deeper run yields a lower cost per target.  It preserves the unified
        single‑pick/single‑drop constraint used throughout the planner.
        """
        self._active_excavation_stack = None
        self._protected_temp_sources.clear()
        if not self.target_ids:
            raise AssertionError("Mode A requires target_ids to be provided.")
        # Verify that all targets exist in the source stacks
        all_ids = {sat.sat for s in self.sources for sat in s.items}
        missing = [t for t in self.target_ids if t not in all_ids]
        if missing:
            raise AssertionError(f"Targets not found in stacks: {missing}")
        # Continue until all targets have been delivered to the destination
        # Note: use a while loop rather than recursion to support deep look‑ahead
        while True:
            remaining_targets = self._remaining_targets()
            if not remaining_targets:
                break
            # Determine how many targets have been delivered so far
            delivered = len({x.sat for x in self.dest.items if x.sat in self.target_ids})
            remaining = len(self.target_ids) - delivered
            if remaining <= 0:
                break
            # --- Step 1: Identify the best top‑run of remaining targets across all stacks ---
            best_top_run = 0
            best_top_stack: Optional[StackState] = None
            for s in self.sources:
                # Skip stacks with no remaining targets
                if not self._stack_has_any_remaining_target(s):
                    continue
                # Compute the length of the contiguous run of remaining targets at the top
                run_len = self._contiguous_top_remaining_targets(s)
                if run_len > best_top_run:
                    best_top_run = run_len
                    best_top_stack = s
            # Compute the cost of delivering this top run.  If run is longer than
            # the hand capacity, multiple lifts will be required.
            if best_top_run > 0:
                lifts_top = math.ceil(best_top_run / self.hand_capacity)
                cost_top = 2 * lifts_top / best_top_run
            else:
                cost_top = float('inf')
            # --- Step 2: Identify the largest contiguous runs of remaining targets within each stack ---
            candidates_info: List[Tuple[StackState, int, int]] = []  # (stack, idx_large, run_len_large)
            # Build a set of currently delivered items to avoid recounting in candidate selection
            dest_set = {x.sat for x in self.dest.items}
            rem_set = remaining_targets
            for s in self.sources:
                # Skip stacks with no remaining targets
                if not self._stack_has_any_remaining_target(s):
                    continue
                # Build a boolean flag list for each item: True if it's a remaining target
                flags = [(sat.sat in rem_set) for sat in s.items]
                # Find the index of the first remaining target
                try:
                    first_idx = flags.index(True)
                except ValueError:
                    continue
                # Consider only the sublist from the first target downward
                items_after = flags[first_idx:]
                # Identify all contiguous segments of True values and record their start offsets and lengths
                segments: List[Tuple[int, int]] = []
                i = 0
                while i < len(items_after):
                    if items_after[i]:
                        start = i
                        while i < len(items_after) and items_after[i]:
                            i += 1
                        segments.append((start, i - start))
                    else:
                        i += 1
                if not segments:
                    continue
                # Choose the largest contiguous run; on tie, use the shallowest start
                segments.sort(key=lambda x: (-x[1], x[0]))
                largest_start, run_len_large = segments[0]
                # Compute the absolute index of the start of this large run in the stack
                idx_large = first_idx + largest_start
                # We only consider peeling to reach runs that are not at the very top
                if idx_large > 0:
                    candidates_info.append((s, idx_large, run_len_large))
            # If candidates exist, compute their cost per target for peeling and delivering
            best_large_cost = float('inf')
            best_cand: Optional[Tuple[StackState, int, int]] = None
            if candidates_info:
                # Compute approximate cost ratio for peeling blockers and delivering the largest run
                cost_candidates: List[Tuple[float, int, Tuple[StackState, int, int]]] = []
                for stack_c, idx_large, run_len_large in candidates_info:
                    lifts_blockers = math.ceil(idx_large / self.hand_capacity)
                    lifts_goods = math.ceil(run_len_large / self.hand_capacity)
                    actions = 2 * (lifts_blockers + lifts_goods)
                    cost = actions / run_len_large if run_len_large > 0 else float('inf')
                    cost_candidates.append((cost, idx_large, (stack_c, idx_large, run_len_large)))
                # Select candidate with lowest cost; break ties by shallower depth
                cost_candidates.sort(key=lambda x: (x[0], x[1]))
                best_large_cost, _, best_cand = cost_candidates[0]
            # --- Step 3: Decide whether to deliver a top‑run or peel to a deeper run ---
            if best_top_run > 0 and cost_top <= best_large_cost:
                # Deliver the contiguous top run of targets from the best stack
                take = min(best_top_run, self.hand_capacity, remaining)
                # Note: we use _pick and _drop directly to enforce unified single pick/drop
                self._active_excavation_stack = best_top_stack
                self._pick(best_top_stack, take, note=f"Mode A unified: pick {take} target(s) from top")
                self._drop(self.dest, take, note=f"Mode A unified: deliver {take} target(s) to DEST")
                # Update delivered count automatically via dest stack
                self.maybe_early_temp_return()
                continue
            # Otherwise, peel blockers (and intermediate targets) to reach the large run
            if best_cand is not None:
                stack_c, idx_large, run_len_large = best_cand
                # Determine how many blockers to peel: the depth of the large run
                peel_n = idx_large
                # Peel up to hand capacity or the full depth, whichever is smaller
                chunk = min(self.hand_capacity, peel_n)
                # Identify an offload destination for the peeled items
                offload = self._choose_offload_stack(chunk, exclude=stack_c)
                if offload is None:
                    # No safe source available; use TEMP (expand it if needed)
                    if self.temp.space_left() < chunk:
                        self._ensure_temp_space(chunk)
                    offload = self.temp
                # Pick and offload blockers/targets from the chosen stack
                self._active_excavation_stack = stack_c
                self._pick(stack_c, chunk, note="Mode A unified: peel blockers")
                self._drop(offload, chunk, note=f"Mode A unified: offload to {offload.name}")
                self.maybe_early_temp_return()
                continue
                        # --- TEMP recovery path: if no source has remaining targets but TEMP does, deliver/unwind TEMP ---
            if not self._candidate_stacks() and self._stack_has_any_remaining_target(self.temp):
                temp_run = self._contiguous_top_remaining_targets(self.temp)
                if temp_run > 0:
                    take = min(temp_run, self.hand_capacity, remaining)
                    self._pick(self.temp, take, note="Mode A unified: deliver from TEMP")
                    self._drop(self.dest, take, note="Mode A unified: deliver from TEMP to DEST")
                    self.maybe_early_temp_return()
                    continue
                if len(self.temp.items) > 0:
                    remaining = self._remaining_targets()
                    blockers_prefix = 0
                    for sat in self.temp.items:
                        if sat.sat in remaining:
                            break
                        blockers_prefix += 1
                    desired_move = min(blockers_prefix, self.hand_capacity)
                    if desired_move == 0:
                        desired_move = min(self.hand_capacity, len(self.temp.items))
                    move_n = desired_move
                    chosen_site: Optional[StackState] = None
                    while move_n > 0:
                        candidate = self._choose_offload_stack(move_n, allow_temp=False)
                        if candidate is not None:
                            chosen_site = candidate
                            break
                        move_n -= 1
                    if chosen_site is not None and move_n > 0:
                        self._pick(self.temp, move_n, note="Unwind TEMP blockers (minimal exposure)")
                        self._drop(chosen_site, move_n, note=f"Unwind TEMP blockers to {chosen_site.name}")
                        self.maybe_early_temp_return()
                        continue
# Fallback: no large run candidates and no top run; use the original heuristics
            # Fall back to beam search when deeper look‑ahead cannot find a candidate
            if not self._remaining_targets():
                break
            # Use existing beam search heuristics for tough cases
            stack = self._beam_choose_next_stack()
            self._extract_one_from_stack(stack)
        # After all targets delivered, return items from TEMP to sources
        self.return_all_temp()
        self._active_excavation_stack = None
        self._protected_temp_sources.clear()

    def return_all_temp(self):
        while len(self.temp.items) > 0:
            best = None; best_stack=None
            for s in self.sources:
                if s.space_left() <= 0: continue
                if self._is_remaining_target(s.top()): continue
                if self._mode_B_active and s.top() and s.top().cleared: continue
                score = self._score_offload_stack(s, min(self.hand_capacity, len(self.temp.items), s.space_left()))
                if best is None or score < best:
                    best = score; best_stack = s
            if not best_stack:
                cand = [s for s in self.sources if s.space_left() > 0]
                if not cand:
                    raise RuntimeError("No space to return TEMP items at end.")
                if self._mode_B_active:
                    non_cleared_top = [s for s in cand if not (s.top() and s.top().cleared)]
                    if non_cleared_top:
                        best_stack = max(non_cleared_top, key=lambda s: s.space_left())
                    else:
                        best_stack = max(cand, key=lambda s: s.space_left())
                else:
                    best_stack = max(cand, key=lambda s: s.space_left())
            move_n = min(self.hand_capacity, len(self.temp.items), best_stack.space_left())
            self._pick(self.temp, move_n, note="Final cleanup: empty TEMP")
            self._drop(best_stack, move_n, note="Return TEMP to source")

    # === Unified single-pick/single-drop overrides ===
    # Enforce: one pick and one drop per lift; picks come from a single stack;
    # drop goes to exactly one destination (DEST, TEMP, or a safe source).

    def _ensure_unified_state(self):
        if not hasattr(self, 'unified_mode'):
            self.unified_mode = True
        if not hasattr(self, '_lift_in_progress'):
            self._lift_in_progress = False

    def _pick(self, stack: 'StackState', k: int, note: str = ""):
        # Unified guard: only one pick per lift
        self._ensure_unified_state()
        if getattr(self, 'unified_mode', False):
            if self._lift_in_progress:
                raise AssertionError("Unified mode: cannot perform multiple picks in a single lift.")
            if len(self.hand) != 0:
                # Any residual in hand implies previous drop wasn't completed; disallow
                raise AssertionError("Unified mode: hand must be empty before a pick.")
            self._lift_in_progress = True
        # Original logic
        assert 1 <= k <= len(stack.items), "Pick exceeds available."
        assert len(self.hand) + k <= self.hand_capacity, "Hand over capacity."
        batch = stack.pop_batch(k)  # top-to-bottom
        self.hand.extend(batch)
        self._record("pick", stack.name, k, batch, note)

    def _drop(self, stack: 'StackState', k: int, note: str = ""):
        # Unified guard: exactly one drop per lift, must drop entire hand
        self._ensure_unified_state()
        if getattr(self, 'unified_mode', False):
            if not self._lift_in_progress:
                raise AssertionError("Unified mode: drop without a prior pick.")
            if k != len(self.hand):
                raise AssertionError("Unified mode: must drop all lifted sats in a single drop.")
        # Original safety checks
        assert 1 <= k <= len(self.hand), "Drop exceeds hand content."
        batch_top_to_bottom = self.hand[-k:]
        if stack is not self.dest:
            assert len(stack.items) + k <= stack.cap, f"Drop would exceed cap of {stack.name}"
            if stack is not self.temp and self._is_remaining_target(stack.top()):
                raise AssertionError(f"Cannot drop onto {stack.name}: top is a remaining target.")
        stack.push_batch(batch_top_to_bottom)
        del self.hand[-k:]
        self._record("drop", stack.name, k, batch_top_to_bottom, note)
        # Unified guard: close the lift
        if getattr(self, 'unified_mode', False):
            self._lift_in_progress = False
        if stack is self.temp:
            self.just_dropped_to_temp = True
            if (
                not self._mode_B_active
                and self._active_excavation_stack
                and self._active_excavation_stack is not self.temp
            ):
                self._protected_temp_sources.add(self._active_excavation_stack.name)
        elif stack is self.dest:
            if (
                not self._mode_B_active
                and self._active_excavation_stack
                and self._active_excavation_stack is not self.temp
                and not self._stack_has_any_remaining_target(self._active_excavation_stack)
            ):
                self._protected_temp_sources.discard(self._active_excavation_stack.name)

    def _extract_one_from_stack(self, stack: 'StackState') -> bool:
        """Unified override for Mode A: one pick and one drop per lift."""
        # If stack top already has remaining targets, deliver them in one lift
        top_run = self._contiguous_top_remaining_targets(stack)
        if top_run > 0:
            take = min(top_run, self.hand_capacity)
            self._active_excavation_stack = stack
            self._pick(stack, take, note=f"Pick {take} target(s) from top (unified)")
            self._drop(self.dest, take, note=f"Deliver {take} target(s) to DEST (unified)")
            self.maybe_early_temp_return()
            return True
        frt = self._first_remaining_target_in_stack(stack)
        if frt is None:
            return False
        idx, _ = frt
        # Peel blockers only (no mixed pick), offload in one drop
        chunk = min(self.hand_capacity, idx)
        if chunk <= 0:
            return False
        # Ensure we have somewhere to offload; prefer a safe source
        offload = self._choose_offload_stack(chunk, exclude=stack)
        if offload is None:
            if self.temp.space_left() < chunk:
                self._ensure_temp_space(chunk)
            offload = self.temp
        self._active_excavation_stack = stack
        self._pick(stack, chunk, note="Peel blockers (unified)")
        self._drop(offload, chunk, note=f"Offload blockers to {offload.name} (unified)")
        self.maybe_early_temp_return()
        return True

    def plan_mode_B(self, count: int = 28):
        """Unified Mode B: only one pick and one drop per lift.
        Strategy: deliver top-run cleared when present; otherwise peel blockers from the stack
        with the shallowest next-cleared target, offloading to a safe source or TEMP.
        """
        total_cleared = (
            sum(1 for s in self.sources for sat in s.items if sat.cleared)
            + sum(1 for sat in self.temp.items if sat.cleared)
        )
        if total_cleared < count:
            raise ValueError("not enough cleared sats for a stack")
        self._mode_B_active = True
        delivered = len([x for x in self.dest.items if x.cleared])
        while delivered < count:
            remaining = count - delivered
            # 1) Determine candidate stacks by analysing the largest contiguous cleared run in each stack
            candidates_info: List[Tuple[StackState, int, int]] = []  # (stack, idx_large, run_len_large)
            dest_set = {x.sat for x in self.dest.items}
            for s in self.sources + [self.temp]:
                # Compute flags for cleared sats not yet delivered
                flags = [(sat.cleared and sat.sat not in dest_set) for sat in s.items]
                # Find all contiguous cleared segments and record their start indices and lengths
                segments = []  # list of (start_idx, length)
                i = 0
                while i < len(flags):
                    if flags[i]:
                        start = i
                        while i < len(flags) and flags[i]:
                            i += 1
                        segments.append((start, i - start))
                    else:
                        i += 1
                if not segments:
                    continue
                # Choose the largest segment by length; if tie, pick the shallowest (smallest start)
                segments.sort(key=lambda x: (-x[1], x[0]))
                idx_large, run_len_large = segments[0]
                # Only consider peeling if there is at least one blocker above the large run
                if idx_large > 0:
                    candidates_info.append((s, idx_large, run_len_large))
            # If no candidates found, attempt to deliver any top-run cleared sats.
            # This situation arises when all remaining cleared satellites are already
            # exposed at the top of the stacks. The previous logic would break out of
            # the loop before delivering these sats, leaving the plan short of the
            # requested count. Instead, deliver the available top-run cleared sats
            # before deciding to terminate the loop.
            if not candidates_info:
                best_stack = None
                best_run = 0
                for s in self.sources + [self.temp]:
                    run = self._contiguous_top_cleared(s)
                    if run > best_run:
                        best_run = run
                        best_stack = s
                if best_run > 0:
                    take = min(best_run, self.hand_capacity, remaining)
                    self._pick(best_stack, take, note="Mode B unified: pick cleared from top")
                    self._drop(self.dest, take, note="Mode B unified: drop to DEST")
                    delivered += take
                    self.maybe_early_temp_return()
                    continue
                break
            # Compute approximate cost ratio for each candidate
            cost_candidates: List[Tuple[float, int, Tuple]] = []
            for cand in candidates_info:
                stack_c, idx_large, run_len_large = cand
                # Number of lifts needed to peel blockers above the large run
                lifts_blockers = math.ceil(idx_large / self.hand_capacity)
                # Number of lifts needed to deliver the large run
                lifts_goods = math.ceil(run_len_large / self.hand_capacity)
                actions = 2 * (lifts_blockers + lifts_goods)
                cost = actions / run_len_large if run_len_large > 0 else float('inf')
                cost_candidates.append((cost, idx_large, cand))
            # Sort by cost then by shallower idx
            cost_candidates.sort(key=lambda x: (x[0], x[1]))
            best_cost = cost_candidates[0][0]
            # Determine top-run cleared and decide whether to deliver or peel
            best_stack = None; best_run = 0
            for s in self.sources + [self.temp]:
                run = self._contiguous_top_cleared(s)
                if run > best_run:
                    best_run = run; best_stack = s
            # Compute cost of delivering top-run cleared (one pick and one drop) per cleared
            deliver_cost = (2 / best_run) if best_run > 0 else float('inf')
            # Decide: deliver top-run if cost is no worse than peeling; else peel
            if best_run > 0 and deliver_cost <= best_cost:
                take = min(best_run, self.hand_capacity, remaining)
                self._pick(best_stack, take, note=f"Mode B unified: pick {take} cleared from top")
                self._drop(self.dest, take, note="Mode B unified: drop to DEST")
                delivered += take
                self.maybe_early_temp_return()
                continue
            # Otherwise, peel blockers from the best candidate based on largest run
            best_c = cost_candidates[0][2]  # (stack, idx_large, run_len_large)
            stack, idx_large, run_len_large = best_c
            # Peel enough blockers to expose at least one part of the large run
            peel_n = idx_large if idx_large > 0 else 1
            chunk = min(self.hand_capacity, peel_n)
            offload = self._choose_offload_stack(chunk, exclude=stack)
            if offload is None:
                if stack is self.temp:
                    raise RuntimeError("Cannot offload blockers from TEMP: no available stack")
                if self.temp.space_left() < chunk:
                    self._ensure_temp_space(chunk)
                offload = self.temp
            self._pick(stack, chunk, note="Mode B unified: peel blockers")
            self._drop(offload, chunk, note=f"Mode B unified: offload to {offload.name}")
            self.maybe_early_temp_return()
            delivered = len([x for x in self.dest.items if x.cleared])
        self._mode_B_active = False
        self.return_all_temp()
def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)): return bool(x)
    s = str(x).strip().lower()
    return s in ("true","t","1","yes","y")

def read_stack_csv(path: Path, name: str) -> StackState:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Support either a 'sat' or 'sat_id' column for satellite identifiers
    # Normalise column names to ensure both variants are accepted
    if 'sat' not in df.columns:
        # Accept 'sat_id' as an alternative to 'sat'
        if 'sat_id' in df.columns:
            df['sat'] = df['sat_id']
        else:
            raise ValueError(f"CSV {path} missing required column 'sat' or 'sat_id'")
    for c in ["filled","interim","final","issue"]:
        if c not in df.columns:
            raise ValueError(f"CSV {path} missing required column '{c}'")
    # Drop any unnamed columns (common when saving from spreadsheets)
    drop_cols = [c for c in df.columns if c.startswith('unnamed')]
    df = df.drop(columns=drop_cols, errors='ignore')
    items: List[Sat] = []
    for _, row in df.iterrows():
        items.append(Sat(
            sat=str(row["sat"]),
            filled=_parse_bool(row["filled"]),
            interim=_parse_bool(row["interim"]),
            final=_parse_bool(row["final"]),
            issue=_parse_bool(row["issue"]),
        ))
    # Set stack capacity to accommodate the initial number of items or the default of 15, whichever is larger.
    # This prevents capacity-related errors when returning blockers to the source stack for datasets
    # that have more than 15 satellites in a location.
    cap_size = max(15, len(items))
    return StackState(name=name, items=items, cap=cap_size)


def run_mode_A_with_failsafe_csv(planner: LiftPlannerV1P5, csv_path: Path, compact: bool = True) -> str:
    """
    Run Mode A planning and always emit a CSV of the moves seen so far, even if an exception occurs.
    Returns a status string ("success" or the exception message).
    """
    try:
        planner.plan_mode_A()
        status = "success"
    except Exception as e:
        status = f"failure: {e}"
    finally:
        try:
            planner.save_log_to_csv(csv_path, compact=compact)
        except Exception as write_err:
            status += f"; additionally failed to write CSV: {write_err}"
    return status
