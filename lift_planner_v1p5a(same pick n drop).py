
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
import pandas as pd
import copy
from pathlib import Path

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

    # --- Heuristics for temp returns ---
    def _score_offload_stack(self, stack: 'StackState', batch_size: int):
        """
        Compute a score for offloading a batch of TEMP items to a given stack.  Lower scores
        are better.  Penalize stacks that would require re-peeling soon by examining the depth
        of the next remaining target.  If the next target is very close to the top (e.g.,
        within a few items), avoid offloading to this stack so we don't immediately pick
        these blockers again.  Return a large penalty when the stack cannot accommodate
        the batch or when the depth is below a threshold.  Otherwise, prefer stacks with
        more free space and deeper remaining targets.
        """
        # If offloading would overflow the stack, forbid this candidate by returning huge
        if len(stack.items) + batch_size > stack.cap:
            return (10**9, 0, 0)
        # Determine if this stack still contains any remaining target
        has_any = int(self._stack_has_any_remaining_target(stack))
        # Find the depth (index) of the first remaining target in this stack
        depth = None
        rem = self._remaining_targets()
        for i, sat in enumerate(stack.items):
            if sat.sat in rem:
                depth = i
                break
        # If the next target is very close to the top (depth is small), avoid offloading here.
        # We use a threshold based on the hand capacity and batch_size: if the remaining target
        # would be within the current offload batch or within a small constant (3) above it,
        # we penalize heavily.  This helps prevent the situation where items are returned to
        # this stack only to be peeled again immediately afterwards.
        if depth is not None:
            # Determine threshold: the number of items we plan to offload plus a small margin.
            # Using min(batch_size, hand_capacity) ensures we only penalize when the target
            # would be buried within or just below the offloaded run.
            threshold = min(batch_size, self.hand_capacity)
            # Add a small margin to catch cases where the target is only a few items below
            # the offloaded region.  A margin of 2 provides a good balance between avoiding
            # immediate re-peeling and still utilizing available stacks when necessary.
            margin = 2
            if depth < threshold + margin:
                # Penalize extremely; this makes this stack a last resort for offloading
                return (10**9, 0, 0)
        # Compute a softer penalty based on depth; deeper targets are less likely to require
        # immediate peeling, so they receive a smaller penalty.  If there is no remaining
        # target in this stack, depth_pen is zero.
        depth_pen = 0 if depth is None else max(0, 10 - depth)
        # More space is better: we use negative space so that larger space yields a smaller
        # overall score.  We also use negative stack size to break ties by preferring
        # shorter stacks when space is equal.
        space = stack.space_left()
        return (has_any * 100 + depth_pen * 5, -space, -len(stack.items))

    def _choose_offload_stack(self, batch_size: int, exclude: Optional['StackState']=None) -> Optional['StackState']:
        """
        Choose a source stack to offload a batch of blockers into.  The stack must be
        able to accommodate the entire batch without exceeding its capacity, and
        cannot have a remaining target or (in Mode B) a cleared satellite at the
        top.  Optionally exclude a specific stack (typically the origin) to avoid
        burying the target we are trying to expose.

        Args:
            batch_size: Number of blockers to offload.
            exclude: A stack to exclude from consideration.

        Returns:
            The chosen StackState, or None if no valid offload site exists.
        """
        def allowed(s: 'StackState') -> bool:
            # Exclude the specified stack
            if exclude is not None and s is exclude:
                return False
            # Cannot exceed capacity
            if len(s.items) + batch_size > s.cap:
                return False
            # Cannot drop onto a remaining target
            if self._is_remaining_target(s.top()):
                return False
            # In Mode B, avoid dropping on cleared satellites
            if self._mode_B_active and s.top() and s.top().cleared:
                return False
            return True
        candidates = [s for s in self.sources if allowed(s)]
        if not candidates:
            return None
        # Choose the candidate with the lowest score according to _score_offload_stack
        candidates.sort(key=lambda s: self._score_offload_stack(s, batch_size))
        return candidates[0]

    def _ensure_temp_space(self, needed: int):
        while self.temp.space_left() < needed:
            # Determine which source stacks can safely receive offloaded items.
            def safe_drop_target(s: 'StackState') -> bool:
                # Cannot offload to full stacks or onto a remaining target. Mode B additionally
                # avoids dropping on cleared satellites.
                if s.space_left() <= 0:
                    return False
                if self._is_remaining_target(s.top()):
                    return False
                if self._mode_B_active and s.top() and s.top().cleared:
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
        if len(self.temp.items) >= self.early_temp_threshold:
            best = None
            best_stack = None
            for s in self.sources:
                if s.space_left() <= 0: continue
                if self._is_remaining_target(s.top()): continue
                if self._mode_B_active and s.top() and s.top().cleared: continue
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
        if len(cands) == 1:
            return cands[0]
        beams = [(0, self.clone(), [])]
        for depth in range(self.lookahead_depth):
            new_beams = []
            for score, state, path in beams:
                cands2 = state._candidate_stacks()
                if not cands2:
                    new_beams.append((score, state, path))
                    continue
                for c in cands2:
                    sim, delta = state._simulate_extract_from_stack(c.name)
                    new_beams.append((score + delta, sim, path + [c.name]))
            new_beams.sort(key=lambda t: t[0])
            beams = new_beams[:self.beam_width]
        best_score, best_state, best_path = beams[0]
        first_choice = best_path[0] if best_path else cands[0].name
        return next(s for s in self.sources if s.name == first_choice)

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
        if not self.target_ids:
            raise AssertionError("Mode A requires target_ids to be provided.")
        all_ids = {sat.sat for s in self.sources for sat in s.items}
        missing = [t for t in self.target_ids if t not in all_ids]
        if missing:
            raise AssertionError(f"Targets not found in stacks: {missing}")
        while self._remaining_targets():
            stack = self._beam_choose_next_stack()
            self._extract_one_from_stack(stack)
        self.return_all_temp()

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
        batch_top_to_bottom = self.hand[-k:][::-1]
        if stack is not self.dest:
            assert len(stack.items) + k <= stack.cap, f"Drop would exceed cap of {stack.name}"
            if stack is not self.temp and self._is_remaining_target(stack.top()):
                raise AssertionError(f"Cannot drop onto {stack.name}: top is a remaining target.")
            if self._mode_B_active and stack is not self.temp and stack.top() and stack.top().cleared:
                raise AssertionError(f"Cannot drop onto {stack.name}: top is a cleared sat (Mode B).")
        stack.push_batch(batch_top_to_bottom)
        del self.hand[-k:]
        self._record("drop", stack.name, k, batch_top_to_bottom, note)
        # Unified guard: close the lift
        if getattr(self, 'unified_mode', False):
            self._lift_in_progress = False
        if stack is self.temp:
            self.just_dropped_to_temp = True

    def _extract_one_from_stack(self, stack: 'StackState') -> bool:
        """Unified override for Mode A: one pick and one drop per lift."""
        # If stack top already has remaining targets, deliver them in one lift
        top_run = self._contiguous_top_remaining_targets(stack)
        if top_run > 0:
            take = min(top_run, self.hand_capacity)
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
        self._pick(stack, chunk, note="Peel blockers (unified)")
        self._drop(offload, chunk, note=f"Offload blockers to {offload.name} (unified)")
        self.maybe_early_temp_return()
        return True

    def plan_mode_B(self, count: int = 28):
        """Unified Mode B: only one pick and one drop per lift.
        Strategy: deliver top-run cleared when present; otherwise peel blockers from the stack
        with the shallowest next-cleared target, offloading to a safe source or TEMP.
        """
        self._mode_B_active = True
        delivered = len([x for x in self.dest.items if x.cleared])
        while delivered < count:
            remaining = count - delivered
            # 1) Deliver top-run cleared if available
            best_stack = None; best_run = 0
            for s in self.sources:
                run = self._contiguous_top_cleared(s)
                if run > best_run:
                    best_run = run; best_stack = s
            if best_stack and best_run > 0:
                take = min(best_run, self.hand_capacity, remaining)
                self._pick(best_stack, take, note=f"Mode B unified: pick {take} cleared from top")
                self._drop(self.dest, take, note="Mode B unified: drop to DEST")
                delivered += take
                self.maybe_early_temp_return()
                continue
            # 2) Otherwise peel blockers from the best candidate stack
            candidates = []
            for s in self.sources:
                for idx, sat in enumerate(s.items):
                    if sat.cleared and sat.sat not in {x.sat for x in self.dest.items}:
                        candidates.append((idx, s))
                        break
            if not candidates:
                break
            candidates.sort(key=lambda t: t[0])  # shallowest first
            idx, stack = candidates[0]
            chunk = min(self.hand_capacity, idx if idx > 0 else 1)
            offload = self._choose_offload_stack(chunk, exclude=stack)
            if offload is None:
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
