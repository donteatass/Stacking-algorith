"""I believe this removed old version 2 from mode B which is actually the most recent working portions
I think the iteriations are all kept and it just removed them and returned to
state of prior to all fixes

Tested and that was not the case, an overrwap was applied to plan mode B so that version 2
ran instead, the count was significantly higher in required moves 21 vs 33

Mode B does not return to source as well as mode A but this is a problem in 1.3
as well, not a problem in same pick n drop on mode A or B, it will return to source
"""
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
    def space_left(self) -> int:
        return self.cap - len(self.items)
    def push_batch(self, sats_top_to_bottom: List[Sat]):
        self.items = list(sats_top_to_bottom) + self.items
    def pop_batch(self, k: int) -> List[Sat]:
        batch = self.items[:k]
        self.items = self.items[k:]
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

    # --- Movement primitives ---
    def _pick(self, stack: 'StackState', k: int, note: str = ""):
        assert 1 <= k <= len(stack.items), "Pick exceeds available."
        assert len(self.hand) + k <= self.hand_capacity, "Hand over capacity."
        batch = stack.pop_batch(k)  # top-to-bottom
        # Keep pick order so drops are reverse-of-pick (LIFO)
        self.hand.extend(batch)
        self._record("pick", stack.name, k, batch, note)

    def _drop(self, stack: 'StackState', k: int, note: str = ""):
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
        # Mark that we just dropped items to TEMP.  This flag will be checked by
        # maybe_early_temp_return to prevent immediately offloading freshly stashed
        # blockers back to a source.  The flag is only set when dropping to the
        # temporary stack.
        if stack is self.temp:
            self.just_dropped_to_temp = True

    def _multi_stack_pick_and_drop(self, picks: List[Tuple['StackState', int]], flags: List[bool], note_prefix: str = ""):
        """
        Perform a pick across multiple stacks and then drop all items in a single sequence.

        `picks` is a list of (stack, count) tuples specifying how many items to pick
        from each stack, in order. `flags` is a list of booleans of the same length
        as the total items picked: True indicates the corresponding item is a
        deliverable satellite that should go to DEST, False indicates it is a
        blocker and should be stashed to TEMP. Items are assumed to be listed in
        the same order they will be picked (first pick's top-to-bottom items
        followed by subsequent picks).

        The method first executes the specified picks, accumulating all items in
        the hand. It then processes the `flags` in reverse (LIFO order) and
        issues the appropriate drops to the destination or temp stacks. All
        drops of consecutive items of the same type are batched together to
        minimize the number of drop actions.
        """
        # Perform the picks in sequence
        total = 0
        for stack, count in picks:
            if count <= 0:
                continue
            # annotate pick as part of multi-stack sequence
            note = f"{note_prefix}: pick"
            self._pick(stack, count, note=note)
            total += count
        # Validate that flags length matches number of items picked
        assert len(flags) == total, "Flags length must match total items picked"
        # Build runs of identical flag values in reverse order (top of hand first)
        run_type: Optional[bool] = None
        run_length = 0
        runs: List[Tuple[bool, int]] = []
        for f in reversed(flags):
            if run_type is None:
                run_type = f
                run_length = 1
            elif f == run_type:
                run_length += 1
            else:
                runs.append((run_type, run_length))
                run_type = f
                run_length = 1
        if run_type is not None:
            runs.append((run_type, run_length))
        # Drop each run: deliverables go to DEST, blockers to TEMP
        for is_deliver, length in runs:
            if length <= 0:
                continue
            if is_deliver:
                note = f"{note_prefix}: deliver run"
                self._drop(self.dest, length, note=note)
            else:
                note = f"{note_prefix}: stash blockers"
                self._drop(self.temp, length, note=note)

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

    def _drop_blockers_prefer_offload(self, origin: 'StackState', run_len: int, note_base: str):
        """
        Drop a run of blockers, preferring to offload directly onto a safe source stack
        rather than stashing in the temporary stack. If no single source stack can
        accept the entire run, fall back to the temporary stack.

        Args:
            origin: The stack from which the blockers were picked. This stack is
                excluded from consideration when selecting an offload stack.
            run_len: The number of blockers to drop.
            note_base: The base note used to annotate the drop action. The chosen
                destination will be appended to this note.
        """
        # Attempt to find a safe offload stack other than the origin.
        offload = self._choose_offload_stack(run_len, exclude=origin)
        if offload is not None:
            # Directly drop blockers to the selected offload stack.
            self._drop(offload, run_len, note=f"{note_base}: offload to {offload.name}")
            return
        # Otherwise, ensure there is enough TEMP space and drop to TEMP.
        if self.temp.space_left() < run_len:
            self._ensure_temp_space(run_len)
        self._drop(self.temp, run_len, note=f"{note_base}: stash to TEMP")

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
            # When freeing TEMP space, we can only remove items from the top of the TEMP stack.
            # Pop a batch of items (top-to-bottom) and extend the hand accordingly.  The
            # orientation of the items will be preserved when they are dropped back onto
            # the source stack via _drop.
            batch = self.temp.pop_batch(move_n)
            self.hand.extend(batch)
            self._record("pick", self.temp.name, move_n, batch, note="Freeing TEMP space (early offload)")
            # Drop onto the chosen stack
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
                # Pop items from the top of TEMP.  We cannot physically access the bottom
                # of the stack, so we always remove from the top.  The order returned is
                # top-to-bottom.  When pushing back to the source stack via _drop, the
                # orientation will be restored.
                batch = self.temp.pop_batch(move_n)
                # Extend the hand with the batch in the order returned.  _drop will
                # internally reverse the order as needed.
                self.hand.extend(batch)
                self._record("pick", self.temp.name, move_n, batch, note="Proactive TEMP reduction")
                self._drop(best_stack, move_n, note="Early temp return (reduces future costs)")

    # --- Generic multi-segment one-pick helper ---
    def _one_pick_multiseg(self, stack: 'StackState', is_deliv: Callable[[Sat], bool],
                           remaining_limit: int, note_prefix: str) -> bool:
        if len(stack.items) == 0:
            return False
        max_k = min(self.hand_capacity - len(self.hand), len(stack.items))
        if max_k <= 0:
            return False
        bestK = 0
        best_deliv = 0
        best_blockers: Optional[int] = None
        temp_free = self.temp.space_left()
        # Choose a feasible prefix that fits TEMP capacity or returns blockers directly. Prefer
        # prefixes with more deliverables; when equal, choose the one with fewer blockers. If
        # deliverables and blockers are equal, prefer the shorter prefix (smaller K). This
        # avoids unnecessarily lifting extra blockers beyond the last deliverable.
        for K in range(1, max_k + 1):
            prefix = stack.items[:K]
            d = sum(1 for s in prefix if is_deliv(s))
            # Skip prefixes that deliver nothing or exceed the remaining limit
            if d == 0 or d > remaining_limit:
                continue
            b = K - d
            # Determine if any deliverables remain below this prefix.  If none remain,
            # blockers can be returned directly to the stack rather than TEMP.  If there are
            # deliverables deeper, ensure there is space in TEMP for the blockers.
            deliverable_deeper = any(is_deliv(sat) for sat in stack.items[K:])
            if b > 0 and deliverable_deeper and b > temp_free:
                # Attempt to free space in TEMP for this prefix's blockers
                self._ensure_temp_space(b)
                if b > self.temp.space_left():
                    # Still insufficient space; skip this prefix
                    continue
            # Evaluate this feasible prefix; prefer more deliverables, then fewer blockers,
            # then smaller K
            if (d > best_deliv or
                (d == best_deliv and (best_blockers is None or b < best_blockers)) or
                (d == best_deliv and best_blockers is not None and b == best_blockers and K < bestK)):
                bestK = K
                best_deliv = d
                best_blockers = b
        # If no feasible prefix found, bail out.
        if bestK == 0:
            return False
        # Resolve None for best_blockers to 0 for numeric comparisons and logging
        best_blockers_val = best_blockers or 0
        # Check if we need TEMP space for the chosen prefix.  If blockers must be stashed and
        # deliverables remain deeper, ensure there is space in TEMP; attempt to free it if
        # necessary.  If still not enough space, bail out.
        deliverable_deeper_best = any(is_deliv(sat) for sat in stack.items[bestK:])
        if best_blockers_val > 0 and deliverable_deeper_best:
            if self.temp.space_left() < best_blockers_val:
                self._ensure_temp_space(best_blockers_val)
                if self.temp.space_left() < best_blockers_val:
                    return False  # still cannot do it
        # Perform the single pick
        self._pick(stack, bestK, note=f"{note_prefix}: one-pick multi-seg | {best_blockers_val} blocker(s) + {best_deliv} deliverable(s)")
        # Build last-to-first flags from the picked slice (which is at the end of hand)
        picked_slice = self.hand[-bestK:]  # top->bottom order
        last_to_first_flags = [is_deliv(s) for s in picked_slice[::-1]]  # from deepest to shallowest
        # Determine if any deliverables remain in this stack after the pick.  If none remain,
        # we will return blockers back to this stack instead of stashing them to TEMP.
        location_complete = not any(is_deliv(sat) for sat in stack.items)
        # Emit alternating drops following LIFO.  When run_type is False (blocker), drop to
        # the original stack if location_complete, otherwise to TEMP.
        run_type = last_to_first_flags[0]  # True=deliverable, False=blocker
        run_len = 1
        for flag in last_to_first_flags[1:]:
            if flag == run_type:
                run_len += 1
            else:
                # Emit the accumulated run before switching types
                if run_type:
                    # Deliverable run goes to destination
                    target = self.dest
                    note = "Deliver run (one-pick multi-seg)"
                    self._drop(target, run_len, note=note)
                else:
                    # Blocker run
                    # Determine if we can safely return blockers directly to the source stack.
                    # In Mode B we must also ensure the current top of the stack is not a cleared satellite.
                    if location_complete and stack.space_left() >= run_len and not (
                        self._mode_B_active and stack.top() and stack.top().cleared
                    ):
                        # Enough space to return blockers directly to the source stack
                        target = stack
                        note = "Return blockers to stack (one-pick multi-seg)"
                        self._drop(target, run_len, note=note)
                    else:
                        # Prefer to offload blockers directly to another safe stack or stash to TEMP
                        self._drop_blockers_prefer_offload(stack, run_len, note_base="Blockers (one-pick multi-seg)")
                # Reset for the next run
                run_type = flag
                run_len = 1
        # Handle the final run
        if run_type:
            target = self.dest
            note = "Deliver run (one-pick multi-seg)"
            self._drop(target, run_len, note=note)
        else:
            # For blockers, determine if we can return them to the source stack.
            # In Mode B we must avoid dropping on a cleared satellite.
            if location_complete and stack.space_left() >= run_len and not (
                self._mode_B_active and stack.top() and stack.top().cleared
            ):
                target = stack
                note = "Return blockers to stack (one-pick multi-seg)"
                self._drop(target, run_len, note=note)
            else:
                # Prefer to offload blockers directly to another safe stack or stash to TEMP
                self._drop_blockers_prefer_offload(stack, run_len, note_base="Blockers (one-pick multi-seg)")
        # Attempt early TEMP return if appropriate
        self.maybe_early_temp_return()
        return True

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
    def _extract_one_from_stack(self, stack: 'StackState') -> bool:
        # If stack top already has remaining targets, batch them.
        top_run = self._contiguous_top_remaining_targets(stack)
        if top_run > 0:
            can_take = min(top_run, self.hand_capacity - len(self.hand))
            self._pick(stack, can_take, note=f"Pick {can_take} target(s) from top (batched)")
            self._drop(self.dest, can_take, note=f"Deliver {can_take} target(s) to DEST (batched)")
            self.maybe_early_temp_return()
            return True

        frt = self._first_remaining_target_in_stack(stack)
        if frt is None:
            return False
        idx, target = frt
        blockers = idx

        # NEW: try one-pick multi-seg (targets interleaved with blockers) up to capacity & TEMP space
        def is_target(s: Sat) -> bool:
            return s.sat in self._remaining_targets()
        remain = len(self._remaining_targets())
        if self._one_pick_multiseg(stack, is_target, remain, note_prefix="Mode A"):
            return True

        # Fallback: perform a contiguous one-lift if multi-seg failed.  We lift the
        # blockers above the first remaining target and the contiguous run of targets.
        # After delivering the targets, we determine whether the blockers can be returned
        # directly to the source stack or must be stashed to TEMP.  This avoids
        # unnecessarily peeling blockers one by one when a single lift suffices.
        run_from_idx = self._contiguous_remaining_from_index(stack, idx)
        total = blockers + run_from_idx
        if total <= self.hand_capacity:
            # If blockers exist and deliverables remain deeper, we may need TEMP space to
            # stash them.  Attempt to free space if necessary.  When no deliverables
            # remain deeper, blockers can be returned directly to the stack, so TEMP
            # space is not required.
            if blockers > 0 and self.temp.space_left() < blockers:
                self._ensure_temp_space(blockers)
            if blockers == 0 or self.temp.space_left() >= blockers:
                # Pick blockers plus contiguous targets
                self._pick(stack, total, note=f"Lift {blockers} blocker(s) + {run_from_idx} target(s) (one-lift)")
                # Drop the contiguous targets to the destination
                if run_from_idx > 0:
                    self._drop(self.dest, run_from_idx, note=f"Drop {run_from_idx} target(s) to DEST (one-lift)")
                # Decide whether to return blockers to the stack or stash them in TEMP.
                if blockers > 0:
                    # Determine if any remaining targets remain deeper in this stack
                    has_deeper_targets = any(s.sat in self._remaining_targets() for s in stack.items)
                    # In Mode B, avoid returning blockers to the stack when its current top is a cleared satellite
                    if (
                        not has_deeper_targets
                        and stack.space_left() >= blockers
                        and not (
                            self._mode_B_active and stack.top() and stack.top().cleared
                        )
                    ):
                        # Return blockers to original stack
                        self._drop(stack, blockers, note="Return blockers to stack (one-lift)")
                    else:
                        # Prefer to offload blockers directly to another safe stack or TEMP
                        self._drop_blockers_prefer_offload(stack, blockers, note_base="Blockers (one-lift)")
                # Attempt an early TEMP return if appropriate
                self.maybe_early_temp_return()
                return True

        # Peel blockers path.  We peel blockers one batch at a time but re-evaluate after
        # each peel whether the remaining blockers plus the contiguous run of targets can
        # be extracted in a single lift.  If so, we switch to the one-pick extraction to
        # avoid unnecessary TEMP moves.
        remaining = blockers
        while remaining > 0:
            # Check if the remaining blockers and their contiguous targets now fit
            # in a single hand.  If they do, delegate to _extract_one_from_stack again
            # to perform a one-pick multi-seg or one-lift extraction instead of further peeling.
            frt2 = self._first_remaining_target_in_stack(stack)
            if frt2:
                idx2, _ = frt2
                run2 = self._contiguous_remaining_from_index(stack, idx2)
                # If the current blockers (idx2) plus the contiguous run of targets (run2)
                # fit within hand capacity, then perform the extraction in one lift.
                if idx2 + run2 <= self.hand_capacity:
                    # Recursively call extraction to handle this case.  The recursive call
                    # will not enter the peel path again as it will satisfy the one-pick
                    # conditions.
                    return self._extract_one_from_stack(stack)
            # Otherwise continue peeling blockers
            chunk = min(self.hand_capacity, remaining)
            # Ensure TEMP has enough space for this chunk; free space if necessary
            if self.temp.space_left() < chunk:
                self._ensure_temp_space(chunk)
            # Build a batch to peel blockers from this stack and potentially other stacks
            to_pick: List[Tuple[StackState,int]] = [(stack, chunk)]
            space_left = self.hand_capacity - chunk
            if space_left > 0:
                # Consider peeling blockers from other stacks to fill the hand while
                # respecting TEMP capacity and to reduce future moves.
                next_cands = [s for s in self._candidate_stacks() if s is not stack]
                scored = []
                for s2 in next_cands:
                    frt3 = self._first_remaining_target_in_stack(s2)
                    if frt3:
                        idx3, _ = frt3
                        if idx3 > 0:
                            scored.append((idx3, s2))
                scored.sort(key=lambda x: x[0])
                for idx3, s2 in scored:
                    if space_left <= 0:
                        break
                    take = min(space_left, idx3)
                    if take > 0 and self.temp.space_left() >= take:
                        to_pick.append((s2, take))
                        space_left -= take
            total = 0
            for s_pick, k in to_pick:
                self._pick(s_pick, k, note="Peel blockers (batched)")
                total += k
                if s_pick is stack:
                    remaining -= k
            # Prefer to offload peeled blockers directly to another safe stack or TEMP
            self._drop_blockers_prefer_offload(stack, total, note_base="Peeled blockers (batched)")
        # After peeling blockers completely, attempt to pick the next target(s).  If there is at
        # least one target at the top, pick and deliver it.  Otherwise, if the top is still a
        # blocker (possible when TEMP offloads have placed blockers back onto this stack), peel
        # one more blocker and recurse until the target is exposed.
        top_run2 = self._contiguous_top_remaining_targets(stack)
        if top_run2 > 0:
            take2 = min(top_run2, self.hand_capacity - len(self.hand))
            self._pick(stack, take2, note=f"Pick {take2} target(s) from top")
            self._drop(self.dest, take2, note=f"Drop {take2} target(s) to DEST")
            self.maybe_early_temp_return()
            return True
        else:
            # Still a blocker at top; remove one more blocker and continue extraction
            # Ensure space in TEMP if needed
            if self.temp.space_left() < 1:
                self._ensure_temp_space(1)
                if self.temp.space_left() < 1:
                    # Cannot peel more; bail out
                    return False
            # Peel one blocker from this stack
            self._pick(stack, 1, note="Peel blocker (fallback)")
            self._drop(self.temp, 1, note="Stash blocker (fallback)")
            # Continue extraction recursively
            return self._extract_one_from_stack(stack)

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

    # New improved Mode B planner using general combination of picks across stacks.
    # This definition overrides the earlier Mode B implementations by being defined later in the class.
    def plan_mode_B(self, count: int = 28):
        """
        Execute Mode B: deliver up to `count` cleared satellites using a greedy batched strategy.

        At each iteration the planner considers picking a prefix from each source stack. A prefix
        may consist solely of cleared satellites (a top-run) or include blockers to reach deeper
        cleared satellites. The planner enumerates all combinations of choosing zero or one
        prefix from each source, subject to hand capacity and TEMP capacity constraints, and
        selects the combination that yields the most deliverable satellites. Ties are broken in
        favour of fewer total blockers. The selected prefixes are picked in a single hand and
        dropped in one run, delivering cleared sats to DEST and stashing blockers to TEMP.

        If no deliverable combination is found, the planner falls back to multi-seg or single-target
        extraction heuristics.
        """
        self._mode_B_active = True
        delivered = 0
        while delivered < count:
            remaining_target = count - delivered
            # Build prefix options for each stack
            stack_opts: List[List[Tuple[int,int,int,List[bool],StackState]]] = []
            for s in self.sources:
                opts: List[Tuple[int,int,int,List[bool],StackState,float]] = []
                max_k = min(self.hand_capacity, len(s.items))
                for K in range(1, max_k + 1):
                    prefix = s.items[:K]
                    # Determine which sats in the prefix can be delivered (cleared and not yet delivered)
                    flags = [x.cleared and x.sat not in {d.sat for d in self.dest.items} for x in prefix]
                    d_count = sum(1 for f in flags if f)
                    if d_count == 0:
                        # Skip prefixes that yield zero deliverable sats
                        continue
                    # Do not pick more deliverables than needed
                    if d_count > remaining_target:
                        continue
                    b_count = K - d_count
                    # Compute a score for this prefix that penalizes blockers. Use a ratio of deliverables
                    # over squared blockers to strongly discourage pulls with many blockers relative
                    # to deliverables. This encourages the planner to prefer combinations that
                    # maximize delivered sats per blocked sat.
                    score = d_count / ((b_count + 1) ** 2)
                    opts.append((K, d_count, b_count, flags, s, score))
                stack_opts.append(opts)
            # Determine best combination across stacks
            best_combo: Optional[List[Tuple[int,int,int,List[bool],StackState,float]]] = None
            best_score = -1.0
            best_del = 0
            best_blk: Optional[int] = None
            from itertools import product
            candidates: List[List[Optional[Tuple[int,int,int,List[bool],StackState,float]]]] = [opts + [None] for opts in stack_opts]
            for choices in product(*candidates):
                combo = [c for c in choices if c is not None]
                if not combo:
                    continue
                total_items = sum(c[0] for c in combo)
                if total_items > self.hand_capacity:
                    continue
                total_del = sum(c[1] for c in combo)
                if total_del == 0:
                    continue
                # Do not deliver more than remaining_target satellites
                if total_del > remaining_target:
                    continue
                total_blk = sum(c[2] for c in combo)
                # Sum scores from individual prefixes; higher is better
                total_score = sum(c[5] for c in combo)
                # Select the combination with the highest total score. Break ties by
                # choosing more deliverables and then fewer blockers.
                if (total_score > best_score or
                    (abs(total_score - best_score) < 1e-9 and (total_del > best_del or
                                                             (total_del == best_del and (best_blk is None or total_blk < best_blk))))):
                    best_combo = combo
                    best_score = total_score
                    best_del = total_del
                    best_blk = total_blk
            if best_combo:
                # Build picks list and flags list preserving source order
                picks: List[Tuple[StackState,int]] = []
                flags: List[bool] = []
                for s in self.sources:
                    entry = None
                    for c in best_combo:
                        if c[4] is s:
                            entry = c
                            break
                    if entry:
                        # Tuple layout: (K, d_count, b_count, flags, stack, score)
                        K, d_count, b_count, flgs, _, _ = entry
                        picks.append((s, K))
                        flags.extend(flgs)
                # Ensure TEMP space for blockers
                total_blk_needed = sum(c[2] for c in best_combo)
                if total_blk_needed > 0 and self.temp.space_left() < total_blk_needed:
                    # Attempt to free sufficient TEMP space. If unsuccessful, skip this combo.
                    self._ensure_temp_space(total_blk_needed)
                    # After freeing, if there is still not enough space, treat as failure.
                    if self.temp.space_left() < total_blk_needed:
                        best_combo = None
                        best_del = 0
                        best_blk = None
                        # break out to fallback path
                        # Note: clearing best_combo will cause the following code to proceed to fallback
                        
                # Execute batched pick and drop
                self._multi_stack_pick_and_drop(picks, flags, note_prefix="Mode B")
                delivered += min(best_del, remaining_target)
                # Proactively return TEMP items if threshold exceeded
                if len(self.temp.items) >= self.early_temp_threshold:
                    self.maybe_early_temp_return()
                continue
            # Fallback: no deliverable combination found
            fallback_candidates = []
            for s in self.sources:
                for idx, sat in enumerate(s.items):
                    if sat.cleared and sat.sat not in {x.sat for x in self.dest.items}:
                        fallback_candidates.append((idx, s, sat))
                        break
            if not fallback_candidates:
                break
            def deeper_cleared_fallback(s: 'StackState', i: int) -> int:
                return sum(1 for x in s.items[i+1:] if x.cleared and x.sat not in {y.sat for y in self.dest.items})
            fallback_candidates.sort(key=lambda t: (t[0], -deeper_cleared_fallback(t[1], t[0])))
            idx, stack, sat = fallback_candidates[0]
            def is_cleared_fallback(x: Sat) -> bool:
                return x.cleared and x.sat not in {y.sat for y in self.dest.items}
            if self._one_pick_multiseg(stack, is_cleared_fallback, remaining_target, note_prefix="Mode B"):
                delivered = len(self.dest.items)
                continue
            # Final fallback: single-target extraction
            self.target_ids = {sat.sat}
            self._extract_one_from_stack(stack)
            self.target_ids = set()
            delivered = len(self.dest.items)
        # End while
        self._mode_B_active = False
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

    # Override plan_mode_B with improved implementation. This new definition
    # supersedes the earlier version and provides a more flexible pick strategy
    # that can combine top-run picks and multi-segment picks even near the end
    # of the extraction when only a few deliveries remain.

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
