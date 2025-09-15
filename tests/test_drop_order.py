import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from lift_planner_v1p5a import StackState, Sat, LiftPlannerV1P5


def make_sat(id: str) -> Sat:
    return Sat(id, True, True, True, False)


def test_drop_preserves_order():
    source = StackState('loc1', [make_sat('A'), make_sat('B'), make_sat('C'), make_sat('D')])
    target = StackState('loc2', [])
    temp = StackState('temp', [])
    planner = LiftPlannerV1P5([source, target], temp, StackState('dest', []))
    planner._pick(source, 4)
    planner._drop(target, 4)
    assert [sat.sat for sat in target.items] == ['A', 'B', 'C', 'D']
