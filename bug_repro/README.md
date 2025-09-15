# Bug Reproduction CSVs

These stack files reproduce the issue where blockers are offloaded from one source stack onto another and later moved to TEMP, causing an extra lift.

Run Mode B:
```
python3 run_lift_plan_from_csv_v1p5a.py --csv bug_repro/stack1.csv bug_repro/stack2.csv bug_repro/stack3.csv bug_repro/stack4.csv --mode B --count 15 --out repro_planB
```

Run Mode A with the same targets:
```
python3 run_lift_plan_from_csv_v1p5a.py --csv bug_repro/stack1.csv bug_repro/stack2.csv bug_repro/stack3.csv bug_repro/stack4.csv --mode A --targets-json bug_repro/targets_modeA.json --out repro_planA
```

Inspect `repro_planB_actions_compact.csv` to see blockers from Source4 first dropped onto Source1 and later moved to TEMP.
