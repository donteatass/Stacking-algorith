
import argparse, json
from pathlib import Path
from lift_planner_v1p5a import read_stack_csv, LiftPlannerV1P5, StackState

def build_planner(csvs, lookahead, beam, early_temp):
    sources = [read_stack_csv(Path(p), f"Source{i+1}") for i,p in enumerate(csvs)]
    temp = StackState("Temp", [], cap=15)
    dest = StackState("Dest", [], cap=10**9)
    return LiftPlannerV1P5(sources, temp, dest, hand_capacity=7, temp_cap=15,
                           lookahead_depth=lookahead, beam_width=beam, early_temp_threshold=early_temp)

def main():
    ap = argparse.ArgumentParser(description="Lift plan v1p5: minimize picks via one-pick multi-segment behind blockers.")
    ap.add_argument("--csv", nargs=4, required=True, help="Four source CSVs (first row is TOP).")
    ap.add_argument("--mode", choices=["A","B"], required=True, help="A: fixed target list; B: choose best 28.")
    ap.add_argument("--targets-json", default="", help="JSON array of sat IDs (Mode A).")
    ap.add_argument("--targets-csv",  default="", help="CSV with a 'sat' column (Mode A).")
    ap.add_argument("--verify-cleared", action="store_true", help="If targets CSV has 5 cols, require cleared.")
    ap.add_argument("--count", type=int, default=28, help="Mode B: how many to deliver (default 28).")
    ap.add_argument("--lookahead", type=int, default=2, help="Lookahead depth (default 2).")
    ap.add_argument("--beam", type=int, default=3, help="Beam width (default 3).")
    ap.add_argument("--early-temp", type=int, default=12, help="Early TEMP return threshold (default 12).")
    ap.add_argument("--out", default="plan_v1p5", help="Output prefix for CSV/JSON.")
    args = ap.parse_args()

    planner = build_planner(args.csv, args.lookahead, args.beam, args.early_temp)

    targets = []
    if args.mode == "A":
        if not args.targets_json and not args.targets_csv:
            ap.error("Mode A requires --targets-json or --targets-csv")
        if args.targets_json:
            with open(args.targets_json) as f:
                targets = [str(x) for x in json.load(f)]
        else:
            import pandas as pd
            df = pd.read_csv(args.targets_csv)
            cols = [c.strip().lower() for c in df.columns]
            if "sat" not in cols:
                ap.error("targets CSV must have a 'sat' column")
            if {"filled","interim","final","issue"}.issubset(cols) and args.verify_cleared:
                def as_bool(x): return str(x).strip().lower() in ("true","t","1","yes","y")
                df = df[df["filled"].apply(as_bool) & df["interim"].apply(as_bool) & df["final"].apply(as_bool) & ~df["issue"].apply(as_bool)]
            targets = [str(s) for s in df["sat"].astype(str).tolist()]
        planner.target_ids = set(targets)
        planner.plan_mode_A()
    else:
        planner.plan_mode_B(count=args.count)

    # Raw log
    '''df = planner.get_log_dataframe()
    df.to_csv(Path(f"{args.out}_actions.csv"), index=False)'''

    # Compact log
    df2 = planner.get_log_dataframe_compact()
    df2.to_csv(Path(f"{args.out}_actions_compact.csv"), index=False)

    # Summary
    # Note: json is already imported at module scope. Avoid re-importing it
    # inside this function, which would create a local shadow and cause
    # UnboundLocalError when referenced earlier in the function. Use the
    # module-level json import instead.
    with open(Path(f"{args.out}_summary.json"), "w") as f:
        json.dump(planner.get_summary(), f, indent=2)

if __name__ == "__main__":
    main()
