import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from lift_planner_v1p5a import read_stack_sql, LiftPlannerV1P5, StackState

# Preset configuration values
DB_URL = "sqlite:///stacks.db"
STACK_TABLES = ["stack1", "stack2", "stack3", "stack4"]
MODE = "B"  # "A" uses target lists, "B" selects best automatically
TARGETS_JSON = ""
TARGETS_CSV = ""
VERIFY_CLEARED = False
COUNT = 28
LOOKAHEAD = 2
BEAM = 3
EARLY_TEMP = 12
OUT = "plan_v1p5"


def build_planner(engine, tables, lookahead, beam, early_temp):
    sources = [read_stack_sql(engine, t, f"Source{i+1}") for i, t in enumerate(tables)]
    temp = StackState("Temp", [], cap=15)
    dest = StackState("Dest", [], cap=10**9)
    return LiftPlannerV1P5(
        sources,
        temp,
        dest,
        hand_capacity=7,
        temp_cap=15,
        lookahead_depth=lookahead,
        beam_width=beam,
        early_temp_threshold=early_temp,
    )


def main():
    engine = create_engine(DB_URL)
    planner = build_planner(engine, STACK_TABLES, LOOKAHEAD, BEAM, EARLY_TEMP)

    try:
        if MODE == "A":
            targets = []
            if TARGETS_JSON:
                with open(TARGETS_JSON) as f:
                    targets = [str(x) for x in json.load(f)]
            elif TARGETS_CSV:
                df = pd.read_csv(TARGETS_CSV)
                cols = [c.strip().lower() for c in df.columns]
                if "sat" not in cols:
                    raise ValueError("targets CSV must have a 'sat' column")
                if {"filled", "interim", "final", "issue"}.issubset(cols) and VERIFY_CLEARED:
                    def as_bool(x):
                        return str(x).strip().lower() in ("true", "t", "1", "yes", "y")

                    df = df[
                        df["filled"].apply(as_bool)
                        & df["interim"].apply(as_bool)
                        & df["final"].apply(as_bool)
                        & ~df["issue"].apply(as_bool)
                    ]
                targets = [str(s) for s in df["sat"].astype(str).tolist()]
            planner.target_ids = set(targets)
            planner.plan_mode_A()
        else:
            planner.plan_mode_B(count=COUNT)

        df2 = planner.get_log_dataframe_compact()
        df2.to_csv(Path(f"{OUT}_actions_compact.csv"), index=False)
        with open(Path(f"{OUT}_summary.json"), "w") as f:
            json.dump(planner.get_summary(), f, indent=2)
    except Exception as e:
        err = str(e)
        err_df = pd.DataFrame(
            [
                {
                    "step": 0,
                    "action": "error",
                    "location": "",
                    "count": 0,
                    "items_top_to_bottom": [],
                    "note": err,
                }
            ]
        )
        err_df.to_csv(Path(f"{OUT}_actions_compact.csv"), index=False)
        with open(Path(f"{OUT}_summary.json"), "w") as f:
            json.dump({"error": err}, f, indent=2)
        print(err)


if __name__ == "__main__":
    main()

