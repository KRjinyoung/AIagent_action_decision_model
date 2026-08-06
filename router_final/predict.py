"""
Run the trained router on new session records.

Usage:
    python predict.py --input new_sessions.jsonl --model router_model.pkl --output preds.csv

Input format: a .jsonl file where each line is one session record with the
same shape as train.jsonl (id, session_meta, history, current_prompt).
Output: CSV with columns id,group (group is one of group1..group5).
"""
import argparse
import sys

import joblib
import pandas as pd

import features as feat


def predict(model_path, input_path):
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    records = feat.load_records(input_path)
    df = feat.build_feature_frame(records)

    preds = pipeline.predict(df)
    out = pd.DataFrame({"id": df["id"], "group": preds})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to a .jsonl file of session records")
    ap.add_argument("--model", default="router_model.pkl")
    ap.add_argument("--output", default=None, help="if omitted, prints CSV to stdout")
    args = ap.parse_args()

    out = predict(args.model, args.input)

    if args.output:
        out.to_csv(args.output, index=False, encoding="utf-8")
        print(f"wrote {len(out)} predictions to {args.output}")
    else:
        out.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
