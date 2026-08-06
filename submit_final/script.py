"""Integrated inference: router (group1~5) + per-group specialist models.

Evaluation server runs this file from the submit root:
    ./data/test.jsonl + ./data/sample_submission.csv  ->  ./output/submission.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

# --- numpy 2.x pickle compatibility -----------------------------------------
# Some model pickles were written under numpy>=2 where the core module is
# "numpy._core"; the evaluation server runs numpy 1.26.4 ("numpy.core").
try:
    import numpy._core  # noqa: F401
except ImportError:
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric
    import numpy.core.umath
    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric
    sys.modules["numpy._core.umath"] = numpy.core.umath
    sys.modules["numpy._core._multiarray_umath"] = numpy.core._multiarray_umath

import joblib
import numpy as np
import pandas as pd

import g1_feature
import g2_features
import g4_features
import router_features

# g1/g2/g3_predict import their g{N}_features modules directly;
# all current bundles are plain sklearn/xgboost objects, so no
# `features` module alias is needed for unpickling.
import g1_predict
import g2_predict
import g3_predict

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DATA_DIR = HERE / "data"
OUT_PATH = HERE / "output" / "submission.csv"

GROUP4_MAX_LENGTH = 480


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- router ------------------------------------------------------------------

def predict_router(records):
    bundle = joblib.load(MODEL_DIR / "router_model.pkl")
    df = router_features.build_feature_frame(records)
    return list(bundle["pipeline"].predict(df))


# --- group1: read_file / grep_search / list_directory / glob_pattern ----------
# g1_predict: compressed XGB+ExtraTrees base, plus grep/read linear + RoBERTa
# specialists applied through utility gates on warm sim records
# (model/group1/manifest.json drives paths, policy, and thresholds).

def predict_group1(records):
    if not records:
        return []
    return [str(a) for a in g1_predict.predict(records)]


# --- group2: ask_user / plan_task / web_search ---------------------------------
# structured V3 ensemble (RF/CatBoost + pair specialists) + distilled Ask/Plan
# DistilBERT student + Plan-rescue rule. The full inference logic lives in
# g2_predict.FinalG2Predictor (shipped verbatim from the group2 package).

# g2_predict: HistGradientBoosting + TF-IDF LogisticRegression, stacked by a
# session-GroupKFold-trained LogisticRegression (no language model).

def predict_group2(records):
    if not records:
        return []
    bundle = joblib.load(MODEL_DIR / "group2_model.pkl")
    proba = g2_predict._predict_frame(records, bundle)
    return [str(bundle["classes"][i]) for i in proba.argmax(axis=1)]


# --- group3: run_bash / run_tests / lint_or_typecheck --------------------------
# g3_predict.predict_hybrid: V4 structured (LightGBM+CatBoost+XGBoost, history/AU
# branches) blended with a Context Recovery BERT for "sim" records only.

def predict_group3(records):
    if not records:
        return []
    labels, _scores = g3_predict.predict_hybrid(
        records,
        v4_model=MODEL_DIR / "group3_model.pkl",
        bert_model_dir=MODEL_DIR / "group3_bert",
    )
    return labels


# --- group4: edit_file / write_file / apply_patch (klue/roberta-base) ----------

def predict_group4(records):
    if not records:
        return []
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = MODEL_DIR / "group4"
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tok.truncation_side = "left"  # keep current_prompt (end of text) when truncating
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half()  # fp16 inference is ~4x faster on T4
    model.to(device)
    model.eval()
    batch_size = 64 if device == "cuda" else 16

    id2label = model.config.id2label

    # per-class logit bias shipped with the model (all zeros in this release)
    bias = None
    bias_path = model_dir / "logit_bias.json"
    if bias_path.exists():
        bias_map = json.loads(bias_path.read_text(encoding="utf-8"))
        values = [float(bias_map.get(id2label[i], 0.0)) for i in range(len(id2label))]
        if any(values):
            bias = torch.tensor(values, device=device)

    texts = [g4_features.build_text(r) for r in records]
    preds = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tok(batch, truncation=True, max_length=GROUP4_MAX_LENGTH,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            if bias is not None:
                logits = logits + bias
            preds.extend(id2label[int(i)] for i in logits.argmax(dim=-1).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return preds


# --- submission ----------------------------------------------------------------

def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission columns must start with id, action: {fieldnames}")
    return fieldnames, rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    records = load_jsonl(DATA_DIR / "test.jsonl")
    print(f"test records: {len(records)}")

    print("routing...")
    routed = predict_router(records)

    pred = [None] * len(records)
    specialists = {
        "group1": predict_group1,
        "group2": predict_group2,
        "group3": predict_group3,
        "group4": predict_group4,
    }
    for group, fn in specialists.items():
        idx = [i for i, g in enumerate(routed) if g == group]
        print(f"{group}: {len(idx)} records")
        for i, action in zip(idx, fn([records[i] for i in idx])):
            pred[i] = action

    n_g5 = 0
    for i, g in enumerate(routed):
        if pred[i] is None:  # group5 or any unexpected router label
            pred[i] = "respond_only"
            n_g5 += 1
    print(f"group5(respond_only): {n_g5} records")

    pred_map = {r["id"]: a for r, a in zip(records, pred)}

    sample_path = DATA_DIR / "sample_submission.csv"
    if sample_path.exists():
        fieldnames, rows = load_sample_submission(sample_path)
        missing = 0
        for row in rows:
            action = pred_map.get(row["id"])
            if action is None:
                missing += 1
            else:
                row["action"] = action
        if missing:
            print(f"warning: missing predictions for {missing} ids")
    else:
        fieldnames = ["id", "action"]
        rows = [{"id": r["id"], "action": pred_map[r["id"]]} for r in records]

    save_submission(OUT_PATH, fieldnames, rows)
    print(f"Saved: {OUT_PATH} rows={len(rows)}")


if __name__ == "__main__":
    main()
