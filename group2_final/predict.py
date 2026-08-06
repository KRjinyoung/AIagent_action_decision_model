#!/usr/bin/env python3
"""
G2(plan_task / ask_user / web_search) 전문가 — 학습 + 추론.

구조: LM 없음(전용 인코더 데이터부족으로 무익). 3 브랜치를 스태킹.
  - TREE : HistGradientBoosting (구조+타겟피처, CAT=OrdinalEncoder)
  - LIN  : TF-IDF(word+char) + OneHot(CAT) + StandardScaler(수치) -> LogisticRegression
  - STACK: [TREE prob, LIN prob] -> LogisticRegression (세션 GroupKFold OOF로 학습)
용량 ~1.3MB. features.py를 학습/추론이 글자 그대로 공유(train/serve skew 방지).

사용:
  학습:  python predict.py train  --train_jsonl data/raw/train.jsonl --labels_csv data/raw/train_labels.csv --out model.pkl
  추론:  python predict.py predict --input_jsonl data/raw/test.jsonl  --model model.pkl --out preds.csv
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FE

HGB_PARAMS = dict(max_iter=500, max_depth=7, learning_rate=0.05, l2_regularization=1.0,
                  random_state=0, early_stopping=True, validation_fraction=0.15)

def _load_records(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def _sess(rid):
    return re.sub(r'-step_?\d+$', '', str(rid))

def _prompts(records):
    return np.array([(r.get("current_prompt") or "") for r in records], dtype=object)

# ---------- transforms (fitted 객체는 bundle에 저장) ----------
def _tree_matrix(df, tier_stats, ord_enc):
    df = FE.apply_tier_stats(df, tier_stats)
    num = df[[c for c in FE.TREE_FEATS if c not in FE.CAT_COLS]].astype(float).values
    cat = ord_enc.transform(df[FE.CAT_COLS].astype(str).values)
    # TREE_FEATS 순서대로 재배치
    cols = {c: i for i, c in enumerate([c for c in FE.TREE_FEATS if c not in FE.CAT_COLS])}
    catcols = {c: i for i, c in enumerate(FE.CAT_COLS)}
    out = np.zeros((len(df), len(FE.TREE_FEATS)))
    for j, c in enumerate(FE.TREE_FEATS):
        out[:, j] = cat[:, catcols[c]] if c in catcols else num[:, cols[c]]
    return out

def _lin_matrix(df, texts, wv, cv, oh, scaler):
    df = FE.apply_tier_stats(df, tier_stats_holder[0]) if tier_stats_holder[0] else df
    return hstack([wv.transform(texts), cv.transform(texts),
                   oh.transform(df[FE.CAT_COLS].astype(str).values),
                   csr_matrix(scaler.transform(df[FE.LIN_NUM].astype(float).values))]).tocsr()

tier_stats_holder = [None]  # _lin_matrix가 z-score 적용 위해 참조

# ---------- train ----------
def train(args):
    records = _load_records(args.train_jsonl)
    # 라벨
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv); id2a = dict(zip(lab["id"].astype(str), lab["action"].astype(str)))
    else:
        id2a = {str(r["id"]): r.get("action") for r in records}
    records = [r for r in records if id2a.get(str(r["id"])) in FE.CLASSES]
    y = np.array([FE.CLASSES.index(id2a[str(r["id"])]) for r in records])
    sess = np.array([_sess(r["id"]) for r in records])
    texts = _prompts(records)
    df = FE.records_to_frame(records)

    tier_stats = FE.fit_tier_stats(df); tier_stats_holder[0] = tier_stats
    dz = FE.apply_tier_stats(df, tier_stats)
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(dz[FE.CAT_COLS].astype(str).values)
    Xtree = _tree_matrix(df, tier_stats, ord_enc)
    sw = compute_sample_weight("balanced", y)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xtree, y, sample_weight=sw)

    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True).fit(texts)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=12000, sublinear_tf=True).fit(texts)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(dz[FE.CAT_COLS].astype(str).values)
    scaler = StandardScaler().fit(dz[FE.LIN_NUM].astype(float).values)
    Xlin = _lin_matrix(df, texts, wv, cv, oh, scaler)
    lin = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced").fit(Xlin, y)

    # 스태커: 세션 GroupKFold OOF 로 base prob 만들어 학습(누수 방지)
    tp = np.zeros((len(y), 3)); lp = np.zeros((len(y), 3))
    for tri, tei in GroupKFold(5).split(Xtree, y, sess):
        swf = compute_sample_weight("balanced", y[tri])
        h = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xtree[tri], y[tri], sample_weight=swf)
        for j, cl in enumerate(h.classes_): tp[tei, cl] = h.predict_proba(Xtree[tei])[:, j]
        l = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced").fit(Xlin[tri], y[tri])
        for j, cl in enumerate(l.classes_): lp[tei, cl] = l.predict_proba(Xlin[tei])[:, j]
    stacker = LogisticRegression(max_iter=3000, C=1.0).fit(np.hstack([tp, lp]), y)

    bundle = dict(hgb=hgb, wv=wv, cv=cv, oh=oh, scaler=scaler, lin=lin, stacker=stacker,
                  ord_enc=ord_enc, tier_stats=tier_stats, classes=FE.CLASSES)
    joblib.dump(bundle, args.out, compress=3)
    print(f"saved {args.out} ({os.path.getsize(args.out)/1024/1024:.2f} MB) | trained on {len(y)} G2 rows")

# ---------- predict ----------
def _predict_frame(records, bundle):
    texts = _prompts(records); df = FE.records_to_frame(records)
    tier_stats_holder[0] = bundle["tier_stats"]
    Xtree = _tree_matrix(df, bundle["tier_stats"], bundle["ord_enc"])
    Xlin = _lin_matrix(df, texts, bundle["wv"], bundle["cv"], bundle["oh"], bundle["scaler"])
    tp = np.zeros((len(records), 3)); lp = np.zeros((len(records), 3))
    for j, cl in enumerate(bundle["hgb"].classes_): tp[:, cl] = bundle["hgb"].predict_proba(Xtree)[:, j]
    for j, cl in enumerate(bundle["lin"].classes_): lp[:, cl] = bundle["lin"].predict_proba(Xlin)[:, j]
    proba = bundle["stacker"].predict_proba(np.hstack([tp, lp]))
    return proba

def predict(args):
    bundle = joblib.load(args.model)
    records = _load_records(args.input_jsonl)
    proba = _predict_frame(records, bundle)
    pred = [bundle["classes"][i] for i in proba.argmax(1)]
    out = pd.DataFrame({"id": [r["id"] for r in records], "action": pred})
    for j, c in enumerate(bundle["classes"]): out[f"p_{c}"] = proba[:, j]
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--train_jsonl", required=True); t.add_argument("--labels_csv", default=None); t.add_argument("--out", default="model.pkl")
    p = sub.add_parser("predict"); p.add_argument("--input_jsonl", required=True); p.add_argument("--model", default="model.pkl"); p.add_argument("--out", default="preds.csv")
    args = ap.parse_args()
    {"train": train, "predict": predict}[args.cmd](args)
