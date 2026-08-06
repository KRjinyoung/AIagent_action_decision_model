#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Group3 V4 + Context Recovery BERT hybrid inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import g3_features as F


CLASSES = ["lint_or_typecheck", "run_bash", "run_tests"]

V4_WEIGHT = 0.575
BERT_WEIGHT = 0.425


def _align_probability(
    probability: np.ndarray,
    model_classes: Sequence[Any],
) -> np.ndarray:
    output = np.zeros((len(probability), len(CLASSES)), dtype=np.float64)

    for source_index, class_value in enumerate(model_classes):
        if isinstance(class_value, str):
            target_index = CLASSES.index(class_value)
        else:
            target_index = int(class_value)
        output[:, target_index] = probability[:, source_index]

    return output


def _branch_probability(
    frame: pd.DataFrame,
    branch: Mapping[str, Any],
    categorical_state: Mapping[str, Any],
) -> np.ndarray:
    lgb_frame = frame.copy()
    xgb_frame = frame.copy()
    cat_frame = frame.copy()

    for column in F.CAT_COLS:
        categories = categorical_state[column]["categories"]

        lgb_frame[column] = pd.Categorical(
            lgb_frame[column].astype(str),
            categories=categories,
        )
        xgb_frame[column] = pd.Categorical(
            xgb_frame[column].astype(str),
            categories=categories,
        )
        cat_frame[column] = cat_frame[column].astype(str)

    lgb_model = branch["lightgbm"]
    cat_model = branch["catboost"]
    xgb_model = branch["xgboost"]

    lgb_probability = _align_probability(
        np.asarray(lgb_model.predict_proba(lgb_frame), dtype=np.float64),
        lgb_model.classes_,
    )
    cat_probability = _align_probability(
        np.asarray(cat_model.predict_proba(cat_frame), dtype=np.float64),
        cat_model.classes_,
    )
    xgb_probability = _align_probability(
        np.asarray(xgb_model.predict_proba(xgb_frame), dtype=np.float64),
        xgb_model.classes_,
    )

    return (
        lgb_probability
        + cat_probability
        + xgb_probability
    ) / 3.0


def _prepare_v4_base(
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    frame = F.build_base_dataframe(records, has_label=False)
    frame = F.add_manual_text_features(frame)

    for column in ("last_action_name", "last_action_result"):
        frame[column] = frame[column].fillna("NONE").astype(str)

    state = bundle["preprocess_state"]

    frame = F.apply_group_relative_dict(
        frame,
        state["budget_stats"],
        "budget_tokens_remaining",
        "user_tier",
        "budget_rel_tier",
    )
    frame = F.apply_group_relative_dict(
        frame,
        state["loc_stats"],
        "loc",
        "workspace_template_id",
        "loc_rel_template",
    )

    for column in F.CAT_COLS:
        categorical = state["categorical"][column]
        fallback = str(categorical["fallback"])
        allowed = set(str(value) for value in categorical["categories"])

        values = frame[column].fillna(fallback).astype(str)
        frame[column] = values.where(values.isin(allowed), fallback)

    return frame[state["feature_columns"]].copy()


def _add_text_features(
    base: pd.DataFrame,
    texts: Sequence[str],
    vectorizer: Any,
    svd: Any,
) -> pd.DataFrame:
    dense = svd.transform(vectorizer.transform(list(texts)))
    output = base.copy()

    for index in range(dense.shape[1]):
        output[f"txt_{index}"] = dense[:, index]

    return output


def predict_v4_scores(
    records: Sequence[Mapping[str, Any]],
    model_path: str | Path,
) -> np.ndarray:
    bundle = joblib.load(model_path)

    if list(bundle["classes"]) != CLASSES:
        raise ValueError(
            f"V4 class order mismatch: {bundle['classes']} != {CLASSES}"
        )

    base = _prepare_v4_base(records, bundle)

    history_frame = _add_text_features(
        base,
        F.build_history_texts(records),
        bundle["history_vectorizer"],
        bundle["history_svd"],
    )
    current_frame = _add_text_features(
        base,
        F.build_current_texts(records),
        bundle["current_vectorizer"],
        bundle["current_svd"],
    )

    categorical_state = bundle["preprocess_state"]["categorical"]

    history_global = _branch_probability(
        history_frame,
        bundle["history_global"],
        categorical_state,
    )
    current_global = _branch_probability(
        current_frame,
        bundle["current_global"],
        categorical_state,
    )
    current_au = _branch_probability(
        current_frame,
        bundle["current_au"],
        categorical_state,
    )

    sources = np.asarray(
        [F.source_tag(str(record["id"])) for record in records],
        dtype=object,
    )

    raw_probability = history_global.copy()
    au_mask = sources == "au"

    if au_mask.any():
        au_weight = float(bundle.get("au_weight", 0.5))
        raw_probability[au_mask] = (
            (1.0 - au_weight) * current_global[au_mask]
            + au_weight * current_au[au_mask]
        )

    class_weights = np.asarray(
        bundle["class_weights"],
        dtype=np.float64,
    )
    return raw_probability * class_weights[None, :]


class EncodedDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[torch.Tensor]]) -> None:
        self.encodings = encodings
        self.length = len(next(iter(encodings.values())))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            key: values[index]
            for key, values in self.encodings.items()
        }


def _encode(
    tokenizer: Any,
    texts: Sequence[str],
) -> Dict[str, List[torch.Tensor]]:
    encoded = tokenizer(
        list(texts),
        truncation=True,
        max_length=F.BERT_MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )
    return {
        key: [
            torch.tensor(row, dtype=torch.long)
            for row in rows
        ]
        for key, rows in encoded.items()
    }


@torch.inference_mode()
def predict_bert_probability(
    records: Sequence[Mapping[str, Any]],
    model_dir: str | Path,
    batch_size: int,
) -> np.ndarray:
    model_dir = Path(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )

    # GPU: load the saved FP16 weights as FP16.
    # CPU: load as FP32 because many CPU kernels do not support FP16 well.
    load_dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=load_dtype,
    )

    configured_order = [
        model.config.id2label.get(index, model.config.id2label.get(str(index)))
        for index in range(len(CLASSES))
    ]
    if configured_order != CLASSES:
        raise ValueError(
            f"BERT class order mismatch: {configured_order} != {CLASSES}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = EncodedDataset(
        _encode(
            tokenizer,
            F.build_context_texts(records),
        )
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    outputs: List[np.ndarray] = []

    for batch in loader:
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            enabled=device.type == "cuda",
        ):
            logits = model(**inputs).logits

        outputs.append(
            torch.softmax(logits.float(), dim=-1).cpu().numpy()
        )

    return np.concatenate(outputs, axis=0)


def predict_hybrid(
    records: Sequence[Mapping[str, Any]],
    v4_model: str | Path,
    bert_model_dir: str | Path,
    batch_size: int = 48,
) -> tuple[List[str], np.ndarray]:
    v4_scores = predict_v4_scores(records, v4_model)

    final_scores = v4_scores.copy()
    sources = np.asarray(
        [F.source_tag(str(record["id"])) for record in records],
        dtype=object,
    )
    sim_indices = np.flatnonzero(sources == "sim")

    if len(sim_indices):
        sim_records = [records[index] for index in sim_indices]
        bert_probability = predict_bert_probability(
            sim_records,
            bert_model_dir,
            batch_size,
        )
        final_scores[sim_indices] = (
            V4_WEIGHT * v4_scores[sim_indices]
            + BERT_WEIGHT * bert_probability
        )

    prediction_ids = final_scores.argmax(axis=1)
    labels = [CLASSES[index] for index in prediction_ids]

    return labels, final_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--v4-model",
        type=Path,
        default=Path("model.pkl"),
    )
    parser.add_argument(
        "--bert-model",
        type=Path,
        default=Path("context_bert"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = F.load_jsonl(args.input_jsonl)
    labels, _ = predict_hybrid(
        records,
        args.v4_model,
        args.bert_model,
        args.batch_size,
    )

    output = pd.DataFrame(
        {
            "id": [str(record["id"]) for record in records],
            "action": labels,
        }
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"rows={len(output)}")
    print(f"output={args.output_csv.resolve()}")
    print(f"SIM hybrid weights: V4={V4_WEIGHT}, BERT={BERT_WEIGHT}")
    print("AU policy: V4 only")
    print("BERT GPU inference dtype: FP16")


if __name__ == "__main__":
    main()
