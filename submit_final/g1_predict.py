#!/usr/bin/env python3
"""Inference entrypoint for the size-constrained Group1 role-specialist model.

The package is a directory because the transformer must remain in Hugging Face
format. The public entrypoint is ``predict(records)``; the CLI can verify the
artifact set or predict a JSONL file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

import g1_feature as F


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "model" / "group1"
MANIFEST_NAME = "manifest.json"
OUTCOMES = ("damage", "rescue", "wrong_to_wrong")


def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    spec.loader.exec_module(module)
    return module


R = _load_local_module("g1_final_role_features", DEFAULT_MODEL_DIR / "role_features.py")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _aligned_probabilities(model: Any, values: Any, labels: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = list(model.classes_)
    output = np.zeros((len(raw), len(labels)), dtype=np.float64)
    for source_index, label in enumerate(classes):
        if isinstance(label, (int, np.integer)):
            target_index = int(label)
        else:
            target_index = labels.index(str(label))
        output[:, target_index] = raw[:, source_index]
    return output


def _outcome_probabilities(model: Any, rows: list[Any]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(rows), dtype=np.float64)
    classes = list(model.classes_)
    output = np.zeros((len(rows), len(OUTCOMES)), dtype=np.float64)
    for index, label in enumerate(classes):
        output[:, OUTCOMES.index(str(label))] = raw[:, index]
    return output


class Group1Predictor:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.manifest = _read_json(self.model_dir / MANIFEST_NAME)
        self.labels = list(self.manifest["labels"])
        self._check_required_files()

        components = self.manifest["components"]
        self.structured = joblib.load(self.model_dir / components["structured"]["path"])
        self.linear = joblib.load(self.model_dir / components["linear"]["path"])
        self.linear_gate = joblib.load(self.model_dir / components["gates"]["linear_path"])
        self.bert_gate = joblib.load(self.model_dir / components["gates"]["bert_path"])
        self.bert_path = self.model_dir / components["bert"]["path"]
        self.max_length = int(components["bert"].get("max_length", 384))
        self._bert_cache: dict[str, Any] | None = None

    def _check_required_files(self) -> None:
        missing = [
            relative
            for relative in self.manifest.get("required_files", [])
            if not (self.model_dir / relative).is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "g1_final is waiting for trained SSH artifacts:\n" + formatted
            )

    def _structured_probabilities(self, records: list[dict[str, Any]]) -> np.ndarray:
        frame = pd.DataFrame(records)
        frame["action"] = F.TARGET_ACTIONS[0]
        feature_frame = F.build_features(frame)
        if not {"models", "x_cols", "label_encoder"}.issubset(self.structured):
            raise ValueError("structured_model.joblib is not the compact Group1 base bundle")
        bundle_labels = list(self.structured["label_encoder"].classes_)
        members = []
        for _, model in self.structured["models"]:
            members.append(
                _aligned_probabilities(
                    model,
                    feature_frame[self.structured["x_cols"]],
                    bundle_labels,
                )
            )
        probabilities = np.mean(members, axis=0)
        aligned = np.zeros((len(records), len(self.labels)), dtype=np.float64)
        for index, label in enumerate(self.labels):
            aligned[:, index] = probabilities[:, bundle_labels.index(label)]
        return aligned

    def _load_bert(self) -> dict[str, Any]:
        if self._bert_cache is not None:
            return self._bert_cache
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.bert_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
        if device.type == "cpu":
            model = model.float()
        model.to(device)
        model.eval()
        id2label = {int(key): value for key, value in model.config.id2label.items()}
        expected = set(R.PAIR_SPECS["grep_read"])
        if set(id2label.values()) != expected:
            raise ValueError(f"BERT labels must be {sorted(expected)}, got {id2label}")
        self._bert_cache = {
            "torch": torch,
            "device": device,
            "tokenizer": tokenizer,
            "model": model,
            "id2label": id2label,
        }
        return self._bert_cache

    def _bert_probabilities(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        cache = self._load_bert()
        torch = cache["torch"]
        device = cache["device"]
        batch_size = 64 if device.type == "cuda" else 8
        output = np.zeros((len(texts), 2), dtype=np.float64)
        labels = [cache["id2label"][index] for index in range(2)]
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = cache["tokenizer"](
                    batch,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                probability = torch.softmax(cache["model"](**encoded).logits, dim=1)
                output[start : start + len(batch)] = probability.float().cpu().numpy()
        return labels, output

    def predict(self, records: Iterable[dict[str, Any]]) -> list[str]:
        rows = list(records)
        if not rows:
            return []

        base_probs = self._structured_probabilities(rows)
        base_ids = base_probs.argmax(axis=1)
        base = np.asarray([self.labels[index] for index in base_ids], dtype=object)
        result = base.copy()
        role_rows = [R.build_role_row(record, self.labels[0]) for record in rows]

        trigger_indices: list[int] = []
        for index, role in enumerate(role_rows):
            order = np.argsort(-base_probs[index])[:2]
            top_pair = {self.labels[int(position)] for position in order}
            if (
                role["source"] == "sim"
                and role["history_len"] > 0
                and top_pair == set(R.PAIR_SPECS["grep_read"])
            ):
                trigger_indices.append(index)
        if not trigger_indices:
            return result.tolist()

        linear_texts = [role_rows[index]["linear_text"] for index in trigger_indices]
        linear_predictions = np.asarray(self.linear.predict(linear_texts), dtype=object)
        linear_decision = np.asarray(self.linear.decision_function(linear_texts), dtype=np.float64)
        linear_decision = linear_decision.reshape(-1)

        semantic_texts = [role_rows[index]["semantic_text"] for index in trigger_indices]
        bert_labels, bert_raw = self._bert_probabilities(semantic_texts)
        bert_predictions = np.asarray(
            [bert_labels[int(index)] for index in bert_raw.argmax(axis=1)], dtype=object
        )

        policy = self.manifest["policy"]
        linear_features: list[dict[str, Any]] = []
        bert_features: list[dict[str, Any]] = []
        for local, row_index in enumerate(trigger_indices):
            probability_map = {
                label: float(base_probs[row_index, index])
                for index, label in enumerate(self.labels)
            }
            bert_map = {
                label: float(bert_raw[local, bert_labels.index(label)])
                for label in R.PAIR_SPECS["grep_read"]
            }
            common = (role_rows[row_index], probability_map, str(base[row_index]))
            linear_features.append(
                R.build_deploy_gate_feature(
                    "linear", *common, str(linear_predictions[local]),
                    float(linear_decision[local]), bert_map,
                )
            )
            bert_features.append(
                R.deploy_gate_text(
                    role_rows[row_index],
                    R.build_deploy_gate_feature(
                        "bert", *common, str(bert_predictions[local]),
                        float(linear_decision[local]), bert_map,
                    ),
                )
            )

        linear_probability = _outcome_probabilities(self.linear_gate, linear_features)
        bert_probability = _outcome_probabilities(self.bert_gate, bert_features)
        linear_config = policy["linear"]
        bert_config = policy["bert"]
        linear_utility = (
            linear_probability[:, OUTCOMES.index("rescue")]
            - float(linear_config["damage_weight"])
            * linear_probability[:, OUTCOMES.index("damage")]
            - float(linear_config["wrong_weight"])
            * linear_probability[:, OUTCOMES.index("wrong_to_wrong")]
        )
        bert_utility = (
            bert_probability[:, OUTCOMES.index("rescue")]
            - float(bert_config["damage_weight"])
            * bert_probability[:, OUTCOMES.index("damage")]
            - float(bert_config["wrong_weight"])
            * bert_probability[:, OUTCOMES.index("wrong_to_wrong")]
        )

        linear_owned: set[int] = set()
        for local, row_index in enumerate(trigger_indices):
            if (
                linear_predictions[local] != base[row_index]
                and linear_utility[local] >= float(linear_config["threshold"])
            ):
                result[row_index] = linear_predictions[local]
                linear_owned.add(row_index)
        for local, row_index in enumerate(trigger_indices):
            if row_index in linear_owned:
                continue
            if (
                bert_predictions[local] != base[row_index]
                and bert_utility[local] >= float(bert_config["threshold"])
            ):
                result[row_index] = bert_predictions[local]
        return result.tolist()


_DEFAULT_PREDICTOR: Group1Predictor | None = None


def predict(
    records: Iterable[dict[str, Any]],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> list[str]:
    global _DEFAULT_PREDICTOR
    requested = Path(model_dir).resolve()
    if _DEFAULT_PREDICTOR is None or _DEFAULT_PREDICTOR.model_dir != requested:
        _DEFAULT_PREDICTOR = Group1Predictor(requested)
    return _DEFAULT_PREDICTOR.predict(records)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Line {line_number} is not an object")
            rows.append(value)
    return rows


def package_size(model_dir: Path) -> int:
    return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())


def verify(model_dir: Path) -> None:
    predictor = Group1Predictor(model_dir)
    size = package_size(model_dir)
    limit = int(
        predictor.manifest.get(
            "size_limit_bytes",
            int(predictor.manifest.get("size_limit_mib", 300)) * 1024 * 1024,
        )
    )
    if size > limit:
        raise RuntimeError(f"Model size {size / 2**20:.2f} MiB exceeds {limit / 2**20:.2f} MiB")
    print(f"OK: {size / 2**20:.2f} MiB, labels={predictor.labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.model_dir)
        return
    if args.input is None or args.output is None:
        parser.error("Pass --verify or both --input and --output")
    records = load_jsonl(args.input)
    predictions = predict(records, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "action"])
        writer.writeheader()
        writer.writerows(
            {"id": record.get("id", ""), "action": action}
            for record, action in zip(records, predictions)
        )


if __name__ == "__main__":
    main()
