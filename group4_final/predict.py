#!/usr/bin/env python3
"""
Group4 final training script.

Runtime artifacts:
- features.py   : A5 text feature generation
- predict.py    : full-data training
- model_group4/ : HuggingFace model, tokenizer, label map, and metadata
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import platform
import random
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

import features as F


CLASSES = F.ACTION_CLASSES
CLASS_TO_ID = {label: index for index, label in enumerate(CLASSES)}

DEFAULT_MODEL_NAME = "klue/roberta-base"
DEFAULT_EPOCHS = 10.0
DEFAULT_CLASS_WEIGHT_POWER = 0.5
DEFAULT_MAX_LENGTH = 480

EXPECTED_VERSIONS = {
    "numpy": None,
    "pandas": None,
    "scikit-learn": None,
    "torch": None,
    "transformers": None,
    "accelerate": None,
}


def installed_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def check_library_versions() -> Dict[str, str]:
    installed = {package: installed_version(package) for package in EXPECTED_VERSIONS}
    missing = {
        package: installed[package]
        for package in EXPECTED_VERSIONS
        if installed[package] == "NOT_INSTALLED"
    }
    if missing:
        print("[WARNING] Required packages are missing.")
        print(json.dumps(missing, ensure_ascii=False, indent=2))
    return installed


def load_labels(path: str | os.PathLike[str]) -> Dict[str, str]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "action"}
        if reader.fieldnames is None or required - set(reader.fieldnames):
            raise KeyError(f"labels CSV must contain columns: {sorted(required)}")
        labels = {row["id"]: row["action"] for row in reader}
    if not labels:
        raise ValueError(f"No labels found: {path}")
    return labels


def attach_labels(
    records: List[Dict[str, Any]],
    labels_csv: str | os.PathLike[str] | None,
) -> List[Dict[str, Any]]:
    if all(str(record.get("action", "")) in CLASSES for record in records):
        return records

    if not labels_csv:
        raise ValueError(
            "Training JSONL does not contain complete action labels. "
            "Provide --labels-csv."
        )

    labels = load_labels(labels_csv)
    output: List[Dict[str, Any]] = []
    missing_ids: List[str] = []

    for record in records:
        copied = dict(record)
        record_id = str(record["id"])
        if record_id not in labels:
            missing_ids.append(record_id)
        else:
            copied["action"] = labels[record_id]
        output.append(copied)

    if missing_ids:
        raise KeyError(
            f"Labels missing for {len(missing_ids)} ids. "
            f"Examples: {missing_ids[:5]}"
        )

    return output


def validate_schema(records: Sequence[Dict[str, Any]]) -> None:
    required = {"id", "session_meta", "history", "current_prompt"}
    seen_ids = set()

    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise KeyError(f"record index={index} missing fields: {sorted(missing)}")

        record_id = str(record["id"])
        if not record_id:
            raise ValueError(f"record index={index} has an empty id")
        if record_id in seen_ids:
            raise ValueError(f"Duplicate id: {record_id}")
        seen_ids.add(record_id)

        if not isinstance(record["session_meta"], dict):
            raise TypeError(f"id={record_id}: session_meta must be an object")
        if not isinstance(record["history"], list):
            raise TypeError(f"id={record_id}: history must be a list")

        action = str(record.get("action", ""))
        if action not in CLASSES:
            raise ValueError(f"id={record_id}: invalid Group4 action={action!r}")


def resolve_data_file(path: str, data_dir: str | os.PathLike[str] | None) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    if data_dir:
        candidate = Path(data_dir) / raw
        if candidate.exists():
            return candidate
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / raw,
        Path.cwd() / "data" / raw,
        Path.cwd() / "group" / raw,
        script_dir / raw,
        script_dir / "data" / raw,
        script_dir / "group" / raw,
        script_dir.parent / "data" / raw,
        script_dir.parent / "group" / raw,
        Path("/root/workspace/dacon_inc/data") / raw,
        Path("/content/open/data") / raw,
        Path("/content/group") / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {path}. Checked:\n{checked}")


class EncodedDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: Sequence[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.class_weights is None:
            loss = torch_f.cross_entropy(outputs.logits, labels)
        else:
            weights = self.class_weights.to(outputs.logits.device)
            loss = torch_f.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = args.truncation_side
    tokenizer.add_special_tokens({"additional_special_tokens": F.SPECIAL_TOKENS})

    id_to_label = {index: label for index, label in enumerate(CLASSES)}
    label_to_id = {label: index for index, label in id_to_label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASSES),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def encode_texts(tokenizer, texts: Sequence[str], max_length: int) -> Dict[str, List[List[int]]]:
    return tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def class_weights(y_train: np.ndarray, power: float) -> torch.Tensor | None:
    if power <= 0:
        return None
    labels = np.arange(len(CLASSES))
    raw = compute_class_weight("balanced", classes=labels, y=y_train)
    return torch.tensor(np.power(raw, power), dtype=torch.float32)


def trainer_tokenizer_kwargs(tokenizer):
    parameter_names = set(inspect.signature(Trainer.__init__).parameters)
    if "processing_class" in parameter_names:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def make_training_args(args: argparse.Namespace, run_dir: Path, has_eval: bool):
    quick_epochs = 1.0 if args.quick else args.epochs
    kwargs = {
        "output_dir": str(run_dir / "trainer"),
        "num_train_epochs": quick_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "fp16": torch.cuda.is_available() and not args.no_fp16,
        "logging_steps": args.logging_steps,
        "save_total_limit": 1,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "disable_tqdm": args.disable_tqdm,
    }
    parameter_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    eval_key = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"

    if has_eval:
        kwargs.update(
            {
                eval_key: "epoch",
                "save_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "macro_f1",
                "greater_is_better": True,
            }
        )
    else:
        kwargs.update({eval_key: "no", "save_strategy": "no"})

    return TrainingArguments(**kwargs)


def evaluate_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    pred = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(
            labels,
            pred,
            labels=np.arange(len(CLASSES)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            pred,
            labels=np.arange(len(CLASSES)),
            average="weighted",
            zero_division=0,
        ),
        "report": report,
    }


def split_indices(args: argparse.Namespace, y_text: np.ndarray, groups: np.ndarray):
    indices = np.arange(len(y_text))
    if args.mode == "full":
        return [(indices, None)]
    if args.split_mode == "row":
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=args.valid_ratio,
            random_state=args.seed,
            stratify=y_text,
        )
        return [(train_idx, valid_idx)]
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(indices, y_text, groups))
    return splits[:1] if args.mode == "holdout" else splits


def train_one(
    args: argparse.Namespace,
    records: Sequence[Dict[str, Any]],
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray | None,
    run_dir: Path,
) -> Dict[str, Any]:
    set_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(args)
    texts = [F.build_text(record) for record in records]
    y_train = y[train_idx].astype(np.int64)
    y_valid = y[valid_idx].astype(np.int64) if valid_idx is not None else None

    train_ds = EncodedDataset(
        encode_texts(tokenizer, [texts[i] for i in train_idx], args.max_length),
        y_train.tolist(),
    )
    valid_ds = None
    if valid_idx is not None:
        valid_ds = EncodedDataset(
            encode_texts(tokenizer, [texts[i] for i in valid_idx], args.max_length),
            y_valid.tolist(),
        )

    def compute_metrics(prediction):
        pred = np.argmax(prediction.predictions, axis=-1)
        return {
            "macro_f1": f1_score(
                prediction.label_ids,
                pred,
                labels=np.arange(len(CLASSES)),
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                prediction.label_ids,
                pred,
                labels=np.arange(len(CLASSES)),
                average="weighted",
                zero_division=0,
            ),
        }

    trainer = WeightedTrainer(
        class_weights=class_weights(y_train, args.class_weight_power),
        model=model,
        args=make_training_args(args, run_dir, has_eval=valid_ds is not None),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if valid_ds is not None else None,
        **trainer_tokenizer_kwargs(tokenizer),
    )

    started = time.perf_counter()
    trainer.train()
    train_minutes = round((time.perf_counter() - started) / 60.0, 2)

    result: Dict[str, Any] = {"train_minutes": train_minutes}
    if valid_ds is not None:
        logits = trainer.predict(valid_ds).predictions.astype(np.float32)
        base = evaluate_logits(logits, y_valid)
        np.savez(
            run_dir / "valid_logits.npz",
            logits=logits,
            labels=y_valid,
            ids=np.array([records[i]["id"] for i in valid_idx]),
            classes=np.array(CLASSES),
        )
        result.update(
            {
                "valid_macro_f1": float(base["macro_f1"]),
                "valid_weighted_f1": float(base["weighted_f1"]),
                "report": base["report"],
            }
        )

    output_model_dir = run_dir / "model_group4"
    if args.save_fp16:
        trainer.model.half()
        trainer.model.config.torch_dtype = torch.float16
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))

    label_map = {
        "id2label": {str(index): label for index, label in enumerate(CLASSES)},
        "label2id": {label: index for index, label in enumerate(CLASSES)},
    }
    (output_model_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_model_dir / "logit_bias.json").write_text(
        json.dumps({label: 0.0 for label in CLASSES}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    installed_versions = check_library_versions()

    train_jsonl = resolve_data_file(args.train_jsonl, args.data_dir)
    labels_csv = resolve_data_file(args.labels_csv, args.data_dir) if args.labels_csv else None

    records = attach_labels(F.load_jsonl(train_jsonl), labels_csv)
    validate_schema(records)

    y_text = np.asarray([str(record["action"]) for record in records], dtype=object)
    y = np.asarray([CLASS_TO_ID[label] for label in y_text], dtype=np.int64)
    groups = np.asarray([F.session_id(str(record["id"])) for record in records], dtype=object)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"rows={len(records)} labels={dict(Counter(y_text))}")
    print(f"train_jsonl={train_jsonl.resolve()}")
    print(f"labels_csv={labels_csv.resolve() if labels_csv else None}")
    print(f"output={output_dir.resolve()}")

    metadata: Dict[str, Any] = {
        "labels": CLASSES,
        "label_counts": dict(Counter(y_text)),
        "args": vars(args),
        "model_name": args.model_name,
        "library_versions": installed_versions,
        "python_version": platform.python_version(),
        "folds": [],
    }

    splits = split_indices(args, y_text, groups)
    for fold_index, (train_idx, valid_idx) in enumerate(splits, start=1):
        run_dir = output_dir if args.mode == "full" else output_dir / f"fold_{fold_index}"
        result = train_one(args, records, y, train_idx, valid_idx, run_dir)
        result["fold"] = fold_index
        metadata["folds"].append(result)

    scored = [fold for fold in metadata["folds"] if "valid_macro_f1" in fold]
    if scored:
        metadata["mean_macro_f1"] = float(np.mean([fold["valid_macro_f1"] for fold in scored]))
        metadata["mean_weighted_f1"] = float(np.mean([fold["valid_weighted_f1"] for fold in scored]))

    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.mode == "full":
        print(f"Model saved: {(output_dir / 'model_group4').resolve()}")
    print(f"Metrics saved: {(output_dir / 'metrics.json').resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Group4 RoBERTa final model.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train-jsonl", default="train_group4.jsonl")
    parser.add_argument("--labels-csv", default="train_labels_group4.csv")
    parser.add_argument("--output", default="group4_final")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--mode", choices=["full", "holdout", "cv"], default="full")
    parser.add_argument("--split-mode", choices=["group", "row"], default="group")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--truncation-side", choices=["left", "right"], default="left")
    parser.add_argument("--class-weight-power", type=float, default=DEFAULT_CLASS_WEIGHT_POWER)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--save-fp16", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
