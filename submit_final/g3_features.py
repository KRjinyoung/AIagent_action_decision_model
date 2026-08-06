#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared feature generation for Group3 V4 + Context Recovery BERT."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd


CLASSES = ["lint_or_typecheck", "run_bash", "run_tests"]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, raw in enumerate(file, start=1):
            raw = raw.strip()
            if not raw:
                continue

            row = json.loads(raw)
            if not isinstance(row, dict):
                raise TypeError(f"Line {line_no} must be a JSON object.")

            row_id = str(row.get("id", ""))
            if not row_id:
                raise KeyError(f"Missing id at line {line_no}.")
            if row_id in seen:
                raise ValueError(f"Duplicate id={row_id}")

            seen.add(row_id)
            rows.append(row)

    if not rows:
        raise ValueError(f"No records found: {path}")

    return rows


def session_id(id_str: str) -> str:
    return re.sub(r"-step_\d+$", "", id_str)


def source_tag(id_str: str) -> str:
    match = re.match(r"sess_(sim|au)_", id_str)
    return match.group(1) if match else "other"


# ---------------------------------------------------------------------------
# V4 structured features
# ---------------------------------------------------------------------------

BUILD_PAT = re.compile(
    r"package\.json|requirements\.txt|dockerfile|makefile|\.github|ci\.ya?ml|"
    r"cargo\.toml|go\.mod|pom\.xml|build\.gradle",
    re.I,
)

LINT_KW = [
    "lint", "typecheck", "type check", "type-check", "타입체크", "타입 체크",
    "린트", "정적분석", "정적 분석", "tsc", "mypy", "eslint", "vet",
    "ruff", "flake8", "타입", "형식", "static analysis", "type checker",
    "type error",
]
TEST_KW = ["test", "테스트", "pytest", "unittest", "spec", "suite"]
BASH_KW = [
    "bash", "돌려", "실행", "shell", "command", "커맨드", "cli", "터미널",
    "run it", "rebuild", "deploy", "script", "스크립트", "restart",
]
FAIL_KW = [
    "fail", "error", "에러", "실패", "깨졌", "안 되", "안돼", "broken",
    "bug", "버그",
]
REPEAT_KW = [
    "again", "rerun", "re-run", "다시", "재실행", "한번 더", "여전히",
    "still",
]

CAT_COLS = [
    "user_tier", "language_pref", "git_dirty", "last_ci_status",
    "last_action_name", "last_action_result", "workspace_template_id", "source",
]
MANUAL_TEXT_COLS = [
    "len_chars", "n_words", "has_lint_kw", "has_test_kw", "has_bash_kw",
    "has_fail_kw", "has_repeat_kw", "n_question_marks",
]
RECENCY_COLS = [
    "turns_since_run_tests", "turns_since_lint", "turns_since_run_bash",
    "turns_since_edit_file", "turns_since_apply_patch",
]
EXTRA_COLS = ["last_arg_has_build_file"]
HIST_CNT_ACTIONS = [
    "run_bash", "run_tests", "lint_or_typecheck", "edit_file", "apply_patch",
    "read_file", "grep_search", "write_file", "glob_pattern",
]


def turns_since(actions: Sequence[Mapping[str, Any]], name: str) -> int:
    for index, action in enumerate(reversed(actions)):
        if action.get("name") == name:
            return index
    return 99


def build_base_dataframe(
    rows: Sequence[Mapping[str, Any]],
    has_label: bool = False,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for row in rows:
        session_meta = row["session_meta"]
        workspace = session_meta["workspace"]
        language_mix = workspace.get("language_mix") or {}
        template_id = ",".join(
            f"{key}:{value}"
            for key, value in sorted(language_mix.items())
        )

        history = row["history"]
        actions = [
            item
            for item in history
            if item.get("role") == "assistant_action"
        ]

        last_name = actions[-1].get("name") if actions else None
        last_args = actions[-1].get("args", {}) if actions else {}
        last_arg_text = " ".join(str(value) for value in last_args.values())
        last_summary = (
            actions[-1].get("result_summary", "") or ""
            if actions
            else ""
        )

        last_result = None
        if actions:
            if re.search(r"\bfail|\berror", last_summary, re.I):
                last_result = "FAIL"
            elif re.search(r"\bpass|^ok\b|; ok", last_summary, re.I):
                last_result = "OK"
            else:
                last_result = "OTHER"

        action_counts = collections.Counter(
            str(action.get("name", ""))
            for action in actions
        )
        user_history_text = " ".join(
            str(item.get("content", ""))
            for item in history
            if item.get("role") == "user"
        )

        record: Dict[str, Any] = {
            "id": str(row["id"]),
            "session_id": session_id(str(row["id"])),
            "current_prompt": str(row.get("current_prompt", "") or ""),
            "user_hist_text": user_history_text,
            "user_tier": session_meta["user_tier"],
            "language_pref": session_meta["language_pref"],
            "budget_tokens_remaining": session_meta["budget_tokens_remaining"],
            "turn_index": session_meta["turn_index"],
            "elapsed_session_sec": session_meta["elapsed_session_sec"],
            "loc": workspace["loc"],
            "git_dirty": workspace["git_dirty"],
            "last_ci_status": workspace["last_ci_status"],
            "n_open_files": len(workspace.get("open_files") or []),
            "workspace_template_id": template_id,
            "source": source_tag(str(row["id"])),
            "history_length": len(history),
            "last_action_name": last_name,
            "last_action_result": last_result,
            "last_arg_has_build_file": int(
                bool(BUILD_PAT.search(last_arg_text))
            ),
            "turns_since_run_tests": turns_since(actions, "run_tests"),
            "turns_since_lint": turns_since(actions, "lint_or_typecheck"),
            "turns_since_run_bash": turns_since(actions, "run_bash"),
            "turns_since_edit_file": turns_since(actions, "edit_file"),
            "turns_since_apply_patch": turns_since(actions, "apply_patch"),
        }

        for action_name in HIST_CNT_ACTIONS:
            record[f"hist_cnt_{action_name}"] = action_counts.get(
                action_name, 0
            )

        if has_label:
            record["action"] = row["action"]

        records.append(record)

    return pd.DataFrame(records)


def add_manual_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    text = output["current_prompt"].str.lower()

    output["len_chars"] = output["current_prompt"].str.len()
    output["n_words"] = output["current_prompt"].str.split().str.len()
    output["has_lint_kw"] = text.apply(
        lambda value: any(keyword in value for keyword in LINT_KW)
    ).astype(int)
    output["has_test_kw"] = text.apply(
        lambda value: any(keyword in value for keyword in TEST_KW)
    ).astype(int)
    output["has_bash_kw"] = text.apply(
        lambda value: any(keyword in value for keyword in BASH_KW)
    ).astype(int)
    output["has_fail_kw"] = text.apply(
        lambda value: any(keyword in value for keyword in FAIL_KW)
    ).astype(int)
    output["has_repeat_kw"] = text.apply(
        lambda value: any(keyword in value for keyword in REPEAT_KW)
    ).astype(int)
    output["n_question_marks"] = output["current_prompt"].str.count(r"\?")

    return output


def get_base_num_cols(frame: pd.DataFrame) -> List[str]:
    excluded = {
        "id", "session_id", "action", "y", "current_prompt", "user_hist_text",
        "budget_rel_tier_z", "loc_rel_template_z",
    }
    excluded |= (
        set(CAT_COLS)
        | set(MANUAL_TEXT_COLS)
        | set(RECENCY_COLS)
        | set(EXTRA_COLS)
    )
    return [
        column
        for column in frame.columns
        if column not in excluded
    ]


def get_feature_columns(frame: pd.DataFrame) -> List[str]:
    return (
        CAT_COLS
        + get_base_num_cols(frame)
        + ["budget_rel_tier_z", "loc_rel_template_z"]
        + MANUAL_TEXT_COLS
        + RECENCY_COLS
        + EXTRA_COLS
    )


def apply_group_relative_dict(
    frame: pd.DataFrame,
    stats: Mapping[str, Any],
    value_col: str,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    means = stats["means"]
    stds = stats["stds"]

    mean_values = (
        frame[group_col]
        .astype(str)
        .map(means)
        .astype(float)
        .fillna(float(stats["global_mean"]))
    )
    std_values = (
        frame[group_col]
        .astype(str)
        .map(stds)
        .astype(float)
        .fillna(
            float(stats["global_std"])
            if float(stats["global_std"])
            else 1.0
        )
        .replace(0, 1.0)
    )

    frame[f"{prefix}_z"] = (
        frame[value_col].astype(float) - mean_values
    ) / std_values

    return frame


def previous_user_text(record: Mapping[str, Any]) -> str:
    turns = [
        str(item.get("content", "") or "")
        for item in record.get("history", [])
        if item.get("role") == "user"
    ]
    return turns[-1] if turns else ""


def build_current_texts(
    records: Sequence[Mapping[str, Any]],
) -> List[str]:
    return [
        f"[CURRENT] {str(record.get('current_prompt', '') or '')}"
        for record in records
    ]


def build_history_texts(
    records: Sequence[Mapping[str, Any]],
) -> List[str]:
    return [
        (
            f"[PREV_USER] {previous_user_text(record)} "
            f"[CURRENT] {str(record.get('current_prompt', '') or '')}"
        )
        for record in records
    ]


# ---------------------------------------------------------------------------
# Context Recovery BERT formatter
# ---------------------------------------------------------------------------

BERT_MAX_LENGTH = 384
PROMPT_CHARS = 1800
LAST_USER_CHARS = 500
LAST_RESULT_CHARS = 320
HISTORY_ITEMS = 5
HISTORY_ITEM_CHARS = 220
SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, max_chars: int) -> str:
    text = SPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def compact_json(value: Any, max_chars: int) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        text = str(value)
    return clean_text(text, max_chars)


def extract_turn(record: Mapping[str, Any]) -> int:
    session_meta = record.get("session_meta") or {}
    if isinstance(session_meta, Mapping):
        value = session_meta.get("turn_index")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    match = re.search(
        r"step[_-](\d+)",
        str(record.get("id", "")),
        re.I,
    )
    return int(match.group(1)) if match else 0


def final_request_segment(prompt: str) -> str:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", prompt)
        if part.strip()
    ]
    return " ".join(parts[-2:]) if parts else prompt


def context_history_summary(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    history = record.get("history") or []
    if not isinstance(history, list):
        history = []

    last_user = ""
    last_action = "none"
    last_result = "none"
    action_names: List[str] = []

    for item in history:
        if not isinstance(item, Mapping):
            continue

        role = str(item.get("role", ""))

        if role == "user":
            content = clean_text(item.get("content"), LAST_USER_CHARS)
            if content:
                last_user = content

        elif role == "assistant_action":
            action_name = clean_text(item.get("name"), 80)
            result = clean_text(
                item.get("result_summary"),
                LAST_RESULT_CHARS,
            )

            if action_name:
                action_names.append(action_name)
                last_action = action_name
            if result:
                last_result = result

    return {
        "history": history,
        "last_user": last_user or "none",
        "last_action": last_action,
        "last_result": last_result,
        "recent_action_flow": (
            " > ".join(action_names[-5:])
            if action_names
            else "none"
        ),
    }


def build_context_recovery_text(
    record: Mapping[str, Any],
) -> str:
    prompt = clean_text(record.get("current_prompt"), PROMPT_CHARS)
    final_request = clean_text(final_request_segment(prompt), 600)

    session_meta = record.get("session_meta") or {}
    workspace = (
        session_meta.get("workspace", {})
        if isinstance(session_meta, Mapping)
        else {}
    )
    if not isinstance(workspace, Mapping):
        workspace = {}

    open_files = workspace.get("open_files", [])
    if not isinstance(open_files, list):
        open_files = []

    language = (
        session_meta.get("language_pref", "unknown")
        if isinstance(session_meta, Mapping)
        else "unknown"
    )

    summary = context_history_summary(record)

    parts = [
        f"[CURRENT_PROMPT] {prompt}",
        f"[FINAL_REQUEST] {final_request}",
        f"[LAST_USER] {summary['last_user']}",
        f"[LAST_ACTION] {summary['last_action']}",
        f"[LAST_RESULT] {summary['last_result']}",
        f"[RECENT_ACTION_FLOW] {summary['recent_action_flow']}",
        (
            f"[SESSION_STATE] turn={extract_turn(record)} "
            f"language={language} "
            f"last_ci={workspace.get('last_ci_status', 'none')} "
            f"git_dirty={int(bool(workspace.get('git_dirty', False)))} "
            f"open_files={len(open_files)}"
        ),
        "[RECENT_HISTORY newest_first]",
    ]

    history = summary["history"]

    for rank, item in enumerate(
        history[-HISTORY_ITEMS:][::-1],
        start=1,
    ):
        if not isinstance(item, Mapping):
            continue

        role = str(item.get("role", "unknown"))

        if role == "user":
            parts.append(
                f"[H{rank} USER] "
                f"{clean_text(item.get('content'), HISTORY_ITEM_CHARS)}"
            )
        elif role == "assistant_action":
            parts.append(
                f"[H{rank} ACTION] "
                f"name={clean_text(item.get('name'), 80)} "
                f"result={clean_text(item.get('result_summary'), 150)} "
                f"args={compact_json(item.get('args', {}), 70)}"
            )
        else:
            content = item.get(
                "content",
                item.get("result_summary", ""),
            )
            parts.append(
                f"[H{rank} {role.upper()}] "
                f"{clean_text(content, HISTORY_ITEM_CHARS)}"
            )

    return "\n".join(parts)


def build_context_texts(
    records: Sequence[Mapping[str, Any]],
) -> List[str]:
    return [
        build_context_recovery_text(record)
        for record in records
    ]
