"""Feature generation shared by Group4 training and inference."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


ACTION_CLASSES = ["edit_file", "write_file", "apply_patch"]

ALL_ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

ACT_TOKENS = [f"<act_{name}>" for name in ALL_ACTIONS] + ["<act_none>"]
CI_TOKENS = ["<ci_passed>", "<ci_failed>", "<ci_none>"]
GIT_TOKENS = ["<git_dirty>", "<git_clean>", "<git_na>"]
SOURCE_TOKENS = ["<src_sim>", "<src_au>", "<src_other>"]
SEG_TOKENS = ["<ctx>", "<hist>", "<now>", "<open>", "<open_none>"]
TURN_TOKENS = [f"<turn_{i}>" for i in range(21)] + ["<turn_na>"]
AGO_TOKENS = [f"<ago_{i}>" for i in range(10)]

SPECIAL_TOKENS = (
    ACT_TOKENS
    + CI_TOKENS
    + GIT_TOKENS
    + SOURCE_TOKENS
    + SEG_TOKENS
    + TURN_TOKENS
    + AGO_TOKENS
)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def compact_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip()


def session_id(sample_id: str) -> str:
    return re.sub(r"-step_\d+$", "", safe_str(sample_id))


def source_token(sample_id: Any) -> str:
    sample_id = safe_str(sample_id)
    if sample_id.startswith("sess_sim_"):
        return "<src_sim>"
    if sample_id.startswith("sess_au_"):
        return "<src_au>"
    return "<src_other>"


def bounded_token(prefix: str, value: Any, max_value: int, missing: str) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return missing
    return f"<{prefix}_{max(0, min(number, max_value))}>"


def turn_token(value: Any) -> str:
    return bounded_token("turn", value, 20, "<turn_na>")


def ci_token(value: Any) -> str:
    token = f"<ci_{safe_str(value) or 'none'}>"
    return token if token in CI_TOKENS else "<ci_none>"


def git_token(value: Any) -> str:
    return {True: "<git_dirty>", False: "<git_clean>"}.get(value, "<git_na>")


def act_token(value: Any) -> str:
    token = f"<act_{safe_str(value)}>"
    return token if token in ACT_TOKENS else "<act_none>"


def fmt_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    return " ".join(f"{safe_str(key)}={safe_str(value)}" for key, value in sorted(args.items()))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parsing failed at line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Line {line_no} is not a JSON object.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found: {path}")
    return records


def build_text(record: Dict[str, Any]) -> str:
    """Build the A5 Group4 text template.

    The current prompt is intentionally placed at the end. The model uses
    left-side truncation, so this keeps the most important decision text.
    """
    session_meta = record.get("session_meta") or {}
    workspace = session_meta.get("workspace") or {}
    history = record.get("history") or []

    user_positions = [i for i, item in enumerate(history) if item.get("role") == "user"]
    keep_user = set(user_positions[-1:])

    action_positions = [
        i for i, item in enumerate(history) if item.get("role") == "assistant_action"
    ]
    n_actions = len(action_positions)
    ago_by_pos = {
        pos: min(n_actions - 1 - rank, len(AGO_TOKENS) - 1)
        for rank, pos in enumerate(action_positions)
    }

    parts = [
        "<ctx>",
        source_token(record.get("id")),
        turn_token(session_meta.get("turn_index")),
        ci_token(workspace.get("last_ci_status", "none")),
        git_token(workspace.get("git_dirty")),
    ]

    open_files = workspace.get("open_files") or []
    if open_files:
        parts.append("<open> " + " ".join(map(safe_str, open_files)))
    else:
        parts.append("<open_none>")

    parts.append("<hist>")
    saw_action = False
    for i, item in enumerate(history):
        role = item.get("role")
        if role == "user" and i in keep_user:
            content = compact_ws(item.get("content"))
            if content:
                parts.append("USER: " + content)
        elif role == "assistant_action":
            saw_action = True
            segment = [f"<ago_{ago_by_pos[i]}>", act_token(item.get("name"))]
            args = fmt_args(item.get("args"))
            if args:
                segment.append(args)
            result = compact_ws(item.get("result_summary"))
            if result:
                segment.append("-> " + result)
            parts.append(" ".join(segment))

    if not saw_action:
        parts.append("<act_none>")

    parts.append("<now>")
    current_prompt = compact_ws(record.get("current_prompt"))
    if current_prompt:
        parts.append(current_prompt)

    return " ".join(part for part in parts if part)
