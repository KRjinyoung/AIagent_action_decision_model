"""
Shared feature engineering for the group1~group5 action router.

Used by both train_model.py and infer.py so training and inference
extract features identically.
"""
import json

import pandas as pd

# action(leaf tool call) -> group, per group info spec
GROUP_MAP = {
    "read_file": "group1",
    "grep_search": "group1",
    "glob_pattern": "group1",
    "list_directory": "group1",
    "plan_task": "group2",
    "ask_user": "group2",
    "web_search": "group2",
    "run_bash": "group3",
    "run_tests": "group3",
    "lint_or_typecheck": "group3",
    "write_file": "group4",
    "edit_file": "group4",
    "apply_patch": "group4",
    "respond_only": "group5",
}

TEXT_COLUMN = "prompt"
CAT_COLUMNS = ["last_action", "user_tier", "language_pref", "last_ci_status", "git_dirty"]
NUM_COLUMNS = [
    "hist_len",
    "turn_index",
    "budget_tokens_remaining",
    "elapsed_session_sec",
    "loc",
    "num_open_files",
    "top_lang_share",
]


def load_records(path):
    """Read a .jsonl file (one session record per line) into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _last_action_name(history):
    for item in reversed(history):
        if item.get("role") == "assistant_action":
            return item.get("name")
    return None


def build_feature_frame(records):
    """Turn a list of raw session records (as in train.jsonl) into the flat
    DataFrame consumed by the model pipeline."""
    rows = []
    for d in records:
        history = d.get("history", []) or []
        session_meta = d.get("session_meta", {}) or {}
        workspace = session_meta.get("workspace", {}) or {}
        lang_mix = workspace.get("language_mix", {}) or {}
        top_lang_share = max(lang_mix.values()) if lang_mix else 0.0

        rows.append(
            {
                "id": d.get("id"),
                "prompt": d.get("current_prompt", "") or "",
                "last_action": _last_action_name(history) or "NONE",
                "hist_len": len(history),
                "turn_index": session_meta.get("turn_index", 0) or 0,
                "budget_tokens_remaining": session_meta.get("budget_tokens_remaining", 0) or 0,
                "elapsed_session_sec": session_meta.get("elapsed_session_sec", 0) or 0,
                "user_tier": session_meta.get("user_tier", "NA") or "NA",
                "language_pref": session_meta.get("language_pref", "NA") or "NA",
                "loc": workspace.get("loc", 0) or 0,
                "git_dirty": str(workspace.get("git_dirty", False)),
                "last_ci_status": workspace.get("last_ci_status", "NA") or "NA",
                "num_open_files": len(workspace.get("open_files", []) or []),
                "top_lang_share": top_lang_share,
            }
        )
    return pd.DataFrame(rows)
