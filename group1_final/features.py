import os
import re
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer


SEED = 42
EXPERIMENT_NAME = "v3_top5_only_soft_voting"

BASE_DIR = Path(".")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs") / EXPERIMENT_NAME
MODEL_DIR = Path("model")

TRAIN_JSONL_PATH = DATA_DIR / "train.jsonl"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"

TARGET_ACTIONS = [
    "read_file",
    "grep_search",
    "glob_pattern",
    "list_directory",
]

PAIR_LIST = [
    ("grep_search", "read_file"),
    ("read_file", "list_directory"),
    ("grep_search", "list_directory"),
    ("glob_pattern", "read_file"),
]


CODE_SEARCH_WORDS = [
    "ì‚¬ìš©ì²˜", "ì°¸ì¡°", "í˜¸ì¶œ", "ì •ì˜", "êµ¬í˜„", "ì—°ê²°", "ê´€ë ¨", "ì–´ë””ì„œ", "ì–´ë””ì—",
    "ì“°ëŠ”ì§€", "ì‚¬ìš©ë˜ëŠ”", "ì‚¬ìš©í•œ", "ì°¾ì•„", "ì¶”ì ",
    "reference", "references", "referenced", "usage", "used", "uses",
    "called", "caller", "callsite", "where used",
    "definition", "implementation", "implements",
]

CODE_ENTITY_WORDS = [
    "handler", "controller", "service", "repository", "router", "route",
    "middleware", "model", "schema", "component", "hook", "store",
    "class", "interface", "function", "method", "module",
]

CODE_IMPORT_WORDS = [
    "import", "export", "require", "from", "include", "dependency", "dependencies",
]

READ_TARGET_WORDS = [
    "ë³¸ë¬¸", "ë‚´ìš©", "íŒŒì¼", "ì—´ì–´", "ì½ì–´", "ë³´ì—¬", "í™•ì¸",
    "open", "read", "show", "view", "content", "inside",
]

LIST_STRUCTURE_WORDS = [
    "êµ¬ì¡°", "ëª©ë¡", "íŠ¸ë¦¬", "í´ë”", "ë””ë ‰í† ë¦¬", "í•˜ìœ„", "ì•ˆì— ë­",
    "structure", "tree", "folder", "directory", "list files",
]

GLOB_PATTERN_WORDS = [
    "ì „ì²´", "ì „ë¶€", "ëª¨ë“ ", "íŒ¨í„´", "í™•ìž¥ìž", "ì¼ê´„", "ëª©ë¡ìœ¼ë¡œ",
    "all", "every", "pattern", "glob", "wildcard", "*.py", "*.js", "*.json",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def clean_text(text, max_len=None):
    text = safe_str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if max_len is not None and len(text) > max_len:
        text = text[:max_len]
    return text


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_session_id(sample_id):
    return re.sub(r"-step_\d+$", "", safe_str(sample_id))


def get_step(sample_id):
    m = re.search(r"-step_(\d+)$", safe_str(sample_id))
    return int(m.group(1)) if m else -1


def flatten_args(args, max_items=8, max_value_len=120):
    if not isinstance(args, dict):
        return ""

    important_keys = [
        "path", "file", "filepath", "pattern", "query",
        "command", "cmd", "url", "target", "directory", "glob",
    ]

    sorted_items = sorted(args.items(), key=lambda kv: 0 if kv[0] in important_keys else 1)

    parts = []
    for i, (k, v) in enumerate(sorted_items):
        if i >= max_items:
            break

        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        else:
            v = safe_str(v)

        v = clean_text(v, max_len=max_value_len)
        parts.append(f"{k}: {v}")

    return " | ".join(parts)


def extract_history_pairs(history):
    if not isinstance(history, list) or len(history) == 0:
        return []

    pairs = []
    last_user = None

    for turn in history:
        role = turn.get("role")

        if role == "user":
            last_user = clean_text(turn.get("content", ""), max_len=260)

        elif role == "assistant_action":
            action_name = safe_str(turn.get("name", ""))
            args_text = flatten_args(turn.get("args", {}))
            result_text = clean_text(turn.get("result_summary", ""), max_len=220)

            pairs.append({
                "user": last_user if last_user is not None else "",
                "action": action_name,
                "args": args_text,
                "result": result_text,
            })
            last_user = None

    return pairs


def classify_result_type(result_summary):
    text = clean_text(result_summary).lower()

    if not text or text == "none":
        return "NONE"
    if any(k in text for k in ["error", "failed", "exception", "traceback", "no such file"]):
        return "ERROR"
    if "not found" in text:
        return "NOT_FOUND"
    if "0 occurrence" in text or "0 occurrences" in text or "0 matches" in text or "no matches" in text:
        return "NO_MATCH"
    if "found" in text and ("occurrence" in text or "match" in text or "matches" in text):
        nums = re.findall(r"\d+", text)
        if nums:
            n = int(nums[0])
            if n == 0:
                return "GREP_NO_MATCH"
            elif n <= 3:
                return "GREP_FEW_MATCH"
            elif n <= 20:
                return "GREP_SOME_MATCH"
            return "GREP_MANY_MATCH"
        return "GREP_FOUND"
    if "ok" in text and "read" in text:
        return "READ_OK"
    if text.startswith("read ") or "; read " in text:
        return "READ_OK"
    if "listed" in text:
        if "empty" in text:
            return "LIST_EMPTY"
        return "LIST_OK"
    if "empty directory" in text:
        return "LIST_EMPTY"
    if "files matched" in text or "file matched" in text or "matched" in text:
        nums = re.findall(r"\d+", text)
        if nums:
            n = int(nums[0])
            if n == 0:
                return "GLOB_NO_MATCH"
            elif n <= 5:
                return "GLOB_FEW_MATCH"
            elif n <= 30:
                return "GLOB_SOME_MATCH"
            return "GLOB_MANY_MATCH"
        return "GLOB_MATCH"
    if "passed" in text:
        return "PASSED"
    if "success" in text or "completed" in text or "done" in text:
        return "SUCCESS"

    return "OTHER"


def contains_any(text, keywords):
    text = safe_str(text).lower()
    return int(any(k in text for k in keywords))


def count_regex(text, pattern):
    return len(re.findall(pattern, safe_str(text)))


def get_action_tail_features(actions):
    tail3 = actions[-3:] if len(actions) >= 3 else actions
    tail5 = actions[-5:] if len(actions) >= 5 else actions

    bigrams = []
    trigrams = []

    for i in range(len(actions) - 1):
        bigrams.append(actions[i] + "->" + actions[i + 1])

    for i in range(len(actions) - 2):
        trigrams.append(actions[i] + "->" + actions[i + 1] + "->" + actions[i + 2])

    return {
        "last_1_action": actions[-1] if len(actions) >= 1 else "none",
        "last_2_action": actions[-2] if len(actions) >= 2 else "none",
        "last_3_action": actions[-3] if len(actions) >= 3 else "none",
        "tail3_action_seq": "->".join(tail3) if tail3 else "none",
        "tail5_action_seq": "->".join(tail5) if tail5 else "none",
        "last_bigram": bigrams[-1] if bigrams else "none",
        "last_trigram": trigrams[-1] if trigrams else "none",

        "has_grep_to_read": int("grep_search->read_file" in bigrams),
        "has_read_to_grep": int("read_file->grep_search" in bigrams),
        "has_list_to_read": int("list_directory->read_file" in bigrams),
        "has_glob_to_read": int("glob_pattern->read_file" in bigrams),

        "grep_read_bigram_count": bigrams.count("grep_search->read_file"),
        "read_grep_bigram_count": bigrams.count("read_file->grep_search"),
        "list_read_bigram_count": bigrams.count("list_directory->read_file"),
        "glob_read_bigram_count": bigrams.count("glob_pattern->read_file"),
    }


def build_features(df):
    rows = []

    for _, row in df.iterrows():
        sample_id = row["id"]
        current_prompt = clean_text(row.get("current_prompt", ""), max_len=1000)
        history = row.get("history", [])

        pairs = extract_history_pairs(history)
        actions = [p["action"] for p in pairs if p.get("action")]
        action_seq = " -> ".join(actions[-8:]) if actions else "none"
        action_tail = get_action_tail_features(actions)

        if pairs:
            last = pairs[-1]
            last_user = clean_text(last.get("user", ""), max_len=260) or "none"
            last_action = safe_str(last.get("action", "")) or "none"
            last_result = clean_text(last.get("result", ""), max_len=220) or "none"
            last_args = clean_text(last.get("args", ""), max_len=220) or "none"
        else:
            last_user = "none"
            last_action = "none"
            last_result = "none"
            last_args = "none"

        last_result_type = classify_result_type(last_result)

        history_text_parts = []
        for p in pairs[-3:]:
            history_text_parts.append(
                f"user: {p['user']} action: {p['action']} args: {p['args']} result: {p['result']}"
            )
        recent_history_text = " ".join(history_text_parts)

        combined_text = f"""
current_prompt: {current_prompt}
last_user: {last_user}
last_action: {last_action}
last_result_type: {last_result_type}
last_result: {last_result}
last_args: {last_args}
action_sequence: {action_seq}
recent_history: {recent_history_text}
""".strip()

        prompt_lower = current_prompt.lower()
        result_lower = last_result.lower()
        args_lower = last_args.lower()

        action_counts = {a: actions.count(a) for a in TARGET_ACTIONS}

        feature = {
            "id": sample_id,
            "action": row["action"],
            "session_id": get_session_id(sample_id),
            "step": get_step(sample_id),

            "current_prompt_text": current_prompt,
            "combined_text": combined_text,
            "recent_history_text": recent_history_text,

            "last_action": last_action,
            "last_result_type": last_result_type,
            "action_sequence": action_seq,

            "history_len": len(pairs),
            "prompt_len_char": len(current_prompt),
            "prompt_len_word": len(current_prompt.split()),
            "last_result_len_char": len(last_result),
            "last_args_len_char": len(last_args),

            "prev_read_count": action_counts["read_file"],
            "prev_grep_count": action_counts["grep_search"],
            "prev_glob_count": action_counts["glob_pattern"],
            "prev_list_count": action_counts["list_directory"],

            "has_slash": int("/" in current_prompt),
            "has_dot": int("." in current_prompt),
            "has_star": int("*" in current_prompt),
            "has_question": int("?" in current_prompt or "ï¼Ÿ" in current_prompt),
            "has_quote": int("'" in current_prompt or '"' in current_prompt or "`" in current_prompt),
            "path_like_count": count_regex(current_prompt, r"[A-Za-z0-9_\-./]+/[A-Za-z0-9_\-./]+"),
            "file_ext_count": count_regex(current_prompt, r"\.[A-Za-z0-9]{1,6}\b"),

            "has_py": contains_any(prompt_lower, [".py", "python"]),
            "has_js": contains_any(prompt_lower, [".js", ".jsx", ".ts", ".tsx"]),
            "has_json": contains_any(prompt_lower, [".json"]),
            "has_yaml": contains_any(prompt_lower, [".yaml", ".yml"]),
            "has_md": contains_any(prompt_lower, [".md", "readme"]),
            "has_txt": contains_any(prompt_lower, [".txt"]),

            "kw_read_open": contains_any(prompt_lower, ["ì—´ì–´", "ì½ì–´", "ë³´ì—¬", "í™•ì¸", "open", "read", "show"]),
            "kw_search": contains_any(prompt_lower, ["ì°¾", "ê²€ìƒ‰", "grep", "search", "where", "ì–´ë””", "ì‚¬ìš©", "ì“°ëŠ”ì§€"]),
            "kw_list": contains_any(prompt_lower, ["ëª©ë¡", "ë¦¬ìŠ¤íŠ¸", "êµ¬ì¡°", "í´ë”", "ë””ë ‰í† ë¦¬", "directory", "folder", "tree"]),
            "kw_glob": contains_any(prompt_lower, ["íŒ¨í„´", "ì „ì²´", "ì „ë¶€", "ëª¨ë“ ", "glob", "pattern", "*."]),

            "has_code_search_semantic": contains_any(prompt_lower, CODE_SEARCH_WORDS),
            "has_code_entity": contains_any(prompt_lower, CODE_ENTITY_WORDS),
            "has_import_semantic": contains_any(prompt_lower, CODE_IMPORT_WORDS),
            "has_read_target_semantic": contains_any(prompt_lower, READ_TARGET_WORDS),
            "has_list_structure_semantic": contains_any(prompt_lower, LIST_STRUCTURE_WORDS),
            "has_glob_pattern_semantic": contains_any(prompt_lower, GLOB_PATTERN_WORDS),

            "camel_case_count": count_regex(current_prompt, r"\b[a-z]+(?:[A-Z][a-zA-Z0-9]*)+\b"),
            "snake_case_count": count_regex(current_prompt, r"\b[a-z]+_[a-zA-Z0-9_]+\b"),
            "has_parentheses": int("(" in current_prompt or ")" in current_prompt),
            "has_colon": int(":" in current_prompt),
            "has_at_symbol": int("@" in current_prompt),
            "has_upper_identifier": int(bool(re.search(r"\b[A-Z][A-Za-z0-9]{2,}\b", current_prompt))),
            "code_identifier_count": count_regex(current_prompt, r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b"),

            "res_found": contains_any(result_lower, ["found", "occurrence", "matches"]),
            "res_read": contains_any(result_lower, ["read"]),
            "res_listed": contains_any(result_lower, ["listed"]),
            "res_matched": contains_any(result_lower, ["matched"]),
            "res_empty": contains_any(result_lower, ["empty"]),
            "res_error": contains_any(result_lower, ["error", "failed", "traceback", "exception"]),

            "args_has_path": int("/" in args_lower),
            "args_has_pattern": contains_any(args_lower, ["*", "pattern", "glob"]),
            "args_has_query": contains_any(args_lower, ["query"]),
        }

        feature.update(action_tail)
        rows.append(feature)

    return pd.DataFrame(rows)


