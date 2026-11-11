# AIagent_action_decision_model

**Next-action prediction for AI coding agents** — a code-submission competition hosted by Dacon (2026).

Given a point in a conversation between a user and an AI coding agent, the model predicts which of 14 actions the agent should take next. It uses session metadata, workspace state, dialogue history, and the current user prompt as input.

## Results

| Model | Public macro-F1 |
| --- | --- |
| Official baseline | 0.436 |
| **Ours (router + group-specific experts)** | **0.789** |

All models were trained on an **NVIDIA H100** GPU. Inference runs under strict competition limits: a single NVIDIA T4 GPU, a 10-minute time budget, and no internet access.

## My Contribution

This was a team project with one owner per component: I owned the router, and each of the four group experts (group1–group4) was assigned to one team member.

- **Overall architecture**: I designed the two-stage routing structure itself: split the 14 actions into five behavioural groups, route each sample to a group first, then let a group-specific expert make the final prediction.
- **Router**: I built the router (word/char TF-IDF on the current prompt + session/workspace metadata, soft-voting ensemble of a calibrated LinearSVC and Logistic Regression). Group routing accuracy is about 99%, so almost all remaining error comes from the within-group decision. See `router_final/`.
- **Group1 (Explore / Retrieve) and Group3 (Execute / Verify)**: in addition to the router, I built my own experts for these two groups. They outperformed the alternatives and were adopted in the final submission.
  - Group1: structured XGBoost + ExtraTrees base model, with `grep_search` / `read_file` specialists (linear + RoBERTa) that may override the base prediction only when a learned utility gate expects a net gain.
  - Group3: V4 structured ensemble (LightGBM + CatBoost + XGBoost with separate history / current-prompt branches and an `au`-specific branch), a Context Recovery BERT, and source-aware blending (BERT is blended only on `sim` records, 0.575 · V4 + 0.425 · BERT). See `group3_final/`.

Group2 and Group4 were built by teammates and are included so the full pipeline is visible.

## Problem Definition

- **Input**
  - `session_meta`: subscription plan, language preference, remaining tokens, turn index, elapsed time, workspace state
  - `history`: prior dialogue and action records (0–12 entries)
  - `current_prompt`: the user's current message
- **Output**: one of the 14 classes below

| Group | Role | Classes |
| --- | --- | --- |
| group1 | Explore / Retrieve | `read_file`, `grep_search`, `glob_pattern`, `list_directory` |
| group2 | Decide / Extend | `plan_task`, `ask_user`, `web_search` |
| group3 | Execute / Verify | `run_bash`, `run_tests`, `lint_or_typecheck` |
| group4 | Edit / Create | `edit_file`, `write_file`, `apply_patch` |
| group5 | Respond only | `respond_only` |

- **Metric**: macro F1
- **Submission format**: code submission. A `submit.zip` containing `model/`, `script.py`, and `requirements.txt` is uploaded, and the evaluation server runs `script.py` to produce `output/submission.csv`. (See `규칙.txt` for the competition rules and `명세서.txt` for the data specification; both are in Korean.)

## Architecture: Router + Group-Specific Experts

Prediction is done in two stages. A router first assigns each sample to one of five action groups, and a dedicated expert model for that group then makes the final 14-class prediction.

```
test.jsonl
    │
    ▼
router_model.pkl  ──► group1 | group2 | group3 | group4 | group5
    │                    │        │        │        │        │
    ▼                    ▼        ▼        ▼        ▼        ▼
router_features.py   g1_predict g2_predict g3_predict g4(RoBERTa)  respond_only
                     (explore)  (decide)  (execute) (edit/create)   (fixed)
    │                    │        │        │        │        │
    └────────────────────┴────────┴────────┴────────┴────────┘
                              ▼
                     output/submission.csv
```

### Models per Group (`submit_final/`)

- **Router**: group classifier (`router_model.pkl`, `router_features.py`)
- **group1 (Explore / Retrieve)**: a compressed XGBoost + ExtraTrees base model, combined with linear and RoBERTa experts specialized for `grep_search` / `read_file`, and a utility gate (`model/group1/`, `g1_feature.py`, `g1_predict.py`)
- **group2 (Decide / Extend)**: a lightweight ensemble that stacks HistGradientBoosting with word- and character-level TF-IDF Logistic Regression using session-wise GroupKFold out-of-fold predictions; no language model is used (`model/group2_model.pkl`, `g2_features.py`, `g2_predict.py`)
- **group3 (Execute / Verify)**: a structured-feature ensemble of LightGBM + CatBoost + XGBoost (V4), blended with a Context Recovery BERT (multilingual DistilBERT) on records where `source == "sim"` (`model/group3_model.pkl`, `model/group3_bert/`, `g3_features.py`, `g3_predict.py`)
- **group4 (Edit / Create)**: fine-tuned `klue/roberta-base` with fp16 inference (`model/group4/`, `g4_features.py`)
- **group5 (Respond only)**: no separate model; outputs `respond_only` when the router assigns this group

`submit_final/` is the deployment bundle that actually goes into the zip. The `router_final/` and `group{1..4}_final/` directories hold each model's standalone code:

| Directory | Contents in this repository |
| --- | --- |
| `router_final/` | training (`train.py`), inference, features, trained `router_model.pkl` |
| `group1_final/` | inference and features only |
| `group2_final/` | training + inference (`predict.py train / predict`), trained `model.pkl` |
| `group3_final/` | inference and features only |
| `group4_final/` | training + inference (`predict.py`), model info |

Large neural-network weights (`*.safetensors`, `*.bin`, `*.pt`, `*.joblib`) are excluded by `.gitignore`, so the Group1, Group3 and Group4 artifacts under `submit_final/model/` are not included in this repository.

## Repository Structure

```
├── group1_final/              # group1 inference + feature code
├── group2_final/              # group2 training/inference code + model
├── group3_final/              # group3 inference + feature code
├── group4_final/              # group4 training/inference code (with its own README)
├── router_final/              # router training/inference code + model
├── submit_final/              # source directory for the submission zip (includes script.py)
├── 명세서.txt                  # data & problem specification (Korean)
└── 규칙.txt                    # submission format & competition rules (Korean)
```

## Building the Submission (`submit.zip`)

`submit_final/` is the source of the zip. The final feature and model files from each `group{N}_final/` and `router_final/` are copied into `submit_final/`, and `submit_final/script.py` runs the router and all four group predictors end to end.

Always build the zip with Python's `zipfile`. Windows `Compress-Archive` writes `\` as the path separator, which breaks the directory structure on the Linux evaluation server.

```python
import os, zipfile

with zipfile.ZipFile("submit.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk("submit_final"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, "submit_final").replace(os.sep, "/")
            zf.write(path, rel)
```

## Hardware

- Training: NVIDIA H100
- Evaluation / inference: NVIDIA T4 (16 GB), see below

## Evaluation Environment

- OS: Ubuntu 22.04.5 LTS / Python 3.11.15 / CUDA 12.8 / GPU: NVIDIA T4 (16 GB)
- Preinstalled: `torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scikit-learn==1.8.0`, `transformers==4.46.3`, etc. (see `규칙.txt`)
- Constraints: zip size up to 1 GB, package installation and inference each within 10 minutes, no internet access
