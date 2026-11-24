# Router

First stage of the pipeline: assigns each session record to one of five action groups (`group1`–`group5`). The group-specific expert then makes the final 14-class prediction. Group definitions are in `그룹정보.txt` (Korean).

## Model

- **Text**: word TF-IDF (1–2 gram) and character TF-IDF (`char_wb`, 3–5 gram) on `current_prompt`
- **Categorical** (one-hot): last action, user tier, language preference, last CI status, git dirty
- **Numeric** (standardized): history length, turn index, remaining token budget, elapsed time, LOC, number of open files, top language share
- **Classifier**: soft-voting ensemble of a calibrated LinearSVC (weight 2) and Logistic Regression (weight 1)

Feature extraction lives in `features.py` and is shared by training and inference, so both see identical inputs.

## Usage

Train (reports a held-out macro-F1 on a 10% stratified split, then refits on all data):

```bash
python train.py --data train.jsonl --labels train_labels.csv --out router_model.pkl
```

Predict groups for new records:

```bash
python predict.py --input test.jsonl --model router_model.pkl --output groups.csv
```

The output CSV has columns `id,group`.
