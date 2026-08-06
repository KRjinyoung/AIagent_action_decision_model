# Group4 Final RoBERTa FP16 Package

This package contains the final Group4 specialist model and the code needed to reproduce training.

## Files

- `features.py`: A5 text template used by the final model.
- `predict.py`: Full-data training entry point.
- `requirements.txt`: Extra dependency not included in the confirmed server package list.
- `model_group4/`: Final HuggingFace RoBERTa model artifact.

## Included Model

- model: `klue/roberta-base`
- labels: `edit_file`, `write_file`, `apply_patch`
- training data: full `train_group4`
- feature template: `a5`
- epochs: `10`
- class weight power: `0.5`
- warmup ratio: `0.10`
- max length: `480`
- saved dtype: `float16`
- final source path: `output/group4_final_a5_w0p5_e10_warmup10_fp16/full/model_group4`

## Validation Reference

The selected setting was validated with 5-fold group CV:

- mean macro F1: `0.9817648493`
- mean bias-tuned macro F1: `0.9819880786`
- worst fold macro F1: `0.9778901462`
- mean apply_patch F1: `0.971104`

The final included model is trained on all Group4 rows, so it does not include a validation score in its own `metrics.json`.

## Reproduce Final Training

```bash
python predict.py \
  --data-dir /root/workspace/dacon_inc/data \
  --mode full \
  --epochs 10 \
  --class-weight-power 0.5 \
  --warmup-ratio 0.10 \
  --save-fp16 \
  --output group4_final
```

The reproduced model will be saved to:

```text
group4_final/model_group4/
```

## FP16 Check

`model_group4/config.json` should contain:

```json
"torch_dtype": "float16"
```
