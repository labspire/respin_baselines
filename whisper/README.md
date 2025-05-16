# Whisper training and inference pipeline

This repository provides a pipeline for training and inference using Whisper models
---

## 🛠️ Data Preparation

Prepare your dataset in **Kaldi-style format** with separate `train`, `dev`, and `test` directories.

Each directory must contain the following files:

- `wav.scp`: Maps utterance IDs to audio file paths.
- `text`: Maps utterance IDs to phoneme or text transcriptions.
- `utt2dial`: Maps utterance IDs to dialect labels.

---

## 🚀 Training

To train the model, run:

```
bash train.sh --model_name <whisper-small|whisper-base|whisper-tiny> --data_path <path_to_kaldi_data>
```
The checkpoints and logs will be stored in exp directory

## 🚀 Inference

Download and copy the checkpoints/ directory from the Hugging Face repository:  https://huggingface.co/SpireLab/Whisper
```
bash infer.sh --model_name <whisper-small|whisper-base|whisper-tiny> --data_path <path_to_kaldi_data>
```
The transcriptions will be stored in transcriptions directory

##  Metrics

To evaluate the model performance run:
```
python get_metrics.py
```
This will calculate the Word Error Rate (WER) and Character Error Rate (CER) both per dialect and overall, and save the results in the metrics/ directory as: `metrics/<model_name>_results.csv`

