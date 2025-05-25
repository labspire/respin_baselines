# respin_baselines
This repository contains list of baselines which are trained on RESPIN datasets, language wise.
The folders contain steps to train and run the inference on the best models whose model weights are uploaded in huggingface.

All the pretrained best performing model weights for different fairseq-based models can be found in this hugging face repository: [https://huggingface.co/SpireLab/RESPIN](https://huggingface.co/SpireLab/RESPIN)

ESPnet-based models can be found here: [https://huggingface.co/SpireLab/spire_respin_baselines_espnet](https://huggingface.co/SpireLab/spire_respin_baselines_espnet).


# 📄 Kaldi Data Preparation from RESPIN Metadata

This script processes RESPIN metadata stored in a JSON file and generates Kaldi-style data files.

## 🧩 Description

Given a RESPIN `meta_*.json` file, this script generates the following Kaldi-format files:

- `wav.scp`: Mapping from utterance ID to waveform path
- `utt2spk`: Mapping from utterance ID to speaker ID (first 5 fields of utterance ID)
- `spk2utt`: Reverse mapping from speaker ID to utterance IDs
- `text`: Mapping from utterance ID to transcription
- `utt2lang`: Mapping from utterance ID to language ID
- `utt2dialect`: Mapping from utterance ID to dialect ID
- `utt2dur`: Mapping from utterance ID to audio duration (in seconds)

## 🔧 Arguments

| Argument        | Description                                    | Required |
|----------------|------------------------------------------------|----------|
| `--meta_json`   | Path to the RESPIN metadata JSON file         | ✅       |
| `--output_dir`  | Directory where output Kaldi files are saved  | ✅       |

## 🚀 Example Usage

```bash
python data_prep.py \
  --meta_json chhattisgarhi/IISc_RESPIN_train_ch_small/meta_train_ch_small.json \
  --output_dir data_respin/ch/train_ch_small

```
---
