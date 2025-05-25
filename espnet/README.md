# ESPnet ASR Baselines for RESPIN-S1.0

This repository provides baseline Automatic Speech Recognition (ASR) models trained using [ESPnet](https://github.com/espnet/espnet) on the RESPIN-S1.0 multilingual corpus. The baselines cover nine Indian languages, and training is orchestrated using ESPnet2.

> ✅ Trained models are hosted here: [Hugging Face Model Repo](https://huggingface.co/SpireLab/spire_respin_baselines_espnet)  
> ▶️ Training script lives here: [run_baseline_small.sh](https://github.com/labspire/espnet_respin_baseline/blob/respin_baselines/egs2/respin_small/asr1/run_baseline_small.sh)

---

## 🗃 Dataset: RESPIN-S1.0

RESPIN (REgional SPeaker and INtelligibility) is a multilingual ASR corpus covering:
- Bhojpuri (bh)
- Bengali (bn)
- Chhattisgarhi (ch)
- Hindi (hi)
- Kannada (kn)
- Magahi (mg)
- Marathi (mr)
- Maithili (mt)
- Telugu (te)

The `small` training subset is used in this baseline.

---

## 🧠 Model Architecture

All models use the [E-Branchformer](https://doi.org/10.48550/arXiv.2210.00077)-based encoder architecture with the following configuration:

- **Encoder**: 8-layer E-Branchformer with hidden size 256 and MLP size 1024
- **Decoder**: Transformer decoder
- **Loss**: Hybrid CTC-Attention with intermediate CTC
- **Training tricks**: SpecAugment, label smoothing, gradient accumulation

---

## 📦 Pretrained Models

Pretrained models for all 9 languages are available at:

🔗 [https://huggingface.co/SpireLab/spire_respin_baselines_espnet](https://huggingface.co/SpireLab/spire_respin_baselines_espnet)

Each language folder contains:
- `config.yaml`: ESPnet2 configuration
- `*.pth`: trained model weights
- `RESULTS.md`: evaluation metrics (CER / WER)
- `meta.yaml`: model card metadata

---

## 📈 Results Summary

| Language | Model Path                                      | CER (%) | WER (%) |
|----------|--------------------------------------------------|---------|---------|
| BH       | exp_small/exp_bh/asr_bh_...                      | 4.4     | 15.2    |
| BN       | exp_small/exp_bn/asr_bn_...                      | 5.1     | 14.3    |
| CH       | exp_small/exp_ch/asr_ch_...                      | 4.8     | 13.9    |
| HI       | exp_small/exp_hi/asr_hi_...                      | 4.2     | 12.7    |
| KN       | exp_small/exp_kn/asr_kn_...                      | 5.5     | 14.8    |
| MG       | exp_small/exp_mg/asr_mg_...                      | 5.0     | 13.5    |
| MR       | exp_small/exp_mr/asr_mr_...                      | 4.7     | 13.2    |
| MT       | exp_small/exp_mt/asr_mt_...                      | 5.3     | 14.1    |
| TE       | exp_small/exp_te/asr_te_...                      | 4.9     | 13.7    |

*Refer to `RESULTS.md` in each subfolder for complete details.*

---

## 🏁 How to Train

The full training pipeline is defined in:

📄 [`run_baseline_small.sh`](https://github.com/labspire/espnet_respin_baseline/blob/respin_baselines/egs2/respin_small/asr1/run_baseline_small.sh)

To run it:

```bash
git clone https://github.com/labspire/espnet_respin_baseline
git checkout respin_baselines
cd egs2/respin_small/asr1
bash run_baseline_small.sh --lang bh  # Replace with other lang codes
```

This script handles:
- Data preprocessing
- Feature extraction
- Training
- Decoding and evaluation
- Model upload (Hugging Face)

---