## Demo
[Bhojpuri](https://bhashini.gov.in/ulca/search-model/67eb8f9a7d193a1beb4b86c5)


### Models

All the below model weights can be found in the [huggingface repository](https://huggingface.co/SpireLab/RESPIN/tree/main/finetuned)

#### Data2Vec-AQC Models
| Model         | Pre-training Data | Fine-tuning Data | Model Links                                                                 |
|---------------|-------------------|------------------|-----------------------------------------------------------------------------|
| data2vec-aqc  | ---               | Bhojpuri         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/bh_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Bengali          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/bn_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Chhattisgarhi    | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/ch_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Hindi            | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/hi_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Kannada          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/kn_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Magahi           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/mg_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Marathi          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/mr_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Maithili         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/mt_checkpoint_best.pt) |
| data2vec-aqc  | ---               | Telugu           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/data2vec-aqc/te_checkpoint_best.pt) |

#### Indic Models
| Model         | Pre-training Data | Fine-tuning Data | Model Links                                                                 |
|---------------|-------------------|------------------|-----------------------------------------------------------------------------|
| indic         | ---               | Bhojpuri         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/bh_checkpoint_best.pt) |
| indic         | ---               | Bengali          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/bn_checkpoint_best.pt) |
| indic         | ---               | Chhattisgarhi    | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/ch_checkpoint_best.pt) |
| indic         | ---               | Hindi            | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/hi_checkpoint_best.pt) |
| indic         | ---               | Kannada          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/kn_checkpoint_best.pt) |
| indic         | ---               | Magahi           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/mg_checkpoint_best.pt) |
| indic         | ---               | Marathi          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/mr_checkpoint_best.pt) |
| indic         | ---               | Maithili         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/mt_checkpoint_best.pt) |
| indic         | ---               | Telugu           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/indic/te_checkpoint_best.pt) |

#### Spring Models
| Model         | Pre-training Data | Fine-tuning Data | Model Links                                                                 |
|---------------|-------------------|------------------|-----------------------------------------------------------------------------|
| spring        | ---               | Bhojpuri         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/bh_checkpoint_best.pt) |
| spring        | ---               | Bengali          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/bn_checkpoint_best.pt) |
| spring        | ---               | Chhattisgarhi    | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/ch_checkpoint_best.pt) |
| spring        | ---               | Hindi            | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/hi_checkpoint_best.pt) |
| spring        | ---               | Kannada          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/kn_checkpoint_best.pt) |
| spring        | ---               | Magahi           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/mg_checkpoint_best.pt) |
| spring        | ---               | Marathi          | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/mr_checkpoint_best.pt) |
| spring        | ---               | Maithili         | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/mt_checkpoint_best.pt) |
| spring        | ---               | Telugu           | [fairseq](https://huggingface.co/SpireLab/RESPIN/resolve/main/finetuned/spring/te_checkpoint_best.pt) |



* finetuning procedures can be found [here](https://github.com/labspire/respin_baselines/tree/main/fairseq/recipes/Training).
* Inference procedures can be found [here](https://github.com/labspire/respin_baselines/tree/main/fairseq/recipes/Inference).
* Single file inference procedures can be found [here](https://github.com/labspire/respin_baselines/tree/main/fairseq/recipes/Single_File_Infer)

## Directory Structure
```
RESPIN
├── configs
│   └── finetuned ── data2vec-aqc.yaml
├── data
│   ├── examples
│   └── bh
├── models
│   ├── finetuned
│   │   └── indic_finetuned
│   └── pretrained
├── recipes
│   ├── Training
│   │   └── train.sh
│   ├── Inference
│   │   └── infer.sh
│   └── fairseq_preprocessing
│       ├── data_prep.py
│       ├── metrics.py
│       └── run_data_prep.sh
├── requirements.txt
└── README.md
```

## Requirements and Installation
* Create a new conda environment:
```bash
conda create -n env_name python=3.10
conda activate env_name
```
* Python version >= 3.10
* [PyTorch](https://pytorch.org/) version >= 2.0.0
* Fairseq version >= 0.12.2
* CUDA >= 11.8
* For training new models, you'll also need an NVIDIA GPU and NCCL
* To install requirements:

```bash
pip install -r requirements.txt
```
* To install fairseq and develop locally (ubuntu):

``` bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --editable ./
```
* For data2vec-aqc 
``` bash
git clone https://github.com/Speech-Lab-IITM/data2vec-aqc
cd data2vec-aqc/
pip install --editable ./
```
* For faster training install NVIDIA's apex library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* For Augmentation to work install [torchaudio-augmentations](https://github.com/Speech-Lab-IITM/torchaudio-augmentations):
```bash
git clone https://github.com/Speech-Lab-IITM/torchaudio-augmentations
cd torchaudio-augmentations
pip install --editable ./ 
```

* Flashlight version >= 0.0.7
* To install [flashlight-text](https://github.com/flashlight/text) and [flashlight-sequence](https://github.com/flashlight/sequence)
```bash
pip install flashlight-text

git clone https://github.com/flashlight/sequence && cd sequence
pip install .
```
* To install parse_options:
```bash
wget https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/wsj/s5/utils/parse_options.sh && sudo mv parse_options.sh /usr/local/bin/

```
<b>Required Step

Add the musan dataset path in: <br>
data2vec-aqc/fairseq/data/audio/raw_audio_dataset.py <br>
```python
path_to_musan_noise_set = 'path_to_musan_dataset'
```

* For musan dataset.
[Musan](https://www.openslr.org/resources/17/musan.tar.gz)


## Reference Code
1. Facebook AI Research Sequence-to-Sequence Toolkit written in Python. [fairseq](https://github.com/facebookresearch/fairseq)
2. SPRING-LAB ([data2vec_aqc](https://asr.iitm.ac.in/models))
3. OpenSLR [musan](https://www.openslr.org/17/)



