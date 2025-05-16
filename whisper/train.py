from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
# from datasets import Audio
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import EarlyStoppingCallback, IntervalStrategy

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import librosa
import argparse
import random



class DataLoaderCustom(torch.utils.data.Dataset):
    def __init__(self, set, data_path, processor, sr=16000):

        self.sr = sr
        with open(os.path.join(data_path, "wav.scp")) as fp:
            wav_lines = [x.strip() for x in fp.readlines()]

        self.wavs = {x.split()[0]:x.split()[1] for x in wav_lines}
        self.ids = [x.split()[0] for x in wav_lines]

        with open(os.path.join(data_path, "text")) as fp:
            self.text = {x.split()[0]: " ".join(x.split()[1:]).strip() for x in fp.readlines()}


    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        f_id = self.ids[idx]
        wav_array, sr = librosa.load(self.wavs[f_id], sr=self.sr)
        wav_len = len(wav_array)
        text = self.text[f_id]
        text_len = len(text)
        
        

        # return {"id": f_id, "wav": wav_array, "wav_len": wav_len, "text": text, "text_len": text_len}
        return [f_id, wav_array, wav_len, text, text_len]
    
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        feats = processor.feature_extractor([data[1] for data in batch], sampling_rate=16000)
        feats_padded = processor.feature_extractor.pad(feats, return_tensors="pt")
        tokens = processor.tokenizer([data[3] for data in batch])

        tokens_padded = processor.tokenizer.pad(tokens, return_tensors="pt")

        tokens_padded = tokens_padded["input_ids"].masked_fill(tokens_padded.attention_mask.ne(1), -100)

        if (tokens_padded[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            tokens_padded = tokens_padded[:, 1:]

        tokens_padded = tokens_padded


        batch = {}
        batch = {k:v for k,v in feats_padded.items()}
        batch["labels"] = tokens_padded

        
        return batch
    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)


    print("Predictions: ", pred_str[5:10])
    print("Labels: ", label_str[5:10])
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        
        logits = logits
    print(logits)
    return logits.argmax(dim=-1)


def filter_ids(ids, text_dict, processor):
    filtered_ids = []
    
    for k in ids:
        if len(processor.tokenizer(text_dict[k], return_tensors="pt").input_ids[0]) < 440:
            filtered_ids.append(k)
            
    return filtered_ids


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="whisper-tiny")
parser.add_argument("--language", type=str, default="marathi")
parser.add_argument("--langid", type=str, default="marathi")
parser.add_argument("--trainpath", type=str, default="/home1/jesuraj/asr/new_split/train_mr_s12345/")
parser.add_argument("--evalpath", type=str, default="/home1/jesuraj/asr/new_split/dev_mr_nt/")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_processes", type=str, default=1)
parser.add_argument("--gpu_ids", type=str, default=1)


args = parser.parse_args()



whisper_model = args.model 
language = args.language
lang_id = args.langid

train_path = args.trainpath
eval_path = args.evalpath

checkpoint = args.checkpoint

processor = WhisperProcessor.from_pretrained("openai/"+ whisper_model, language=lang_id, task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")


model = WhisperForConditionalGeneration.from_pretrained("openai/" + whisper_model)
# model.config.forced_decoder_ids = None
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_id, task="transcribe")
# model.config.max_length = 512





train_dataset = DataLoaderCustom("train", train_path, processor=processor)
eval_dataset = DataLoaderCustom("eval", eval_path, processor=processor)


output_dir = os.path.join("results", whisper_model,language)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=48,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=100000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
)

# checkpoint = "/home1/jesuraj/whisper/results/whisper-tiny/bn-en/checkpoint-11000/"

trainer.train()


