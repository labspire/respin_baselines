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
from torch.utils.data import DataLoader
from tqdm import tqdm


class DataLoaderCustom(torch.utils.data.Dataset):
    def __init__(self, ids, data_path, sr=16000, processor=None, dialect=""):

        self.sr = sr
        with open(os.path.join(data_path, "wav.scp")) as fp:
            wav_lines = [x.strip() for x in fp.readlines()]
            
            
        
        if dialect != "":
            with open(os.path.join(data_path, "utt2dial")) as fp:
                dial_ids = [x.split()[0] for x in fp.readlines() if x.split()[1].strip() == dialect]

            self.wavs = {x.split()[0]:x.split()[1] for x in wav_lines if x.split()[0] in dial_ids}
        
        else:
            self.wavs = {x.split()[0]:x.split()[1] for x in wav_lines}
            
        
    
        with open(os.path.join(data_path, "text")) as fp:
            self.text = {x.split()[0]: " ".join(x.split()[1:]).strip() for x in fp.readlines()}
        
        self.ids = [id_name for id_name in ids if id_name in self.wavs.keys() and id_name in self.text.keys()]
        # self.ids = filter_ids(self.ids, self.text, processor)
        print("Number of ids: ", len(self.ids))
        self.processor = processor
        


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        f_id = self.ids[idx]
        wav_array, sr = librosa.load(self.wavs[f_id], sr=self.sr)
        wav_len = len(wav_array)
        text = self.text[f_id]
        text_len = len(text)
        token_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids
        
        # print(f_id, librosa.get_duration(filename=self.wavs[f_id]), text, len(token_ids[0]))
        

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

        batch_padded = {}
        batch_padded = {k:v for k,v in feats_padded.items()}
        batch_padded["raw_text"] = [data[3] for data in batch]
        batch_padded["labels"] = tokens_padded
        batch_padded["id"] = [data[0] for data in batch]

        # print(batch_padded["raw_text"])
        return batch_padded




parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="whisper-tiny")
parser.add_argument("--language", type=str, default="chhattisgarhi")
parser.add_argument("--langid", type=str, default="hindi")
parser.add_argument("--testpath", type=str, default="/home1/Sathvik/espnet_datasets/RESPIN_FINAL_DATASET/test/test_ch_nt/")
parser.add_argument("--checkpoint", type=str, default="/home1/jesuraj/whisper/results/whisper-tiny/chhattisgarhi/checkpoint-33000")
parser.add_argument("--num_processes", type=str, default=1)
parser.add_argument("--dialect", type=str, default="")

args = parser.parse_args()



whisper_model = args.model 
language = args.language
lang_id = args.langid

test_path = args.testpath
dialect = args.dialect

checkpoint = args.checkpoint
trans_dir = f"/home1/jesuraj/whisper/transcriptions/{args.model}/"


processor = WhisperProcessor.from_pretrained("openai/"+ whisper_model, language=lang_id, task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_id, task="transcribe")
model = model.cuda()
collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor)



with open(os.path.join(test_path, "text")) as fp:
    wav_lines = [x.strip() for x in fp.readlines()]
    
ids = [x.split()[0] for x in wav_lines]

test_dataset = DataLoaderCustom(ids, test_path, processor=processor, dialect=dialect)
test_dataloader = DataLoader(test_dataset, 
                            shuffle=False,
                            batch_size=1, 
                            collate_fn=collate_fn)


gt_str = ""
preds = ""
ids = ""


if not os.path.exists(os.path.join(trans_dir, language, checkpoint.split("/")[-1])):
    os.makedirs(os.path.join(trans_dir, language, checkpoint.split("/")[-1]), exist_ok=True)

    
dialect_str = "all" if dialect == "" else dialect
save_dir = os.path.join(trans_dir, language, checkpoint.split("/")[-1], dialect_str)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


for batch in tqdm(test_dataloader):
    # print(batch.keys())
    predicted_ids = model.generate(batch["input_features"].cuda())
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # print(processor.batch_decode(predicted_ids, skip_special_tokens=False)[0])
        # print(batch["raw_text"][idx])
    gt_str += "\n".join(batch["raw_text"])
    gt_str += "\n"
    preds += "\n".join(transcription)
    preds += "\n"
    ids += "\n".join(batch["id"])
    ids += "\n"


# fin_dir = checkpoint.split("/")[-1]

    
with open(os.path.join(save_dir, "GT.txt"), "w", encoding='utf-8') as fp:
    fp.write(gt_str)

with open(os.path.join(save_dir, "HYP.txt") , "w", encoding='utf-8') as fp:
    fp.write(preds)

with open(os.path.join(save_dir, "ID.txt") , "w", encoding='utf-8') as fp:
    fp.write(ids)



