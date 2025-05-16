
#!/bin/bash

set -e

languages=('bengali' 'bhojpuri' 'chattisgarhi' 'magahi' 'hindi' 'kannada' 'marathi' 'maithili' 'telugu')
lang_id=('bengali' 'hindi' 'hindi' 'hindi' 'hindi' 'kannada' 'marathi' 'hindi' 'telugu')
train_path=('train_bn' 'train_bh' 'train_ch' 'train_mg' 'train_hi' 'train_kn' 'train_mr' 'train_mt' 'train_te')
dev_path=('dev_bn' 'dev_bh' 'dev_ch' 'dev_mg' 'dev_hi' 'dev_kn' 'dev_mr' 'dev_mt' 'dev_te')
root_path='/data/root/'
model='whisper-base'

for i in {0..1}
do
    echo "Training for ${languages[$i]}"
    CUDA_VISIBLE_DEVICES="0" accelerate launch --gpu_ids 0 --num_processes 1 train.py --trainpath ${root_path}${train_path[$i]} --evalpath ${root_path}${dev_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --model ${model} 

done