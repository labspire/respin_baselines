
#!/bin/bash

set -e

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --data_path) data_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if required arguments are set
if [[ -z "$model_name" || -z "$data_path" ]]; then
    echo "Usage: $0 --model_name <name> --data_path <path>"
    exit 1
fi



languages=('bengali' 'bhojpuri' 'chattisgarhi' 'magahi' 'hindi' 'kannada' 'marathi' 'maithili' 'telugu')
lang_id=('bengali' 'hindi' 'hindi' 'hindi' 'hindi' 'kannada' 'marathi' 'hindi' 'telugu')
train_path=('train_bn' 'train_bh' 'train_ch' 'train_mg' 'train_hi' 'train_kn' 'train_mr' 'train_mt' 'train_te')
dev_path=('dev_bn' 'dev_bh' 'dev_ch' 'dev_mg' 'dev_hi' 'dev_kn' 'dev_mr' 'dev_mt' 'dev_te')
root_path=${data_path}
model=${model_name}

mkdir -p exp

for i in {0..1}
do
    echo "Training for ${languages[$i]}"
    CUDA_VISIBLE_DEVICES="0" accelerate launch --gpu_ids 0 --num_processes 1 train.py --trainpath ${root_path}${train_path[$i]} --evalpath ${root_path}${dev_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --model ${model} 

done