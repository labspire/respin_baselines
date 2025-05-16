#!/bin/bash

set -e

model_name=""
data_path=""

# Parse arguments
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
test_path=('test_bn' 'test_bh' 'test_ch' 'test_mg' 'test_hi' 'test_kn' 'test_mr' 'test_mt' 'test_te')
root_path=${data_path}
checkpoint_dir="checkpoints/"
model=${model_name}


for i in {0..8}
do
    
    

    checkpoint_path=($(ls ${checkpoint_dir}${model}/${languages[$i]}))
    checkpoint_path=${checkpoint_dir}${model}/${languages[$i]}/${checkpoint_path[0]}

    
    dialects=`cat ${root_path}${test_path[$i]}/utt2dial | cut -f 2  | sort | uniq`
    for dialect in $dialects
    do

        echo "Testing for ${languages[$i]} with checkpoint ${checkpoint_paths[$i]}"
        python infer.py --testpath ${root_path}${test_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --checkpoint ${checkpoint_path} --model ${model} --dialect ${dialect}
    done

    echo "Testing for ${languages[$i]} with checkpoint ${checkpoint_path}"
    python infer.py --testpath ${root_path}${test_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --checkpoint ${checkpoint_path} --model ${model}
done