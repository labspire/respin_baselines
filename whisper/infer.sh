#!/bin/bash

set -e

languages=('bengali' 'bhojpuri' 'chattisgarhi' 'magahi' 'hindi' 'kannada' 'marathi' 'maithili' 'telugu')
lang_id=('bengali' 'hindi' 'hindi' 'hindi' 'hindi' 'kannada' 'marathi' 'hindi' 'telugu')
test_path=('test_bn' 'test_bh' 'test_ch' 'test_mg' 'test_hi' 'test_kn' 'test_mr' 'test_mt' 'test_te')
root_path='/data/root/'
checkpoint_dir="checkpoints/"
model='whisper-base'


for i in {0..8}
do
    
    

    checkpoint_path=($(ls ${checkpoint_dir}${model}/${languages[$i]}))
    checkpoint_path=${checkpoint_dir}${model}/${languages[$i]}/${checkpoint_path[0]}

    
    dialects=`cat ${root_path}${test_path[$i]}/utt2dial | cut -f 2  | sort | uniq`
    for dialect in $dialects
    do

        echo "Testing for ${languages[$i]} with checkpoint ${checkpoint_paths[$i]}"
        python infer.py --testpath ${root_path}${test_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --checkpoint ${checkpoint_paths[$i]} --model ${model} --dialect ${dialect}
    done

    echo "Testing for ${languages[$i]} with checkpoint ${checkpoint_path}"
    python infer.py --testpath ${root_path}${test_path[$i]} --langid ${lang_id[$i]} --language ${languages[$i]} --checkpoint ${checkpoint_paths[$i]} --model ${model}
done