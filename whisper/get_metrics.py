import jiwer
import os
import argparse
import pandas as pd
import numpy as np

def main():

    
    language_map = {
        "bh": "bhojpuri",
        "bn": "bengali",
        "ch": "chattisgarhi",
        "hi": "hindi",
        "kn": "kannada",
        "mg": "magahi",
        "mr": "marathi",
        "mt": "maithili",
        "te": "telugu"
    }
    
    
    dialects_list = ["D1", "D2", "D3", "D4", "D5", "all"]
    
    root_dir = "transcriptions/"
    models = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    
    metrics_dir = "metrics/"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    
    for model_name in models:
        languages = [os.path.join(model_name, x) for x in os.listdir(model_name)]
        save_dict = {"Language": []}
        
        for lang_id, language in language_map.items():
            if language not in [x.split("/")[-1] for x in languages]:
                continue
            
            save_dict["Language"].append(lang_id)
            language = os.path.join(model_name, language)
            checkpoint = os.listdir(language)[0]
            dialects = [os.path.join(language, checkpoint, x) for x in os.listdir(os.path.join(language, checkpoint)) if ".txt" not in x]
            # print(dialects)
            for dialect in dialects_list:
                
                if dialect + "_CER" not in save_dict.keys():
                    save_dict[dialect + "_CER"] = []
                    save_dict[dialect + "_WER"] = []
                
                
                if dialect not in [x.split("/")[-1] for x in dialects]:
                    save_dict[dialect + "_CER"].append("-")
                    save_dict[dialect + "_WER"].append("-")
                    continue
                
                
                    
                
                dialect = os.path.join(language, checkpoint, dialect)
                
                gt_file = "GT.txt"
                pred_file = "HYP.txt"
                

                with open(os.path.join(dialect, gt_file)) as fp:
                    gt = [x.strip() for x in fp.readlines()]

                with open(os.path.join(dialect, pred_file)) as fp:
                    hyp = [x.strip() for x in fp.readlines()]

                wer = jiwer.wer(gt, hyp)
                cer = jiwer.cer(gt, hyp)

                # save_dict["Model"].append(f"{model_name.split('/')[-1]}__{language.split('/')[-1]}__{dialect.split('/')[-1]}")
                # save_dict["WER"].append(np.round(wer * 100, 2))
                # save_dict["CER"].append(np.round(cer * 100, 2))
                
                save_dict[dialect.split('/')[-1] + "_CER"].append(np.round(cer * 100, 2))
                save_dict[dialect.split('/')[-1] + "_WER"].append(np.round(wer * 100, 2))
                
        print(save_dict)
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in save_dict.items() ]))
        df.to_csv(os.path.join(metrics_dir, model_name.split("/")[-1] + "_results.csv"), index=False)


if __name__ == '__main__':
    main()
