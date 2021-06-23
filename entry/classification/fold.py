import os
import argparse
import json
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def main(args):
    study_df = pd.read_csv(args.study_df)
    image_df = pd.read_csv(args.image_df)
    pid = study_df['id'].to_numpy()

    label = study_df[
        [
            'Negative for Pneumonia',
            'Typical Appearance',
            'Indeterminate Appearance',
            'Atypical Appearance'
        ]
    ].to_numpy()

    study_id = np.array([n.split("_")[0] for n in pid])
    label_cls = np.argmax(label, axis=1)
    skf = StratifiedKFold(n_splits=5, random_state=63)

    idx = 0
    fold = {}
    for train_index, test_index in skf.split(study_id, label_cls):
        fold[str(idx)] = study_id[test_index].tolist()
        print(label[test_index].sum(axis=0))
        idx += 1
    with open(args.json_file, 'w') as outfile:
        json.dump(fold, outfile, indent=4)

    dic = {f"{n}": [] for n in range(5)}
    for k in fold.keys():
        for file in fold[k]:
            tmp_df = image_df[image_df["StudyInstanceUID"] == file]
            if len(tmp_df) > 1:
                if tmp_df["boxes"].isnull().sum() != len(tmp_df):
                    tmp = tmp_df[tmp_df.boxes.notnull()]
                    dic[k] += [
                        {
                            "image": img_bs.split("_")[0] + ".png",
                            "label": label[np.where(study_id == file)[0], :].tolist()[0]
                        }
                        for img_bs in tmp.id
                    ]

                    continue

            dic[k] += [
                {
                    "image": img_bs.split("_")[0] + ".png",
                    "label": label[np.where(study_id == file)[0], :].tolist()[0]
                }
                for img_bs in tmp_df.id
            ]

    for i in range(5):
        write_dic = {}
        write_dic["label_format"] = [1, 1, 1, 1]
        with open(f"{args.fold_json}_{i}.json", 'w') as outfile:
            keys = list(dic.keys())
            keys.pop(i)
            write_dic["training"] = [ins for k in keys for ins in dic[k]]
            write_dic["validation"] = dic[f"{i}"]
            json.dump(write_dic, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-df")
    parser.add_argument("--image-df")
    parser.add_argument("--img-dir")
    parser.add_argument("--json_file")
    parser.add_argument("--fold_json")
    args = parser.parse_args()
    main(args)
