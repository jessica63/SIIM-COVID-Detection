import os
import json
import argparse
import numpy as np
import pandas as pd


def check_fold(uid, fold_info, img_id, label, write_dic, cnt):
    for key, study_id in fold_info.items():
        if uid in study_id:
            write_dic[key] += [
                    {
                        "image": img_id.split("_")[0] + ".png",
                        "label": [label]
                    }
                ]
            cnt[key][label] += 1

    return write_dic, cnt


def main(args):
    csv_file = pd.read_csv(args.image_df)
    uid = np.unique(csv_file.StudyInstanceUID.to_numpy())

    json_file = json.load(open(args.json_file))
    dic = {f"{k}": [] for k in json_file.keys()}
    cnt = {f"{k}": [0, 0] for k in json_file.keys()}

    for n in uid:
        tmp = csv_file[csv_file['StudyInstanceUID'] == n]
        if len(tmp) > 1:
            if tmp['boxes'].isnull().sum() == len(tmp):
                for j in range(len(tmp)):
                    img_id = tmp.iloc[j]['id']
                    label = 1
                    dic, cnt = check_fold(n, json_file, img_id, label, dic, cnt)
                continue

            for i in range(len(tmp[tmp.boxes.notnull()])):
                img_id = tmp.iloc[i]['id']
                label = 0
                dic, cnt = check_fold(n, json_file, img_id, label, dic, cnt)

        else:
            img_id = tmp['id'].item()
            if tmp['boxes'].isnull().item():
                label = 1
            else:
                label = 0
            dic, cnt = check_fold(n, json_file, img_id, label, dic, cnt)

    for i in range(5):
        write_dic = {}
        write_dic["label_format"] = [1]
        with open(f"{args.fold_json}_{i}.json", 'w') as outfile:
            keys = list(dic.keys())
            keys.pop(i)
            write_dic["training"] = [ins for k in keys for ins in dic[k]]
            write_dic["validation"] = dic[f"{i}"]
            json.dump(write_dic, outfile, indent=4)
    print(cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-df")
    parser.add_argument("--json_file")
    parser.add_argument("--fold_json")
    args = parser.parse_args()
    main(args)
