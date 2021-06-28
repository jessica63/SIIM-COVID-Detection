import os
import json
import argparse
import numpy as np
import pandas as pd


def check_uid(uid, train_ls, valid_ls, json_ls, img_id, img_dir):
    if uid in json_ls:
        valid_ls.append(f"{img_dir}{img_id.split('_')[0]}.png\n")
    else:
        train_ls.append(f"{img_dir}{img_id.split('_')[0]}.png\n")

    return train_ls, valid_ls


def main(args):
    csv_file = pd.read_csv(args.train_csv)
    uid = np.unique(csv_file.StudyInstanceUID.to_numpy())
    train_ls = []
    valid_ls = []
    duplicate = []

    json_file = json.load(open(args.fold_json))

    for n in uid:
        tmp = csv_file[csv_file['StudyInstanceUID'] == n]
        if len(tmp) > 1:
            if tmp['boxes'].isnull().sum() == len(tmp):
                for j in range(len(tmp)):
                    img_id = tmp.iloc[j]['id']
                    train_ls, valid_ls = check_uid(
                                            n,
                                            train_ls,
                                            valid_ls,
                                            json_file[f"{args.fold}"],
                                            img_id,
                                            args.img_dir
                                         )
                continue

            for i in range(len(tmp)):
                img_id = tmp.iloc[i]['id']
                if tmp.iloc[i]['label'][:4] == 'none':
                    duplicate.append(img_id)
                else:
                    train_ls, valid_ls = check_uid(
                                        n,
                                        train_ls,
                                        valid_ls,
                                        json_file[f"{args.fold}"],
                                        img_id,
                                        args.img_dir
                                     )

        else:
            img_id = csv_file[csv_file['StudyInstanceUID'] == n]['id'].item()
            train_ls, valid_ls = check_uid(
                                        n,
                                        train_ls,
                                        valid_ls,
                                        json_file[f"{args.fold}"],
                                        img_id,
                                        args.img_dir
                                     )
    print(f'There are {len(duplicate)} duplicate images.')
    print(f'There are {len(train_ls)} images in train set.')
    print(f'There are {len(valid_ls)} images in valid set.')

    with open(f"/work/Lung/SIIM/train_list_{args.fold}.txt", 'w') as f:
        f.writelines(train_ls)
    with open(f"/work/Lung/SIIM/valid_list_{args.fold}.txt", 'w') as f:
        f.writelines(valid_ls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv")
    parser.add_argument("--fold")
    parser.add_argument("--fold_json")
    parser.add_argument("--img_dir")
    args = parser.parse_args()
    main(args)
