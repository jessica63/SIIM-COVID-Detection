import os
import json
import numpy as np
import pandas as pd


def check_uid(uid, train_ls, valid_ls, json_ls, img_id):
    if uid + '_study' in json_ls:
        valid_ls.append(
            f"/work/Lung/SIIM/preprocessed/images/{img_id.split('_')[0]}.png\n"
        )
    else:
        train_ls.append(
            f"/work/Lung/SIIM/preprocessed/images/{img_id.split('_')[0]}.png\n"
    )

    return train_ls, valid_ls


def main():
    csv_file = pd.read_csv('/data2/chest_xray/siim-covid19-detection/train_image_level.csv')
    uid = np.unique(csv_file.StudyInstanceUID.to_numpy())
    train_ls = []
    valid_ls = []
    duplicate = []

    json_file = json.load(open('/data2/chest_xray/siim-covid19-detection/entry/fold.json'))

    for n in uid:
        tmp = csv_file[csv_file['StudyInstanceUID']==n]
        if len(tmp) > 1:
            for j in range(len(tmp)):
                img_id = tmp.iloc[j]['id']

                train_ls, valid_ls = check_uid(
                                        n,
                                        train_ls,
                                        valid_ls,
                                        json_file['0'],
                                        img_id
                                     )

        else:
            img_id = csv_file[csv_file['StudyInstanceUID'] == n]['id'].item()
            train_ls, valid_ls = check_uid(
                                        n,
                                        train_ls,
                                        valid_ls,
                                        json_file['0'],
                                        img_id
                                     )
    print(f'There are {len(duplicate)} duplicate images.')
    print(f'There are {len(train_ls)} images in train set.')
    print(f'There are {len(valid_ls)} images in valid set.')

    with open('/work/Lung/SIIM/train_all_list.txt', 'w') as f:
        f.writelines(train_ls)
    with open('/work/Lung/SIIM/valid_all_list.txt', 'w') as f:
        f.writelines(valid_ls)


if __name__ == '__main__':
        main()
