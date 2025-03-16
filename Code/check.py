from pathlib import Path
import pandas as pd
from itertools import combinations
import numpy as np
from tabulate import tabulate

idx_files = { 
    "PN_1_1": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/PN_1_1",
    "TF_split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TF_split",
    "TF_TG_split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TF_TG_split",
    "TG_split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TG_split"
}

files = {
    "train": "Train_set.tsv",
    "val": "Val_set.tsv",
    "test": "Test_set.tsv"
}

# wo_iea = "/home/llan/Desktop/WUR/thesis2/GO/go_wo_iea_n.txt"
# wo_comp = "/home/llan/Desktop/WUR/thesis2/GO/go_wo_comp_n.txt"
# def to_tensor(txt):
#     tensoren = []
#     idxs = []
#     with open(txt, "r") as f:
#         vals = []
#         for line in f:
#             idx, val = line.strip().split("\t")
#             val = val.split(" ")
#             idx = int(idx)
#             val = np.array(val).astype(int)
#             vals.extend(val)
#             idxs.append(idx)
#     return idxs


ath_path = Path("/home/llan/Desktop/WUR/thesis2/LABELS/Regulations_in_ATRM.tsv")
print(ath_path.exists())
ath = pd.read_table(ath_path, index_col=0)

def label_check(df):
    num_pos_labels = df.loc[df["Label"] == 1].shape[0]
    count = 0
    for i, (tf, tg, l) in df.iterrows():
        present = ath.loc[(ath["TF index"] == tf) & (ath["Target index"] == tg)].shape[0]
        if present > 0:
            count +=1
    return num_pos_labels, count

missing_idxs = set()

# wo_iea = to_tensor(wo_iea)
# wo_comp = to_tensor(wo_comp)

def tf_overlap(df1, df2):
    empty_dict = {True: 0, False: 0}
    tf_num = df1.TF.isin(df2.TF).value_counts().to_dict()
    for key in tf_num.keys():
        empty_dict[key] = tf_num[key]

    return list(empty_dict.values())
    tf_yes = len(set(df1.TF.values).intersection(df2.TF.values))
    tf_no = len(set(df1.TF.values).difference(df2.TF.values))
    return tf_yes, tf_no

def tg_overlap(df1, df2):
    empty_dict = {True: 0, False: 0}
    tg_yes = len(set(df1.Target.values).intersection(df2.Target.values))
    tg_no = len(set(df1.Target.values).difference(df2.Target.values))
    tf_num = df1.Target.isin(df2.Target).value_counts().to_dict()
    for key in tf_num.keys():
        empty_dict[key] = tf_num[key]

    return list(empty_dict.values())
    return tg_yes, tg_no

overlap = []
for idx_set, file1 in idx_files.items():
    print(25*"#", idx_set, 25*"#", "\n")
    idx_path = Path(file1)
    data = {}
    for dataset, file2 in files.items():
        data_path = idx_path.joinpath(file2)
        data[dataset] = pd.read_table(data_path, index_col=0)

       
        # missing_idxs.update(data[dataset].loc[~data[dataset].TF.isin(wo_iea)].TF)
        # missing_idxs.update(data[dataset].loc[~data[dataset].Target.isin(wo_iea)].Target)

        # missing_idxs.update(data[dataset].loc[~data[dataset].TF.isin(wo_comp)].TF)
        # missing_idxs.update(data[dataset].loc[~data[dataset].Target.isin(wo_comp)].Target)

        # missing_idxs.update(data[dataset].TF.isin(to_tensor(wo_iea)).value_counts())
        # missing_idxs.update(data[dataset].Target.isin(to_tensor(wo_iea)).value_counts())
        # missing_idxs.update(data[dataset].TF.isin(to_tensor(wo_comp)).value_counts())
        # missing_idxs.update(data[dataset].Target.isin(to_tensor(wo_comp)).value_counts())

    combis = combinations(data.keys(), 2)

# missing_idxs = pd.Series(list(missing_idxs))
# missing_idxs.to_csv("missing_idxs.tsv", sep="\t")

    for key1, key2 in combis:
        print()
        print(f"combination: {key1} vs {key2}")
        print(f"Correct labels{key1}:")
        n, c = label_check(data[key1])
        print(f"{key1}: {c}/{n}")
        print(f"Correct labels {key2}:")
        n, c = label_check(data[key2])
        print(f"{key1}: {c}/{n}")
        
    
        df1, df2 = data[key1], data[key2]
        print("TF OVERLAP:")
        tf_num = df1.TF.isin(df2.TF).value_counts().reset_index()
        print(tabulate(tf_num, headers="keys", tablefmt="grid"), "\n")
        print("TARGET OVERLAP")
        tg_num = df1.Target.isin(df2.Target).value_counts().reset_index()
        print(tabulate(tg_num, headers="keys", tablefmt="grid"), "\n")

        tf = tf_overlap(df1, df2)
        tg = tg_overlap(df1, df2)

        overlap.append([idx_set, key1, key2, 
                        tf[0], tf[1], tg[0], tg[1]
                        ])
        

df = pd.DataFrame(overlap)
df.columns = ["Dataset", "set1", "set2", "tf_u", "tf_d", "tg_u", "tg_d"]
print(df.head())
# df.to_csv("overlap_df.tsv", sep="\t")

