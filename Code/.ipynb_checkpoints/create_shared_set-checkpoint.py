import pandas as pd
import os

go_path = "/home/llan/Desktop/WUR/thesis2/GO/GO_data.txt"
g2v_path = "/home/llan/Desktop/WUR/thesis2/gene2vec_data/dataset/g2v_embeddings.tsv"

go = pd.read_csv(go_path, sep="\t", usecols=[0]).iloc[:, 0].tolist()
g2v = pd.read_csv(g2v_path, sep="\t", usecols=[0]).iloc[:,0].tolist()

tf_path = "/home/llan/Desktop/WUR/thesis2/LABELS/TF_list.tsv"
tg_path = "/home/llan/Desktop/WUR/thesis2/LABELS/TG_list.tsv"
label_path = "/home/llan/Desktop/WUR/thesis2/LABELS/Labels.tsv"

tfs = pd.read_csv(tf_path, sep="\t", index_col=0, header=0)
tgs = pd.read_csv(tg_path, sep="\t", index_col=0, header=0)
label = pd.read_csv(label_path, sep="\t", index_col=0, header=0)

intersect = set(go) & set(g2v)
intersect = list(intersect)
print(len(intersect))

tfs = tfs.loc[tfs["idx"].isin(intersect) == True]
tgs = tgs.loc[tgs["idx"].isin(intersect) == True]
print(f"tgs shape = {tgs.shape} and tfs shape = {tfs.shape}")

label = label.loc[
        (label["TF ID"].isin(intersect) == True) &
        (label["Target ID"].isin(intersect) == True)
        ]
print(f"label shape = {label.shape}")

if not os.path.exists("shared_data"):
    os.mkdir("shared_data")

tfs.to_csv("shared_data/TF_list.tsv", sep="\t", index=False)
tgs.to_csv("shared_data/TG_list.tsv", sep="\t", index=False)
label.to_csv("shared_data/Labels.tsv", sep="\t")


