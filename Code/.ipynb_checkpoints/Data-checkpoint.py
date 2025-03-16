from collections import defaultdict
from functools import reduce
from torch.utils.data import Dataset
import torch
from Train import train
import func
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import itertools


def create_expression_data(path, sample_data, output, mapped_unique=0.9, ecotype="Col-0", genotype="wild type"):
    # load expression data & sample information
    df = dd.read_csv(path, blocksize=int(25e6), sample=int(1e6))
    sample_data = pd.read_csv(sample_data, header=0, sep="\t")

    # filter sample information to only include hq samples and wildtype information
    if mapped_unique:
        to_include = (sample_data.uniquelymapped >= 0.9)
    if ecotype:
        to_include = (to_include & (sample_data.ecotype == "Col-0"))
    if genotype:
        to_include = (to_include & (sample_data.genotype == "wild type"))

    columns_include = sample_data[to_include.values]["sample"].tolist()

    df = df[["Sample"] + columns_include]
    df.index = df.Sample
    df = df.drop(columns="Sample")
    df = df.repartition(npartitions=1)
    df.to_csv(output, single_file=True, sep="\t")


def create_label_data(regulations, genetoi, output_dir):
    # regulations
    reg = pd.read_excel(regulations, header=0, usecols=["TF ID", "Target ID"])

    # Labels.tsv
    def f(x): return genetoi[x]
    reg = reg.map(f)

    # to tsv
    reg.to_csv(output_dir + "/" + "Labels.tsv", sep="\t")


def create_tf_n_tg_list(tf_list, expression, out_dir):
    # list of tfs single gene model
    tfs = pd.read_table(tf_list, sep="\t", header=0)
    tfs = tfs.Gene_ID.unique().tolist()
    tfs_set = set(tfs)

    # expression
    df = pd.read_csv(expression, sep="\t", header=0, usecols=[0])
    df = df["Sample"].tolist()
    df_set = set(df)

    # conversion dict
    genetoi = {x: y for x, y in zip(df, range(len(df)))}

    # TF list
    idx_tf = [genetoi[x] for x in tfs]
    pd.DataFrame({"ID": tfs, "idx": idx_tf}).sort_values("idx").to_csv(
        out_dir + "/" + "TF_list.tsv", sep="\t", index=False)

    # Gene list
    tgs = list(df_set.difference(tfs_set))
    idx_tg = [genetoi[x] for x in tgs]
    pd.DataFrame({"ID": list(tgs), "idx": idx_tg}).sort_values(
        "idx").to_csv(out_dir + "/" + "TG_list.tsv", sep="\t", index=False)

    return genetoi


class _DataSet(Dataset):
    def __init__(self,  data_path,  expression_data):
        data = pd.read_csv(data_path,  sep='\t',  index_col=0, header=0)
        self.dataset = np.array(data.iloc[:, :2])
        label = np.array(data.iloc[:, -1])
        self.label = np.eye(2)[label]
        self.expression_data = expression_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,  i):
        gene_pair_index = self.dataset[i]
        gene1_expr = np.expand_dims(
            self.expression_data[gene_pair_index[0]], axis=0)
        gene2_expr = np.expand_dims(
            self.expression_data[gene_pair_index[1]], axis=0)
        expr_embedding = np.concatenate((gene1_expr, gene2_expr), axis=0)
        label = self.label[i]
        return gene_pair_index, expr_embedding, label[-1]


class GoDataSet(Dataset):
    def __init__(self, data_path, go_data):
        data = pd.read_csv(data_path,  sep='\t',  index_col=0, header=0)
        self.dataset = np.array(data.iloc[:, :2])
        label = np.array((data.iloc[:, -1]))
        self.label = np.eye(2)[label]

        go_data = pd.read_table(go_data, names=["idx", "go"], dtype={
                                0: int}, index_col=0)
        go_data = go_data.go.apply(lambda x: list(
            map(int, str(x).split()))).to_dict()
        self.go_data = go_data
        self.n_go = len(
            set(itertools.chain.from_iterable(self.go_data.values())))

    def __len__(self):
        return self.dataset.shape[0]

    def _to_one_hot(self, go):
        go = torch.tensor(go, dtype=torch.long)
        go = torch.nn.functional.one_hot(go, num_classes=self.n_go)

        go = go.sum(dim=0)

        assert torch.any(
            go > 1) == False, "one hot contains values larger than 1"

        return go

    def __getitem__(self, i):
        gene_pair_index = self.dataset[i]
        gene1_go = self._to_one_hot(self.go_data[gene_pair_index[0]])
        gene2_go = self._to_one_hot(self.go_data[gene_pair_index[1]])
        go = torch.stack((gene1_go, gene2_go), dim=0)
        label = self.label[i]
        return gene_pair_index, go, label[-1]


class g2vDataSet(Dataset):
    def __init__(self, data_path, g2v_data):
        data = pd.read_csv(data_path,  sep='\t',  index_col=0, header=0)
        self.dataset = np.array(data.iloc[:, :2])
        label = np.array((data.iloc[:, -1]))
        self.label = np.eye(2)[label]

        g2v_data = pd.read_table(g2v_data, sep="\t", index_col=0)
        g2v_data = g2v_data.apply(list, axis=1).to_dict()
        self.g2v_data = g2v_data

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, i):
        gene_pair_index = self.dataset[i]
        gene1 = torch.Tensor(self.g2v_data[gene_pair_index[0]])
        gene2 = torch.Tensor(self.g2v_data[gene_pair_index[1]])
        go = torch.stack((gene1, gene2), dim=0)
        label = self.label[i]
        return gene_pair_index, go, label[-1]
    

class CombinedData(Dataset):
    def __init__(self, data_path, g2v_data, go_data, exp_data):
        data = pd.read_csv(data_path,  sep='\t',  index_col=0, header=0)
        self.dataset = np.array(data.iloc[:, :2])
        label = np.array((data.iloc[:, -1]))
        self.n_classes = len(set(label)) 
        self.label = np.eye(self.n_classes)[label]

        g2v_data = pd.read_table(g2v_data, sep="\t", index_col=0)
        g2v_data = g2v_data.apply(list, axis=1).to_dict()
        self.g2v_data = g2v_data

        go_data = pd.read_table(go_data, names=["idx", "go"], dtype={
                                0: int}, index_col=0)
        go_data = go_data.go.apply(lambda x: list(
            map(int, str(x).split()))).to_dict()
        self.go_data = go_data
        self.n_go = len(
            set(itertools.chain.from_iterable(self.go_data.values())))

        self.expression_data = exp_data

    def __len__(self):
        return self.dataset.shape[0]

    def _to_one_hot(self, go):
        go = torch.tensor(go, dtype=torch.long)
        go = torch.nn.functional.one_hot(go, num_classes=self.n_go)

        go = go.sum(dim=0)

        assert torch.any(
            go > 1) == False, "one hot contains values larger than 1"

        return go

    def __getitem__(self, i):
        gene_pair_index = self.dataset[i]
        
        gene1_g2v = torch.Tensor(self.g2v_data[gene_pair_index[0]])
        gene2_g2v = torch.Tensor(self.g2v_data[gene_pair_index[1]])
        g2v = torch.stack((gene1_g2v, gene2_g2v), dim=0)

        gene1_exp = torch.Tensor(self.expression_data[gene_pair_index[0], :])
        gene2_exp = torch.Tensor(self.expression_data[gene_pair_index[1], :]) 
        exp = torch.stack((gene1_exp, gene2_exp), dim=0)

        gene1_go = self._to_one_hot(self.go_data[gene_pair_index[0]])
        gene2_go = self._to_one_hot(self.go_data[gene_pair_index[1]])
        go = torch.stack((gene1_go, gene2_go), dim=0)
        
        if self.n_classes > 2:
            label = self.label[i]
        else:
            label = self.label[i][-1]
        return gene_pair_index, label, g2v, go, exp






if __name__ == "__main__":
    SEED = 2010
    func.seed_everything(SEED)
    # create_expression_data("EXP/OriginalData/gene_FPKM_200501.csv", "EXP/OriginalData/sample_data.tsv", "EXP/expression.tsv")
    # genetoi = create_tf_n_tg_list("EXP/OriginalData/Ath_TF_list.txt", "EXP/expression.tsv", "LABELS")
    # create_label_data("EXP/OriginalData/Regulations_in_ATRM.xlsx", genetoi, "LABELS")

    LABEL_DIR = "shared_data/"

    label = pd.read_csv(LABEL_DIR + "Labels.tsv", sep="\t", index_col=0)
    tf_set = pd.read_csv(LABEL_DIR + "TF_list.tsv", sep="\t").idx.values
    tg_set = pd.read_csv(LABEL_DIR + "TG_list.tsv",
                         sep="\t").idx.values

    hard_negative_sampling(label, tf_set, tg_set,
                           "shared_data/PN_1_1", pn_factor=1)
