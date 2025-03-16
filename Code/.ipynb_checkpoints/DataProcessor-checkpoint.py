import pandas as pd
import numpy as np
import os
from collections import defaultdict


class DatasetProcessor:
    def __init__(self, train_test_ratio: float = 0.67):
        self.train_test_ratio = train_test_ratio
        self.binary_class_ratio = 1
        self.p_val = 0.5
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def _label_data_to_dict(self, label_data: pd.DataFrame) -> dict:
        pos_dict = defaultdict(list)
        for _, (tf, target) in label_data.iterrows():
            if tf != target:
                pos_dict[tf].append(target)
        return pos_dict

    def _neg_dict_from_pos_dict(self, pos_dict: dict, tg_genes: list) -> dict:
        neg_dict = dict()
        tg_genes = set(tg_genes)
        for tf, targets in pos_dict.items():
            targets = set(tg_genes).difference(targets)
            neg_dict[tf] = list(targets)
        return neg_dict

    def _distribute_pos_dict(self, pos_dict: dict) -> tuple:
        train, val, test = dict(), dict(), dict()

        for tf, targets in pos_dict.items():
            np.random.shuffle(targets)
            if len(targets) == 1:
                p = np.random.uniform(0, 1)
                if p >= self.p_val:
                    train[tf] = targets
                else:
                    test[tf] = targets
            elif len(targets) == 2:
                train[tf] = [targets[0]]
                test[tf] = [targets[1]]
            else:
                n_targets = len(targets)
                n_train = int(n_targets * self.train_test_ratio)
                n_test = int(n_targets * (self.train_test_ratio + 0.1))

                train[tf] = targets[:n_train]
                val[tf] = targets[n_train:n_test]
                test[tf] = targets[n_test:]
        return train, val, test

    def _distribute_neg_dict(self, pos_dict: dict, neg_dict: dict) -> tuple:
        train, val, test = dict(), dict(), dict()

        for tf, targets in pos_dict.items():
            neg_targets = neg_dict[tf]
            n_positives = len(targets)
            n_negatives = int(n_positives * self.binary_class_ratio)
            neg_targets = neg_targets[:n_negatives]
            n_train = int(n_negatives * self.train_test_ratio)
            n_test = int(n_negatives * (self.train_test_ratio + 0.1))
            np.random.shuffle(neg_targets)

            train[tf] = neg_targets[:n_train]
            val[tf] = neg_targets[n_train:n_test]
            test[tf] = neg_targets[n_test:]

        return train, val, test

    def _compile_data(self, data: dict, label: int) -> list:
        dataset = [[tf, target, label]
                   for tf, targets in data.items() for target in targets]
        return dataset

    def split_binary_dataset(self, label_data: pd.DataFrame,
                             targets: list) -> None:
        pos_dict = self._label_data_to_dict(label_data)
        neg_dict = self._neg_dict_from_pos_dict(pos_dict, targets)
        train_p, val_p, test_p = self._distribute_pos_dict(
            pos_dict)
        train_n, val_n, test_n = self._distribute_neg_dict(
            pos_dict, neg_dict)

        self.train_set = np.array(self._compile_data(
            train_p, 1) + self._compile_data(train_n, 0))
        self.val_set = np.array(self._compile_data(
            val_p, 1) + self._compile_data(val_n, 0))
        self.test_set = np.array(self._compile_data(
            test_p, 1) + self._compile_data(test_n, 0))

    def _multiclass_dataset_to_pos_dict(self, label_data: pd.DataFrame):
        pos_dicts = {key: df for key, df in label_data.groupby("Label")}
        for key, df in pos_dicts.items():
            pos_dicts[key] = self._label_data_to_dict(df)
        return pos_dicts

    def _multiclass_neg_dict(self, pos_dicts: dict, tg_list: list) -> dict:
        neg_dicts = dict()
        for key, pos_dict in pos_dicts.items():
            neg_dicts[key] = self._neg_dict_from_pos_dict(pos_dict, tg_list)
        return neg_dicts

    def _multiclass_distribute_pos_dict(self, pos_dicts):
        train, val, test = dict(), dict(), dict()
        for key, pos_dict in pos_dicts.items():
            train[key], val[key], test[key] = self._distribute_pos_dict(
                pos_dict)
        return train, val, test

    def _multiclass_distribute_neg_dict(self, pos_dicts, neg_dicts):
        train, val, test = dict(), dict(), dict()
        for key, pos_dict in pos_dicts.items():
            neg_dict = neg_dicts[key]
            train[key], val[key], test[key] = self._distribute_neg_dict(
                pos_dict, neg_dict)
        return train, val, test

    def _multiclass_compile_dataset(self, dataset: dict,
                                    negative: bool = False) -> list:
        compiled_data = []
        for label, data in dataset.items():
            label = 0 if negative else label
            compiled_data.extend(self._compile_data(data, label))
        return compiled_data

    def split_multiclass_dataset(self, label_data: pd.DataFrame,
                                 targets: list) -> None:
        pos_dicts = self._multiclass_dataset_to_pos_dict(label_data)
        neg_dicts = self._multiclass_neg_dict(pos_dicts, targets)
        train_p, val_p, test_p = self._multiclass_distribute_pos_dict(pos_dicts
                                                                      )
        train_n, val_n, test_n = self._multiclass_distribute_neg_dict(
            pos_dicts,
            neg_dicts,
        )
        self.train_set = np.array(self._multiclass_compile_dataset(train_p) +
                                  self._multiclass_compile_dataset(train_n, True))
        self.val_set = np.array(self._multiclass_compile_dataset(val_p) +
                                self._multiclass_compile_dataset(val_n, True))
        self.test_set = np.array(self._multiclass_compile_dataset(test_p) +
                                 self._multiclass_compile_dataset(test_n, True))

    def to_tsv(self, output_dir: str) -> None:
        names = ["Train_set.tsv", "Val_set.tsv", "Test_set.tsv"]
        dfs = [self.train_set, self.val_set, self.test_set]
        assert any(df is not None for df in dfs), \
            "must fill dataslots first"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        stats = []
        for df, name in zip(dfs, names):
            df = pd.DataFrame(df, columns=["TF", "Target", "Label"])
            df.to_csv(output_dir + "/" + name, sep="\t")

            stat_i = df.Label.value_counts().index.to_list()
            stat_val = df.Label.value_counts().to_list()
            stats.append(stat_val)

            print(stat_i, "/n", stat_val, type(stat_val))
        pd.DataFrame(stats, index=names, columns=stat_i).to_csv(
            output_dir + "/" + "stats.tsv",
            sep="\t")


if __name__ == "__main__":
    os.chdir("/home/llan/Desktop/WUR/thesis2")
    LABEL_DIR = "shared_data/binary_labels/"
    label_data = LABEL_DIR + "Labels.tsv"
    tf_list = LABEL_DIR + "TF_list.tsv"
    tg_list = LABEL_DIR + "TG_list.tsv"

    label_data = pd.read_table(label_data, header=0, index_col=0)
    tf_list = pd.read_table(tf_list, header=0).idx.tolist()
    tg_list = pd.read_table(tg_list, header=0).idx.tolist()

    processor = DatasetProcessor()
    processor.split_binary_dataset(label_data, tg_list)
    print(processor.train_set.shape)
    print(processor.val_set.shape)
    print(processor.test_set.shape)

    mlc_labels = pd.read_table("shared_data/multiclass_labels/Labels.tsv",
                               header=0, index_col=0,
                               usecols=["TF ID", "Target ID", "Label"])
    mlc_processor = DatasetProcessor()
    mlc_processor.split_multiclass_dataset(mlc_labels, tg_list)
    print(mlc_processor.train_set.shape)
    print(mlc_processor.val_set.shape)
    print(mlc_processor.test_set.shape)
    mlc_processor.to_tsv("test")
