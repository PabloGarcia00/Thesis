import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd 
import numpy as np

def stack_vector(v1, v2):
    stacked = torch.stack((v1, v2), dim=1)
    return stacked


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)

def init_weights(m):
    if isinstance(m, nn.Linear):
        if isinstance(m, nn.ReLU):
            # He initialization for ReLU layers
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # Initialize best_loss if not done yet
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0       



import random 

class ConfigSampler:
    def __init__(self, old_configs=None):
        self.sampled_configs = set() if not old_configs else old_configs
       
    def sample_config(self):

        while True:
            config = {
                "batch_size": random.choice([16, 32]),
                "embed_size": random.choice([50, 100, 300, 600]),
                "n_encoder_layers": random.randint(1, 9),
                "n_ffout_layers": random.randint(1, 5),
                "activation_function": random.choice(["ReLU", "PReLU", "ELU", "GELU", "Tanh", "LeakyReLU"]),
                "dropout": round(random.uniform(0.0, 0.5), 1),
                "batch_normalization": random.choice([True, False]),
                "aggregation": random.choice([True, False]),
                "learning_rate": random.choice([1e-3, 1e-4, 1e-5]),
            }
            # Convert to tuple for set storage (hashable)
            config_tuple = tuple(config.items())
            if config_tuple not in self.sampled_configs:
                self.sampled_configs.add(config_tuple)
                self.draws = len(self.sampled_configs)
                return config

class ConfigSampler3D:
    def __init__(self, old_configs=None):
        self.sampled_configs = set() if not old_configs else old_configs
       
    def sample_config(self):

        while True:
            config = {
                "batch_size": random.choice([32]),
                "embed_size": random.choice([25, 50, 100, 200, 400, 600, 800]),
                "n_encoder_layers": random.randint(1, 9),
                "n_ffout_layers": random.randint(1, 9),
                "activation_function": random.choice(["ReLU"]),
                "dropout": 0,# round(random.uniform(0.0, 0.5), 1),
                "batch_normalization": random.choice([False]),
                "aggregation": random.choice([True]),
                "learning_rate": random.choice([1e-3, 1e-4, 1e-5]),
            }
            # Convert to tuple for set storage (hashable)
            config_tuple = tuple(config.items())
            if config_tuple not in self.sampled_configs:
                self.sampled_configs.add(config_tuple)
                self.draws = len(self.sampled_configs)
                return config
            

class ConfigSampler3D:
    def __init__(self, old_configs=None):
        self.sampled_configs = set() if not old_configs else old_configs
       
    def sample_config(self):

        while True:
            config = {
                "batch_size": random.choice([32]),
                "embed_size": random.choice([200, 400, 600]),
                "n_encoder_layers": random.randint(1, 6),
                "n_ffout_layers": random.randint(1, 6),
                "activation_function": random.choice(["ReLU", "PReLU","LeakyReLU", "Tanh", "ELU", "GELU"]),
                "dropout": random.choice(np.arange(0, 0.6, 0.1)),# round(random.uniform(0.0, 0.5), 1),
                "batch_normalization": random.choice([False, True]),
                "aggregation": random.choice([True]),
                "learning_rate": random.choice([5e-3, 1e-4, 5e-4]),
            }
            # Convert to tuple for set storage (hashable)
            config_tuple = tuple(config.items())
            if config_tuple not in self.sampled_configs:
                self.sampled_configs.add(config_tuple)
                self.draws = len(self.sampled_configs)
                return config




class Metrics:
    ftrain, fval, ftest = Path("train_stats.tsv"), Path("val_stats.tsv"), Path("test_stats.tsv")

    def format_df(df):
        conv = {"PN_1_1": "Overlap",
                "TF_split": "TF-split",
                "TG_split": "TG-split",
                "TF_TG_split": "TFTG-split"}
        
        df["Dataset"] = df.Dataset.apply(lambda x: conv[x] if x in conv.keys() else x)
        return df 
    
    def __init__(self, dir):
        self.dir = Path(dir)
        self.train = Metrics.format_df(pd.read_table(self.dir.joinpath(Metrics.ftrain), index_col=0))
        self.val = Metrics.format_df(pd.read_table(self.dir.joinpath(Metrics.fval), index_col=0))
        self.test = Metrics.format_df(pd.read_table(self.dir.joinpath(Metrics.ftest), index_col=0))


