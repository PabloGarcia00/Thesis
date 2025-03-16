import os 
os.chdir("/home/llan/Desktop/WUR/thesis2/WangyuchenCS-scGREAT-ec4cec6")
import sys
sys.path.append("../Code")
from Utils import EarlyStopping
from Utils import init_weights
from pathlib import Path
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dataset import Dataset
from model import scGREAT
from train_val import train,validate



# Model parameters
Batch_size = 32
Embed_size = 768
Num_layers = 2
Num_head = 4
LR = 1e-5
EPOCHS = 50
RUNS = 10
# step_size = args.step_size
# gamma = args.gamma
global schedulerflag
schedulerflag = False
RANDOM = True


MEDIA = Path("../media2/binary/scgreat/random")
if not MEDIA.exists():
    MEDIA.mkdir(parents=True)

# Load in the index files 
idx_files = { 
    "Overlap": "Datasets/Overlap",
    "TF-split": "Datasets/TF-split",
    "TFTG-split": "Datasets/TFTG-split",
    "TG-split": "Datasets/TG-split"
}


def check_output_dir():
    names = ["train_stats.tsv", "val_stats.tsv", "test_stats.tsv"]
    for name in names:
        if MEDIA.joinpath(name).exists():
            raise ValueError("MEDIA dir already contains the output files")
        
check_output_dir()

train_stats = {"Dataset": [], "Run": [], "Epoch": [], "Batch": [], "Loss": []}
val_stats = {"Dataset": [], "Run": [], "Epoch": [], "Mean_loss": [], "Accuracy": []}
test_stats = {"Dataset": [], "Run": [], "TF": [], "Target": [], "Label": [], "Pred": []}


# Data Preprocessing
expression_data_path = "hESC500" + '/BL--ExpressionData.csv'
biovect_e_path       = "hESC500" + '/biovect768.npy'
expression_data = np.array(pd.read_csv(expression_data_path,index_col=0,header=0))

standard = StandardScaler()
scaled_df = standard.fit_transform(expression_data.T)
expression_data = scaled_df.T
expression_data_shape = expression_data.shape 

if RANDOM:
    print("RAMDOM")
    expression_data = pd.read_csv("hESC500/RandomExpressionData.csv", index_col=0, header=0).to_numpy()
    biovect_e_path = "hESC500/randomvect768.npy"





start = time.time()

for name in idx_files.keys():
    data_dir = idx_files[name]
    train_data_path      = data_dir + '/Train_set.csv'
    val_data_path        = data_dir + '/Val_set.csv'
    test_data_path       = data_dir + '/Test_set.csv'

 
    train_dataset = Dataset(train_data_path, expression_data)
    val_dataset = Dataset(val_data_path, expression_data)
    test_dataset = Dataset(test_data_path, expression_data)

  

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             shuffle=False,
                                             drop_last=False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=False,
                                              drop_last=False)


    model = scGREAT(expression_data_shape,Embed_size,Num_layers,Num_head,biovect_e_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("#" * 25, data_dir, "#" * 25, "\n")
 
    for run in range(RUNS):
        init_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.BCELoss() 
        early_stop = EarlyStopping()

        for epoch in range(EPOCHS):
            train_losses = []
            model.train()
            for idx, (gene_pair_index,expr_embedding,label) in enumerate(train_loader):
                label = label.to(device)
                gene_pair_index = gene_pair_index.to(device)
                expr_embedding = expr_embedding.to(torch.float32)
                expr_embedding = expr_embedding.to(device)
                optimizer.zero_grad()
                predicted_label = model(gene_pair_index,expr_embedding)
                loss = loss_func(predicted_label.squeeze(), label.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                train_losses.append(loss.item())
            l = len(train_loader)
            train_stats["Dataset"].extend([data_dir] * l)
            train_stats["Run"].extend([run] * l)
            train_stats["Epoch"].extend([epoch] * l)
            train_stats["Batch"].extend(list(range(l)))
            train_stats["Loss"].extend(train_losses)
            mean_train_loss = np.array(train_losses).mean()

            model.eval()
            val_losses = []
            val_predictions = []

            for idx, (gene_pair_index,expr_embedding,label) in enumerate(val_loader):
                with torch.no_grad():
                    label = label.to(device).squeeze()
                    gene_pair_index = gene_pair_index.to(device)
                    expr_embedding = expr_embedding.to(torch.float32)
                    expr_embedding = expr_embedding.to(device)
                    predicted_label = model(gene_pair_index,expr_embedding)
                    loss = loss_func(predicted_label.squeeze(), label.float())
                    val_losses.append(loss.item())
                    val_predictions.append(predicted_label.squeeze().item())

            val_stats["Dataset"].append(data_dir)
            val_stats["Run"].append(run)
            val_stats["Epoch"].append(epoch)

            val_predictions = (np.array(val_predictions) >= 0.5).astype(int)
            accuracy = accuracy_score(val_dataset.label.tolist(), val_predictions)
            val_stats["Accuracy"].append(accuracy)

            mean_val_loss = np.array(val_losses).mean()
            val_stats["Mean_loss"].append(mean_val_loss)
            print(f"Run: {run} | Epoch: {epoch:>3} | mean loss (train/val): {mean_train_loss:.4f}/{mean_val_loss:.4f} | accuracy: {accuracy:.4f}")
            early_stop(mean_val_loss)             
 
            if early_stop.early_stop:
                 break
            
            model.eval()
            test_predictions = []

            for idx, (gene_pair_index,expr_embedding,label) in enumerate(test_loader):
                with torch.no_grad():
                    label = label.to(device)
                    gene_pair_index = gene_pair_index.to(device)
                    expr_embedding = expr_embedding.to(torch.float32)
                    expr_embedding = expr_embedding.to(device)
                    predicted_label = model(gene_pair_index,expr_embedding)
                    test_predictions.append(predicted_label.squeeze().item())
                
                 
            l = len(test_dataset)
            test_stats["Dataset"].extend([data_dir] * l)
            test_stats["Run"].extend([run] * l)
            test_stats["TF"].extend(test_dataset.label.tolist())
            test_stats["Target"].extend(test_dataset.label.tolist())
            test_stats["Label"].extend(test_dataset.label.tolist())
            test_stats["Pred"].extend(test_predictions)

train_stats = pd.DataFrame.from_dict(train_stats)
val_stats = pd.DataFrame.from_dict(val_stats)
test_stats = pd.DataFrame.from_dict(test_stats)  

names = ["train_stats.tsv", "val_stats.tsv", "test_stats.tsv"]
zipped = zip(names, [train_stats, val_stats, test_stats])
                     
for name, df in zipped:
    output_path = MEDIA.joinpath(name)
    df.to_csv(output_path, sep="\t")

end = time.time()
print(f"Finished in: {end - start}")