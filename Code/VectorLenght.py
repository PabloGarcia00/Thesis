import torch 
from torch.utils.data import DataLoader
import pandas as pd 
from Data import CreateDataset
from Model import G2VMOD
import numpy as np 
import os 
from Utils import init_weights, stack_vector, EarlyStopping
from pathlib import Path 
from sklearn.metrics import accuracy_score
import time



BATCHSIZE = 32
EPOCHS = 10
LR = 0.0001
EMBED_SIZE = 100
NUM_GO = 7247
NUM_CLASS = 2 
COMBINED_SIZE = 200
RUNS = 10
verbose = False
MEDIA = Path("/home/llan/Desktop/WUR/thesis2/media2/base_line/vector_lengt_integer")

print(f"""
      LR ={LR}
      EPOCH = {EPOCHS}
      EMBED_SIZE = {EMBED_SIZE}
      MEDIA = {MEDIA}
      BATCHSIZE = {BATCHSIZE}
      """
      )

if not MEDIA.exists():
    MEDIA.mkdir(parents=True)

def check_output_dir():
    names = ["train_stats.tsv", "val_stats.tsv", "test_stats.tsv"]
    for name in names:
        if MEDIA.joinpath(name).exists():
            raise ValueError("MEDIA dir already contains the output files")
        
check_output_dir()

os.chdir("/home/llan/Desktop/WUR/thesis2")

# # Load in the files containing the embeddings
# g2v_data = torch.load("gene2vec_data/dataset/g2v_tensor.pt", map_location=torch.device("cuda"), weights_only=True)
# exp_data = torch.load("EXP/expression.pt", weights_only=True, map_location=torch.device("cuda"))
# go_data = torch.load("GO/GO_tensor.pt", map_location=torch.device("cuda"), weights_only=True)

# vector_sizes = pd.read_table("media2/base_line/vector_length/vector_sizes.tsv",
#                               usecols=[3]).values.squeeze()
vector_sizes = np.linspace(0, 2000, 400, dtype=int)

print(vector_sizes)
# Load in the index files 
idx_files = { 
    "Overlap": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/PN_1_1",
    "TF-split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TF_split",
    "TFTG-split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TF_TG_split",
    "TG-split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TG_split"
}


def print_model(model):
    model = model(100, NUM_GO, NUM_CLASS, 200)
    print(25 * "#", model.name, 25 * "#", "\n")
    print(model, "\n")
    del(model)

def read_labels(path, shuffle_labels=False):
    df = pd.read_table(path, index_col=0)
    if shuffle_labels:
        df["Label"] = df["Label"].sample(frac=1.0).values
    return df 


train_stats = {"Dataset": [], "VectorSize": [], "Run": [], "Epoch": [], "Batch": [], "Loss": []}
val_stats = {"Dataset": [], "VectorSize": [], "Run": [], "Epoch": [], "Mean_loss": [], "Accuracy": []}
test_stats = {"Dataset": [], "VectorSize": [], "Run": [], "TF": [], "Target": [], "Label": [], "Pred": []}

start = time.time()

for vector_size in vector_sizes:
    g2v_data = torch.randint(0, 2, (37336, vector_size)).to("cuda")
    model = G2VMOD(EMBED_SIZE, NUM_GO, NUM_CLASS, 200, vector_size).to("cuda")

    for data_dir in idx_files.keys():
        dataset = Path(idx_files[data_dir])
        train_df = read_labels(dataset.joinpath(Path("Train_set.tsv")))
        val_df = read_labels(dataset.joinpath(Path("Val_set.tsv")))
        test_df = read_labels(dataset.joinpath(Path("Test_set.tsv")))

        train_loader = DataLoader(CreateDataset(train_df), shuffle=True, batch_size=BATCHSIZE) 
        val_loader = DataLoader(CreateDataset(val_df), batch_size=BATCHSIZE)
        test_loader = DataLoader(CreateDataset(test_df), batch_size=BATCHSIZE)


        print("#" * 25, data_dir, "&", model.name, "at size", vector_size, "#" * 25, "\n")

        for run in range(RUNS):
                        
            model.apply(init_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            crit = torch.nn.BCELoss()
            early_stop = EarlyStopping(patience=2, delta=0.01)

            for epoch in range(EPOCHS):
                model.train()
                train_losses = []
                for bi, ((tfs, tgs), label) in enumerate(train_loader):
                    label = label.float().to("cuda")
                    g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])
            
                    prediction = model(None, g2v, None, None)
                    optimizer.zero_grad()
                    loss = crit(prediction.view(-1, 1), label.view(-1, 1))
                    train_losses.append(loss.cpu().detach().item())
                    loss.backward()
                    optimizer.step()
                    status = f"batch: {bi:>3} | loss: {loss.item():.4f}"
                    if verbose:
                        print(status)


                l = len(train_loader)
                train_stats["Dataset"].extend([data_dir] * l)
                train_stats["VectorSize"].extend([vector_size] * l)
                train_stats["Run"].extend([run] * l)
                train_stats["Epoch"].extend([epoch] * l)
                train_stats["Batch"].extend(list(range(l)))
                train_stats["Loss"].extend(train_losses)
                mean_train_loss = np.array(train_losses).mean()

                model.eval()
                val_losses = []
                val_predictions = []

                for (tfs, tgs), label in val_loader:
                    with torch.no_grad():
                        label = label.float().to("cuda")
                        g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])

                        prediction = model(None, g2v, None, None)
                        val_predictions.extend(prediction.cpu().detach().tolist())

                        loss = crit(prediction.view(-1, 1), label.view(-1, 1))
                        val_losses.append(loss.item())

                val_stats["Dataset"].append(data_dir)
                val_stats["VectorSize"].append(vector_size)
                val_stats["Run"].append(run)
                val_stats["Epoch"].append(epoch)
    
                
                val_predictions = (np.array(val_predictions) >= 0.5).astype(int)
                accuracy = accuracy_score(val_df.Label, val_predictions)
                val_stats["Accuracy"].append(accuracy)

                mean_val_loss = np.array(val_losses).mean()
                val_stats["Mean_loss"].append(mean_val_loss)
                print(f"Run: {run} | Epoch: {epoch:>3} | mean loss (train/val): {mean_train_loss:.4f}/{mean_val_loss:.4f} | accuracy: {accuracy:.4f}")
                early_stop(mean_val_loss)

                if early_stop.early_stop:
                    print("early stopping")
                    break

            
            test_predictions = []
            for (tfs, tgs), label in test_loader:
                with torch.no_grad():
                    label = label.float().to("cuda")
                    g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])

                    prediction = model(None, g2v, None, None)
                    test_predictions.extend(prediction.cpu().detach().squeeze().tolist())
            
            l = test_df.shape[0]
            test_stats["Dataset"].extend([data_dir] * l)
            test_stats["VectorSize"].extend([vector_size]* l)
            test_stats["Run"].extend([run] * l)
            test_stats["TF"].extend(test_df.TF.to_list())
            test_stats["Target"].extend(test_df.Target.tolist())
            test_stats["Label"].extend(test_df.Label.tolist())
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

