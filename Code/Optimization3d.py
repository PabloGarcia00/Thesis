import torch 
from torch.utils.data import DataLoader
import pandas as pd 
from Data import CreateDataset
from Model import OptimizationModel
import numpy as np 
import os 
from Utils import init_weights, stack_vector, EarlyStopping, ConfigSampler3D
from itertools import product
from pathlib import Path 
from sklearn.metrics import accuracy_score
import time
import pickle

NDRAWS = 16_200
EPOCHS = 100
NUM_CLASS = 2
RUNS = 5
verbose = False
MEDIA = Path("/home/llan/Desktop/WUR/thesis2/media2/binary/optimization6")


if not MEDIA.exists():
    MEDIA.mkdir(parents=True)

def check_output_dir():
    if any(MEDIA.iterdir()):
        raise ValueError("MEDIA dir already contains output files")
        
check_output_dir()


os.chdir("/home/llan/Desktop/WUR/thesis2")

# Load in the files containing the embeddings
g2v_data = torch.load("gene2vec_data/dataset/g2v_1000emb.pt", map_location=torch.device("cuda"), weights_only=True)
exp_data = torch.load("EXP/expression.pt", weights_only=True, map_location=torch.device("cuda"))
go_data = torch.load("GO/GO_tensor.pt", map_location=torch.device("cuda"), weights_only=True)

# Load in the index files 
idx_files = { 
    "TFTG-split": "/home/llan/Desktop/WUR/thesis2/shared_data/binary_labels/BaseLine/TF_TG_split",
}



def print_model(model):
    # model = model(100, NUM_GO, NUM_CLASS, COMBINED_SIZE)
    print(25 * "#", model.name, 25 * "#", "\n")
    print(model, "\n")
    del(model)


def read_labels(path, shuffle_labels=False):
    df = pd.read_table(path, index_col=0)
    if shuffle_labels:
        df["Label"] = df["Label"].sample(frac=1.0).values
    return df 


with open("media2/binary/optimization3/tried_configs.txt", "rb") as f:
    old_configs = pickle.load(f)

start = time.time()

champs = {"TFTG-split": [None, None, None, None]}

test_stats = {"Dataset": [], "Enc Layers (n)": [], "FF Layers (n)":[], "Hidden size":[], "LR": [], "Aggregation": [], "Run": [], "TF": [], "Target": [], "Label": [], "Pred": []}

configsampler = ConfigSampler3D()

for i in range(NDRAWS):
    print("#" * 25, i, "#" * 25, "\n")
    config = configsampler.sample_config()
    print(config)

    for data_dir in idx_files.keys():
        print(data_dir, "\n")
        # train_stats = {"Dataset": [], "Run": [], "Epoch": [], "Batch": [], "Loss": []}
        # val_stats = {"Dataset": [], "Run": [], "Epoch": [], "Mean_loss": [], "Accuracy": []}
                

        dataset = Path(idx_files[data_dir])
        train_df = read_labels(dataset.joinpath(Path("Train_set.tsv")))
        val_df = read_labels(dataset.joinpath(Path("Val_set.tsv")))
        test_df = read_labels(dataset.joinpath(Path("Test_set.tsv")))


        train_loader = DataLoader(CreateDataset(train_df), shuffle=True, batch_size=config["batch_size"])
        val_loader = DataLoader(CreateDataset(val_df), batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(CreateDataset(test_df), batch_size=config["batch_size"], shuffle=False)    

        
        model = OptimizationModel(config["embed_size"], NUM_CLASS, config["n_encoder_layers"], 
                                  config["n_ffout_layers"], config["activation_function"],
                                  config["dropout"], config["batch_normalization"],
                                  config["aggregation"])
        model = model.to("cuda")

        for run in range(RUNS):
                        
            model.apply(init_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            crit = torch.nn.BCELoss()
            early_stop = EarlyStopping(patience=2, delta=0.01)

            for epoch in range(EPOCHS):
                model.train()
                train_losses = []
                for bi, ((tfs, tgs), label) in enumerate(train_loader):
                    label = label.float().to("cuda")
                    g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])
                    go = stack_vector(go_data[tfs], go_data[tgs])
                    exp = stack_vector(exp_data[tfs], exp_data[tgs])

                    prediction = model(None, g2v, go, exp)
                    optimizer.zero_grad()
                    loss = crit(prediction.view(-1, 1), label.view(-1, 1))
                    train_losses.append(loss.cpu().detach().item())
                    loss.backward()
                    optimizer.step()
                    status = f"batch: {bi:>3} | loss: {loss.item():.4f}"
                    if verbose:
                        print(status)


                l = len(train_loader)
                # train_stats["Dataset"].extend([data_dir] * l)
                # # train_stats["Model"].extend([model.name] * l)
                # train_stats["Run"].extend([run] * l)
                # train_stats["Epoch"].extend([epoch] * l)
                # train_stats["Batch"].extend(list(range(l)))
                # train_stats["Loss"].extend(train_losses)
                mean_train_loss = np.array(train_losses).mean()

                model.eval()
                val_losses = []
                val_predictions = []

                for (tfs, tgs), label in val_loader:
                    with torch.no_grad():
                        label = label.float().to("cuda")
                        g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])
                        go = stack_vector(go_data[tfs], go_data[tgs])
                        exp = stack_vector(exp_data[tfs], exp_data[tgs])

                        prediction = model(None, g2v, go, exp)
                        val_predictions.extend(prediction.cpu().detach().tolist())

                        loss = crit(prediction.view(-1, 1), label.view(-1, 1))
                        val_losses.append(loss.item())

                # val_stats["Dataset"].append(data_dir)
                # # val_stats["Model"].append(model.name)
                # val_stats["Run"].append(run)
                # val_stats["Epoch"].append(epoch)

                
                val_predictions = (np.array(val_predictions) >= 0.5).astype(int)
                accuracy = accuracy_score(val_df.Label, val_predictions)
                # val_stats["Accuracy"].append(accuracy)

                mean_val_loss = np.array(val_losses).mean()
                # val_stats["Mean_loss"].append(mean_val_loss)
                print(f"Run: {run} | Epoch: {epoch:>3} | mean loss (train/val): {mean_train_loss:.4f}/{mean_val_loss:.4f} | accuracy: {accuracy:.10f}")
                early_stop(mean_val_loss)

                if early_stop.early_stop:
                    print("early stopping")
                    break
            
            test_predictions = []

            model.eval()
            for (tfs, tgs), label in test_loader:
                with torch.no_grad():
                    label = label.float().to("cuda")
                    g2v = stack_vector(g2v_data[tfs], g2v_data[tgs])
                    go = stack_vector(go_data[tfs], go_data[tgs])
                    exp = stack_vector(exp_data[tfs], exp_data[tgs])

                    prediction = model(None, g2v, go, exp)
                    test_predictions.extend(prediction.cpu().detach().squeeze().tolist())
            
            l = test_df.shape[0]
            test_stats["Dataset"].extend([data_dir] * l)
            # test_stats["Model"].extend([model.name]* l)
            test_stats["Run"].extend([run] * l)
            test_stats["TF"].extend(test_df.TF.to_list())
            test_stats["Target"].extend(test_df.Target.tolist())
            test_stats["Label"].extend(test_df.Label.tolist())
            test_stats["Pred"].extend(test_predictions)
            test_stats["LR"].extend([config["learning_rate"]] * l)
            test_stats["Aggregation"].extend([config["aggregation"]] * l)
            test_stats["FF Layers (n)"].extend([config["n_ffout_layers"]] * l)
            test_stats["Enc Layers (n)"].extend([config["n_encoder_layers"]] * l)
            test_stats["Hidden size"].extend([config["embed_size"]] * l)

        with torch.no_grad():
            prediction_tensor = torch.tensor(test_predictions, dtype=torch.float32)
            label_tensor = torch.tensor(test_df.Label.tolist(), dtype=torch.float32)
            test_loss = crit(prediction_tensor, label_tensor).item()
            print(test_loss)
            test_predictions = (np.array(test_predictions) >= 0.5).astype(int)
            accuracy = accuracy_score(test_df.Label, test_predictions)
            print("accuracy:", accuracy)
        
        # if champs[data_dir][1] is None or test_loss < champs[data_dir][1]:
        #     print(f"new model for {data_dir}, old loss: {champs[data_dir][1]}, new loss: {test_loss}, elapsed time: {time.time() - start}", "\n")
        #     champs[data_dir][0] = model
        #     #champs[data_dir][1] = test_loss
        #     champs[data_dir][2] = [train_stats, val_stats]
        #     champs[data_dir][3] = config
        


# for key, val in champs.items():
#     if val[0] is not None:
#         key_dir = MEDIA.joinpath(Path(key))
#         key_dir.mkdir()
#         model = val[0]
#         torch.save(model, key_dir.joinpath("model.pt"))
#         with open(key_dir.joinpath("configuration.txt"), "w") as f:
#             for key, param in val[3].items():
#                 line = f"{key}:\t{param}\n"
#                 f.write(line)

#         names = ["train_stats.tsv", "val_stats.tsv", "test_stats.tsv"]
    
#         train_stats = pd.DataFrame.from_dict(val[2][0])
#         val_stats = pd.DataFrame.from_dict(val[2][1])
#         test_stats = pd.DataFrame.from_dict(test_stats)

#         zipped = zip(names, [train_stats, val_stats, test_stats])
#         for name, df in zipped:
#             output_path = key_dir.joinpath(name)
#             df.to_csv(output_path, sep="\t")

# with open(MEDIA.joinpath("tried_configs.txt"), "wb") as f:
#     pickle.dump(configsampler.sampled_configs, f)

test_stats = pd.DataFrame.from_dict(test_stats)
test_stats.to_csv(MEDIA.joinpath("test_stats.tsv"), sep="\t")

end = time.time()
print(f"Finished in: {end - start}")


                
        
        

  

         









