import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Data import _DataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_dead_neurons(activation):
    # Counts the number of dead neurons (zeros)
    dead_neurons = (activation == 0).sum().item()
    total_neurons = activation.numel()
    dead_percentage = (dead_neurons / total_neurons) * 100
    return dead_percentage



class RegPred(nn.Module):

    def __init__(self, expression_data_shape, embed_size, num_layers, num_head,
                 ):
        # dimension reduction using auto encoder
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        self.AE_encoder = nn.Sequential(
            nn.Linear(expression_data_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embed_size),
        )

        # transformer definition
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_head,
            batch_first=True)
        self. transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers)

        # position embedding
        self.position_embedding = nn.Embedding(2, embed_size)

        # dense layer
        self.dense = nn.Sequential(
            nn.Linear(2*embed_size, 72),
            nn.Dropout(),
            nn.PReLU(),
            nn.Linear(72, 36),
            nn.Dropout(),
            nn.PReLU(),
            nn.Linear(36, 1)
        )

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, gene_pair_index, expr_embedding):
        expr_embedding = expr_embedding.to(torch.float32)
        bs = expr_embedding.shape[0]
        # position = torch.Tensor([0, 1]*bs).reshape(bs,
        #                                           -1).to(torch.int32).to(device)
        # p_e = self.position_embedding(position).to(self.device)
        expr_embedding = expr_embedding.to(self.device)
        gene_pair_index = gene_pair_index.to(self.device)

        z = self.AE_encoder(expr_embedding)
        gene2vec = None
        # transformer_input = torch.add(z, p_e)
        transformer_output = self.transformer_encoder(z)
        flattened = self.flatten(transformer_output)
        logits = self.dense(flattened)
        prob = self.sigmoid(logits)

        return prob


class LinearModel(nn.Module):
    def __init__(self, embed_size, num_go, n_classes, combine_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        self.combined_size = combine_size
        self.exp_encoder = nn.Sequential(
            nn.Linear(1343, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_size),
        )
        self.go_encoder = nn.Sequential(
            nn.Linear(num_go, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_size)
        )
        self.g2v_encoder = nn.Sequential(
            nn.Linear(1000, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embed_size)
        )
        
        self.n_class = 1 if n_classes <= 2 else n_classes
        self.out = nn.Sequential(
            nn.Linear(self.combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_class)
        )

        self.flatten = nn.Flatten()
        self.open = lambda x: x
        self.activation = self.open if self.n_class > 2 else nn.Sigmoid()

    def combine_vec(self, vec1, vec2, vec3=None):
        combined = None
        if torch.is_tensor(vec3):
            if self.combined_size == 200:
                combined = torch.add(torch.add(vec1, vec2), vec3)
            elif self.combined_size == 600:
                combined = torch.cat((vec1, vec2, vec3), dim=1)
            else:
                raise ValueError("incorrect concatenation argument")

        else:
            if self.combined_size == 200:
                combined = torch.add(vec1, vec2)
            elif self.combined_size == 400:
                combined = torch.cat((vec1, vec2), dim=1)
            else:
                raise ValueError("incorrect concatenation argument")
        return combined


class GO(LinearModel):
    name = "GO"

    def forward(self, gp, g2v, go, exp):
        go = go.to(self.device).float()
        z_go = self.go_encoder(go)
        flatten = self.flatten(z_go)
        logit = self.out(flatten)
        prob = self.activation(logit)
        return prob


class EXP(LinearModel):
    name = "EXP"
    def forward(self, gp, g2v, go, exp):
        exp = exp.to(self.device).float()
        z_exp = self.exp_encoder(exp)
        flatten = self.flatten(z_exp)
        logit = self.out(flatten)
        prob = self.activation(logit)
        return prob


class G2V(LinearModel):
    name = "G2V"
    def forward(self, gp, g2v, go, exp):
        g2v = g2v.to(self.device).float()
        z_g2v = self.g2v_encoder(g2v)
        flatten = self.flatten(z_g2v)
        logit = self.out(flatten)
        prob = self.activation(logit)
        return prob
    
class G2VMOD(LinearModel):
    name = "G2V"

    def __init__(self, embed_size, num_go, n_classes, combine_size, input_size):
        super().__init__(embed_size=embed_size, num_go=num_go, n_classes=n_classes, combine_size=combine_size)
        self.g2v_encoder = nn.Sequential(
        nn.Linear(input_size, 768),
        nn.ReLU(),
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, embed_size)
        )


    def forward(self, gp, g2v, go, exp):
        g2v = g2v.to(self.device).float()
        z_g2v = self.g2v_encoder(g2v)
        flatten = self.flatten(z_g2v)
        logit = self.out(flatten)
        prob = self.activation(logit)
        return prob


class GOEXP(LinearModel):
    name = "GOEXP"

    def forward(self, gp, g2v, go, exp):
        go = go.to(self.device).float()
        exp = exp.to(self.device).float()

        z_go = self.go_encoder(go)
        z_exp = self.exp_encoder(exp)
        comb = self.combine_vec(z_go, z_exp)
        flattened = self.flatten(comb)
        print(check_dead_neurons(flattened))
        logit = self.out(flattened)
        prob = self.activation(logit)
        return prob


class GOG2V(LinearModel):
    name = "GOG2V"
    def forward(self, gp, g2v, go, exp):
        go = go.to(self.device).float()
        g2v = g2v.to(self.device).float()
        
        z_g2v = self.g2v_encoder(g2v)
        z_go = self.go_encoder(go)
        comb = self.combine_vec(z_go, z_g2v)
        flattened = self.flatten(comb)
        logit = self.out(flattened)
        prob = self.activation(logit)
        return prob


class EXPG2V(LinearModel):
    name = "EXPG2V"

    def forward(self, gp, g2v, go, exp):
        g2v = g2v.to(self.device).float()
        exp = exp.to(self.device).float()

        z_g2v = self.g2v_encoder(g2v)
        z_exp = self.exp_encoder(exp)
        comb = self.combine_vec(z_exp, z_g2v)
        flattened = self.flatten(comb)
        logit = self.out(flattened)
        prob = self.activation(logit)
        return prob


class ComboBreaker(LinearModel):
    name = "COMBI"
    def forward(self, gp, g2v, go, exp):
        go = go.to(self.device).float()
        g2v = g2v.to(self.device).float()
        exp = exp.to(self.device).float()

        z_g2v = self.g2v_encoder(g2v)
        z_g2v = nn.ReLU()(z_g2v)
        z_go = self.go_encoder(go)
        z_go = nn.ReLU()(z_go)
        z_exp = self.exp_encoder(exp)
        z_g2v = nn.ReLU()(z_exp)
        comb = self.combine_vec(z_go, z_exp, z_g2v)
        flattened = self.flatten(comb)
        print(check_dead_neurons(flattened))
        logit = self.out(flattened)
        prob = self.activation(logit)
        return prob

class Transformer(LinearModel):
    name = "TRANSFORMER"
    def __init__(self, embed_size, num_go, n_classes, combine_size, num_layers, num_head):
        super().__init__(embed_size, num_go, n_classes, combine_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=combine_size, nhead=num_head,
            batch_first=True)
        
        self. transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers)
        
    def forward(self, gp, g2v, go, exp):
        go = go.to(self.device).float()
        g2v = g2v.to(self.device).float()
        exp = exp.to(self.device).float()

        z_g2v = self.g2v_encoder(g2v)
        z_go = self.go_encoder(go)
        z_exp = self.exp_encoder(exp)
        comb = self.combine_vec(z_go, z_exp, z_g2v)
        flattened = self.flatten(comb)
        transformed = self.transformer_encoder(flattened)
        logit = self.out(transformed)
        prob = self.activation(logit)
        return prob

   

# class OptimizationModel(nn.Module):
    # def _generate_layer_size(max_val, min_val, num_values):
    #     """Generate `num_values` integers between `min_val` and `max_val`."""
    #     if min_val >= max_val:
    #         raise ValueError("min_val must be less than max_val.")
    #     if num_values < 1:
    #         raise ValueError("num_values must be at least 1.")

    #     # Generate evenly spaced values between min_val and max_val
    #     values = np.linspace(min_val, max_val, num_values, dtype=int)

    #     # Ensure the result is sorted in descending order (if desired)
    #     return sorted(values, reverse=True)
    
    # def combine_vec(self, list_vec):
    #     combined = None

        if self.conc: 
            combined = torch.cat(list_vec, dim=1)
        else:
            combined = torch.sum(torch.stack(list_vec), dim=0)
        return combined
        # if torch.is_tensor(vec3):
        #     if not self.concat:
        #         combined = torch.add(torch.add(vec1, vec2), vec3)
        #     elif self.concat:
        #         combined = torch.cat((vec1, vec2, vec3), dim=1)
        #     else:
        #         raise ValueError("incorrect concatenation argument")

        # else:
        #     if self.combined_size == 200:
        #         combined = torch.add(vec1, vec2)
        #     elif self.combined_size == 400:
        #         combined = torch.cat((vec1, vec2), dim=1)
        #     else:
        #         raise ValueError("incorrect concatenation argument")
        # return combined


    # def __init__(self, embed_size,  n_classes, enc_layers=3, ffout_layers=3, act="ReLU", dropout=0.2, batch_norm=True, concat=True):
    #     super().__init__()
    #     self.conc = concat
    #     self.n_class = 1 if n_classes <= 2 else n_classes
    #     self.flatten = nn.Flatten()
    #     self.encoder_batchnorm = nn.BatchNorm1d(2) if batch_norm else nn.Identity()
    #     self.ffout_batchnorm = nn.LazyBatchNorm1d() if batch_norm else nn.Identity()
    #     self.out_act = nn.Identity() if self.n_class > 2 else nn.Sigmoid()
    #     self.combined_size = 3 * embed_size if concat else embed_size
    #     self.activations = nn.ModuleDict({
    #             "ReLU": nn.ReLU(),
    #             "PReLU": nn.PReLU(),
    #             "LeakyReLU": nn.LeakyReLU(),
    #             "Tanh": nn.Tanh(),
    #             "ELU": nn.ELU(),
    #             "GELU": nn.GELU()
    #             })
        
    #     self.act = self.activations[act]
        
    #     self.exp_encoder = nn.ModuleList([])
    #     self.go_encoder = nn.ModuleList([])
    #     self.g2v_encoder = nn.ModuleList([])
    #     self.ffout = nn.ModuleList([])
   

    #     self.exp_shapes = [1343] + OptimizationModel._generate_layer_size(1280, embed_size, enc_layers)
    #     self.go_shapes = [7247] + OptimizationModel._generate_layer_size(1280, embed_size, enc_layers)
    #     self.g2v_shapes = [1000] + OptimizationModel._generate_layer_size(928, embed_size, enc_layers)
    #     self.ffout_shapes = OptimizationModel._generate_layer_size(self.combined_size  *2, self.n_class, ffout_layers+1)
    
    #     for i, (insize, outsize) in enumerate(zip(self.exp_shapes, self.exp_shapes[1:])):
    #         self.exp_encoder.add_module(f"{i}", 
    #                                     nn.Sequential(
    #                                         nn.Linear(insize, outsize),
    #                                         self.encoder_batchnorm,
    #                                         self.act,
    #                                         nn.Dropout(dropout)
    #                                     )
    #                                 )
            
    #     for i, (insize, outsize) in enumerate(zip(self.go_shapes, self.go_shapes[1:])):
    #         self.go_encoder.add_module(f"{i}", 
    #                                     nn.Sequential(
    #                                         nn.Linear(insize, outsize),
    #                                         self.encoder_batchnorm,
    #                                         self.act,
    #                                         nn.Dropout(dropout)
    #                                     )
    #                                 )
            
    #     for i, (insize, outsize) in enumerate(zip(self.g2v_shapes, self.g2v_shapes[1:])):
    #         self.g2v_encoder.add_module(f"{i}", 
    #                                     nn.Sequential(
    #                                         nn.Linear(insize, outsize),
    #                                         self.encoder_batchnorm,
    #                                         self.act,
    #                                         nn.Dropout(dropout)
    #                                     )
    #                                 )
        
    #     for i, (insize, outsize) in enumerate(zip(self.ffout_shapes, self.ffout_shapes[1:])):
    #         ffout_batchnorm = nn.LazyBatchNorm1d()
    #         if i == ffout_layers -1 or not batch_norm:
    #             dropout = 0.0
    #             ffout_batchnorm = nn.Identity()
                
                
    #         self.ffout.add_module(f"{i}", 
    #                                     nn.Sequential(
    #                                         nn.Linear(insize, outsize),
    #                                         ffout_batchnorm,
    #                                         self.act,
    #                                         nn.Dropout(dropout)
    #                                     )
    #                                 )
            
    #     self.exp_encoder = nn.Sequential(*self.exp_encoder)
    #     self.go_encoder = nn.Sequential(*self.go_encoder)
    #     self.g2v_encoder = nn.Sequential(*self.g2v_encoder)
    #     self.ffout = nn.Sequential(*self.ffout)


class OptimizationModel(nn.Module):
    name = "OPTIM-model"
    def _generate_layer_size(max_val, min_val, num_values):
        """Generate `num_values` integers between `min_val` and `max_val`."""
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val.")
        if num_values < 1:
            raise ValueError("num_values must be at least 1.")

        # Generate evenly spaced values between min_val and max_val
        values = np.linspace(min_val, max_val, num_values, dtype=int)

        # Ensure the result is sorted in descending order (if desired)
        return sorted(values, reverse=True)
    
    def combine_vec(self, list_vec):
        combined = None

        if self.conc: 
            combined = torch.cat(list_vec, dim=1)
        else:
            combined = torch.sum(torch.stack(list_vec), dim=0)
        return combined


    def __init__(self, embed_size,  n_classes, enc_layers=3, ffout_layers=3, act="ReLU", dropout=0.2, batch_norm=True, concat=True):
        super().__init__()
        self.conc = concat
        self.ffout_layers = ffout_layers+1
        self.n_class = 1 if n_classes <= 2 else n_classes
        self.flatten = nn.Flatten()
        self.encoder_batchnorm = nn.BatchNorm1d(2) if batch_norm else nn.Identity()
        self.ffout_batchnorm = nn.LazyBatchNorm1d() if batch_norm else nn.Identity()
        self.out_act = nn.Identity() if self.n_class > 2 else nn.Sigmoid()
        self.combined_size = 3 * embed_size if concat else embed_size
        self.activations = nn.ModuleDict({
                "ReLU": nn.ReLU(),
                "PReLU": nn.PReLU(),
                "LeakyReLU": nn.LeakyReLU(),
                "Tanh": nn.Tanh(),
                "ELU": nn.ELU(),
                "GELU": nn.GELU()
                })
        
        self.act = self.activations[act]
        
        self.exp_encoder = nn.ModuleList([])
        self.go_encoder = nn.ModuleList([])
        self.g2v_encoder = nn.ModuleList([])
        self.ffout = nn.ModuleList([])
   

        self.exp_shapes = [1343] + OptimizationModel._generate_layer_size(1280, embed_size, enc_layers)
        self.go_shapes = [7247] + OptimizationModel._generate_layer_size(1280, embed_size, enc_layers)
        self.g2v_shapes = [1000] + OptimizationModel._generate_layer_size(928, embed_size, enc_layers)
        self.ffout_shapes = OptimizationModel._generate_layer_size(self.combined_size  *2, self.n_class, self.ffout_layers)

        for i, (insize, outsize) in enumerate(zip(self.exp_shapes, self.exp_shapes[1:])):
            self.exp_encoder.append(nn.Linear(insize, outsize)) 
            if batch_norm:           
                self.exp_encoder.append(self.encoder_batchnorm)
            if i+2 != len(self.exp_shapes):               
                self.exp_encoder.append(self.act)   
            if dropout > 0:                        
                self.exp_encoder.append(nn.Dropout(dropout))  
                                    
            
        for i, (insize, outsize) in enumerate(zip(self.go_shapes, self.go_shapes[1:])):
            self.go_encoder.append(nn.Linear(insize, outsize))
            if batch_norm:  
                self.go_encoder.append(self.encoder_batchnorm)
            if i+2 != len(self.go_shapes):               
                self.go_encoder.append(self.act)                
            if dropout > 0:
                self.go_encoder.append(nn.Dropout(dropout))  

            
        for i, (insize, outsize) in enumerate(zip(self.g2v_shapes, self.g2v_shapes[1:])):
            self.g2v_encoder.append(nn.Linear(insize, outsize))
            if batch_norm:
                self.g2v_encoder.append(self.encoder_batchnorm)
            if i+2 != len(self.g2v_shapes):
                self.g2v_encoder.append(self.act)                            
            if dropout > 0:   
                self.g2v_encoder.append(nn.Dropout(dropout))  
                                    
        
        for i, (insize, outsize) in enumerate(zip(self.ffout_shapes, self.ffout_shapes[1:])):
            ffout_batchnorm = nn.LazyBatchNorm1d()
            if i == self.ffout_layers -1 or not batch_norm:
                dropout = 0.0
                ffout_batchnorm = False

            self.ffout.append(nn.Linear(insize, outsize))

            if ffout_batchnorm:
                self.ffout.append(ffout_batchnorm)
            if i+1 != ffout_layers:            
                self.ffout.append(self.act)
            if dropout > 0:   
                self.ffout.append(nn.Dropout(dropout))
            
        self.exp_encoder = nn.Sequential(*self.exp_encoder)
        self.go_encoder = nn.Sequential(*self.go_encoder)
        self.g2v_encoder = nn.Sequential(*self.g2v_encoder)
        self.ffout = nn.Sequential(*self.ffout)

    def forward(self, gp, g2v, go, exp):
        go = go.to("cuda").float()
        g2v = g2v.to("cuda").float()
        exp = exp.to("cuda").float()

        z_g2v = self.g2v_encoder(g2v)
        # print(check_dead_neurons(z_g2v))
        # print(z_g2v.size())
        
        z_go = self.go_encoder(go)
        # print(check_dead_neurons(z_go))
        # print(z_go.size())
        z_exp = self.exp_encoder(exp)
        # print(check_dead_neurons(z_exp))
        # print(z_exp.size())
        comb = self.combine_vec([z_go, z_exp, z_g2v])
        # print(comb.size())
        flattened = self.flatten(comb)
        # print(flattened.size())
        # print(check_dead_neurons(flattened))
        logit = self.ffout(flattened)
        # print(logit.size())
        prob = self.out_act(logit)
        return prob
    
if __name__ == "__main__":

    print("running file as main")
    expression_data = pd.read_csv(
        "EXP/EXP_data/expression.csv", sep="\t", header=0, index_col=0)
    expression_data = np.array(expression_data)
    data_path = "LABELS/LABEL_data/Train_set.csv"
    dataset = _DataSet(data_path, expression_data)
    breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.utils.data.DataLoader(dataset, batch_size=32,
                                             shuffle=True)
    breakpoint()
    model = RegPred(expression_data.shape, 50, 1, 2)
    model.to(device)
    breakpoint()
    batch = next(iter(train_data))
    fp = model(batch[0], batch[1])
    breakpoint()
