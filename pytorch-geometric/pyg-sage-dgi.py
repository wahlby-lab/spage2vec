import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import time
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, DeepGraphInfomax
from torch_sparse import SparseTensor
from sklearn.preprocessing import OneHotEncoder

data_dir = os.path.join(os.getcwd(), 'data')
df_taglist = pd.read_csv(os.path.join(data_dir, 'taglist_heart.csv'), names=['tag', 'gene'])
enc = OneHotEncoder(sparse=False).fit(df_taglist['gene'].to_numpy().reshape(-1, 1))

result_dir = os.path.join(os.getcwd(), 'results')
df_nodes = pd.read_csv(os.path.join(result_dir, 'nodes.csv'), index_col=0)
df_nodes = pd.DataFrame(data=enc.transform(df_nodes['gene'].to_numpy().reshape(-1, 1)), index=df_nodes.index)
df_edges = pd.read_csv(os.path.join(result_dir, 'edges.csv'), index_col=0)

index_dict = dict(zip(df_nodes.index, range(len(df_nodes))))
df_edges_index = df_edges[['source', 'target']].applymap(index_dict.get)
x = torch.tensor(df_nodes.to_numpy(), dtype=torch.float)
edge_index = torch.tensor(df_edges_index.to_numpy(), dtype=torch.long)
edge_index = to_undirected(edge_index.t().contiguous())
data = Data(x=x, edge_index=edge_index)
print(data)
print(data.num_edges / data.num_nodes)

# hyperparameters
num_samples = [-1, -1] # number of samples in each layer
batch_size = 64
hidden_channels = 32
epochs = 10
disable = True

train_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=num_samples, batch_size=batch_size, shuffle=True,
    num_workers=0
)
subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None, 
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0
)

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.num_layers = 2
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, disable=disable)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

def summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))

def corruption(x, *args):
    return (x[torch.randperm(x.size(0))], *args)

def train(epoch):
    model.train()
    pbar = tqdm(total=data.x.shape[0], disable=disable)
    pbar.set_description('Epoch {:03d}'.format(epoch))
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x[n_id], adjs)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update(batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = DeepGraphInfomax(
    hidden_channels=hidden_channels,
    encoder=SAGE(data.num_features, hidden_channels),
    summary=summary, corruption=corruption).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    loss = train(epoch)
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))

model_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(model_dir, exist_ok=True)
model_name = os.path.join(model_dir, time.strftime('sage-dgi-%Y%m%d.pt'))
torch.save(model, model_name)
model = torch.load(model_name)

model.eval()
z = model.encoder.inference(data.x)

node_embeddings = z.detach().cpu().numpy()
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
embedding_name = time.strftime('{}-embedding-%Y%m%d.npy'.format(type(model.encoder).__name__))
np.save(os.path.join(result_dir, embedding_name), node_embeddings)
print(embedding_name)


# validate against pciseq
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df_celltype = pd.read_csv(os.path.join(result_dir, 'celltype.csv'), index_col=0)
df_nodes = pd.DataFrame(data=node_embeddings, index=df_nodes.index)
df_nodes = pd.concat([df_nodes, df_celltype], axis=1).reindex(df_nodes.index)
X = df_nodes.loc[df_nodes['cell_type_id'] >= 0].to_numpy()
print(X.shape)
X, y = X[:,:-1], X[:,-1]
clf = LogisticRegression(verbose=False, n_jobs=-1)
# clf = LinearSVC()
print(cross_val_score(clf, X, y, scoring='f1_weighted', cv=5))

