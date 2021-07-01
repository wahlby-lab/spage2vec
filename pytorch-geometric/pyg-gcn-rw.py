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
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from torch_cluster import random_walk
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
data.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes)).t()
print(data)
print(data.num_edges / data.num_nodes)

hidden_channels = 32
walk_length = 1
num_neg_samples = 1
epochs = 1000


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.num_layers = 2
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(data.num_features, hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
deg = degree(data.edge_index[0])
distribution = deg ** 0.75

def train():
    model.train()
    optimizer.zero_grad()
    z_u = model(data.x, data.adj_t)
    node_idx = torch.arange(data.num_nodes).to(device)
    rw = random_walk(data.edge_index[0], data.edge_index[1], node_idx, walk_length=walk_length)
    rw_idx = rw[:,1:].flatten()
    z_v = z_u[rw_idx]
    neg_idx = torch.multinomial(distribution, data.num_nodes * num_neg_samples, replacement=True)
    z_vn = z_u[neg_idx]
    pos_loss = -F.logsigmoid(
        (z_u.repeat_interleave(walk_length, dim=0)*z_v) \
        .sum(dim=1)).mean()
    neg_loss = -F.logsigmoid(
        -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn) \
        .sum(dim=1)).mean()
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()

model_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(model_dir, exist_ok=True)
model_name = os.path.join(model_dir, time.strftime('gcn-rw-%Y%m%d.pt'))
best_loss = float('inf')
for epoch in range(1, epochs + 1):
    loss = train()
    if loss < best_loss:
        best_loss = loss
        torch.save(model, model_name)
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

model = torch.load(model_name)
model.eval()
z = model(data.x, data.adj_t)
node_embeddings = z.detach().cpu().numpy()
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
embedding_name = time.strftime('{}-embedding-%Y%m%d.npy'.format(type(model).__name__))
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
