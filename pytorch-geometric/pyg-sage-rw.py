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

import gc
import time
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, NeighborSampler
from torch_geometric.utils import to_undirected
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_cluster import random_walk
from sklearn.preprocessing import OneHotEncoder

# read data
data_dir = os.path.join(os.getcwd(), 'data')
df_taglist = pd.read_csv(os.path.join(data_dir, 'taglist_heart.csv'), names=['tag', 'gene'])
enc = OneHotEncoder(sparse=False).fit(df_taglist['gene'].to_numpy().reshape(-1, 1))

result_dir = os.path.join(os.getcwd(), 'results')
df_nodes = pd.read_csv(os.path.join(result_dir, 'nodes.csv'), index_col=0)
df_nodes = pd.DataFrame(data=enc.transform(df_nodes['gene'].to_numpy().reshape(-1, 1)), index=df_nodes.index)
df_edges = pd.read_csv(os.path.join(result_dir, 'edges.csv'), index_col=0)

# create pytorch geometric Data
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
walk_length = 1
num_neg_samples = 1
epochs = 10
disable = True

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None, 
    sizes=[-1], batch_size=batch_size, shuffle=False, 
    num_workers=0)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SAGE, self).__init__()
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.num_node_features, hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x = data.x.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=data.num_nodes, disable=disable)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0

    node_idx = torch.randperm(data.num_nodes)
    train_loader = NeighborSampler(
        data.edge_index, node_idx=node_idx, 
        sizes=num_samples, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    # positive sampling
    rw = random_walk(data.edge_index[0], data.edge_index[1], node_idx, walk_length=walk_length)
    rw_idx = rw[:,1:].flatten()
    pos_loader = NeighborSampler(
        data.edge_index, node_idx=rw_idx, 
        sizes=num_samples, batch_size=batch_size * walk_length, shuffle=False,
        num_workers=0
    )
    
    # negative sampling
    deg = degree(data.edge_index[0])
    distribution = deg ** 0.75
    neg_idx = torch.multinomial(
        distribution, data.num_nodes * num_neg_samples, replacement=True)
    neg_loader = NeighborSampler(
        data.edge_index, node_idx=neg_idx,
        sizes=num_samples, batch_size=batch_size * num_neg_samples,
        shuffle=True, num_workers=0)

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in zip(train_loader, pos_loader, neg_loader):
        
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = model(x[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = model(x[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = model(x[vn_id], adjs_vn)

        optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v) \
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn) \
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size_)

    pbar.close()

    loss = total_loss / len(train_loader)

    return loss


for epoch in range(1, epochs + 1):
    gc.collect()
    loss = train(epoch)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

model_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(model_dir, exist_ok=True)
model_name = os.path.join(model_dir, time.strftime('sage-rw-%Y%m%d.pt'))
torch.save(model, model_name)
model = torch.load(model_name)

gc.collect()
model.eval()
z = model.inference(x)
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
