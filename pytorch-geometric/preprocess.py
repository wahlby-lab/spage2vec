import os
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from sklearn.preprocessing import OneHotEncoder

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(result_dir, exist_ok=True)
fig_dir = os.path.join(os.getcwd(), 'figures')
os.makedirs(fig_dir, exist_ok=True)

# read data
df = {}
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    df[name] = pd.read_csv(os.path.join(data_dir, 'spots_PCW{}.csv'.format(name)))
    print(df[name].shape)
    df[name]['pcw'] = int(name[0])
    df[name]['section'] = int(name[-1])
    df[name] = df[name].set_index('{}_'.format(name) + df[name].index.astype(str))
df_heart = pd.concat(df.values())

# process pciseq cell types
df_cell = {}
for name in ['6.5_1', '6.5_2']:
    df_cell_segmentation = pd.read_csv(os.path.join(data_dir, 'spots_w_cell_segmentation_PCW{}.csv'.format(name)))
    df_cell_calling = pd.read_csv(os.path.join(data_dir, 'cell_calling_PCW{}.csv'.format(name)))
    df_cell[name] = pd.merge(
        df_cell_segmentation, df_cell_calling[['cell', 'celltype']], 
        how='left', left_on='parent_id', right_on='cell'
    )
    df_cell[name]['pcw'] = int(name[0])
    df_cell[name]['section'] = int(name[-1])
    df_cell[name] = df_cell[name].set_index('{}_'.format(name) + df_cell[name].index.astype(str))
df_heart_cell = pd.concat(df_cell.values())
cell_types = df_heart_cell['celltype'].dropna().unique()
cell_type_id = [(cell_type, int(re.search(r'\((\d+)\)', cell_type).group(1))) for cell_type in cell_types]
cell_type_id.append(('Uncalled', -1))
cell_type_id.sort(key=lambda x: x[1])
cell_type_id = dict(cell_type_id)
df_heart_cell['celltype'].fillna(value='Uncalled', inplace=True)
df_heart_cell['cell_type_id'] = df_heart_cell['celltype'].map(cell_type_id)
df_heart_cell['cell_type_id'].to_csv(os.path.join(result_dir, 'celltype.csv'))

# build graph
df_edge = {}
pct = 99
distances_all = []
print(pct)
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    spots = df[name][['spotX', 'spotY']]
    kdtree = KDTree(spots)
    distances, _ =  kdtree.query(spots, k=2)
    distances_all.append(distances)
distances_all = np.concatenate(distances_all, axis=0)
d_max = np.percentile(distances_all[:,1], pct)
print(d_max)
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    spots = df[name][['spotX', 'spotY']]
    kdtree = KDTree(spots)
    ind = kdtree.query_radius(spots, d_max)
    df_edge[name] = pd.DataFrame(
        data=[(spots.index[i], spots.index[j]) for i in range(len(spots)) for j in ind[i] if i < j], 
        columns=['source', 'target']
    )
    df_edge[name] = df_edge[name].set_index('{}_'.format(name) + df_edge[name].index.astype(str))
df_edges = pd.concat(df_edge.values())
df_edges = df_edges.reset_index()

# remove components
g = nx.from_pandas_edgelist(df_edges, edge_attr='index')
n_cc = 6
print(n_cc)
for cc in nx.connected_components(g.copy()):
    if len(cc) < n_cc:
        g.remove_nodes_from(cc)
print(g.number_of_nodes())
print(g.number_of_edges() / g.number_of_nodes() * 2)

# save nodes and edges
df_nodes = df_heart.loc[list(g.nodes), 'gene']
df_nodes.to_csv(os.path.join(result_dir, 'nodes.csv'))
df_edges = nx.to_pandas_edgelist(g)
df_edges = df_edges.set_index('index')
df_edges.to_csv(os.path.join(result_dir, 'edges.csv'))

df_heart = df_heart.loc[df_nodes.index]
print(df_heart['pcw'].value_counts())
df_heart_cell = df_heart_cell.loc[df_heart_cell.index.intersection(df_nodes.index)]
print(df_heart_cell['celltype'].value_counts())
