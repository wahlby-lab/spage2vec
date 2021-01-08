import os
import random
import time
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

sc.logging.print_header()
sc.settings.verbosity = 4

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')
df = {}
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    df[name] = pd.read_csv(os.path.join(data_dir, 'spots_PCW{}.csv'.format(name)))
    df[name]['pcw'] = int(name[0])
    df[name]['section'] = int(name[-1])
    df[name] = df[name].set_index('{}_'.format(name) + df[name].index.astype(str))
df_heart = pd.concat(df.values())

model = 'SAGE'
file_name = '.npy' # embedding file name
node_embeddings = np.load(os.path.join(result_dir, file_name))
print(file_name)
print(node_embeddings.shape)
df_nodes = pd.read_csv(os.path.join(result_dir, 'nodes.csv'), index_col=0)
df_heart = df_heart.loc[df_nodes.index]
adata = sc.AnnData(X=np.copy(node_embeddings), obs=df_heart)

# clustering
sc.pp.neighbors(adata, n_neighbors=15, random_state=42)
adata_name = time.strftime('{}-%Y%m%d.h5ad'.format(model))
adata.write(os.path.join(result_dir, adata_name))
sc.tl.louvain(adata, resolution=1.0, random_state=42)
adata.write(os.path.join(result_dir, adata_name))
print(adata)

# umap
sc.tl.umap(adata, min_dist=0.5, n_components=3, random_state=42)
file_name = time.strftime('{}-umap-%Y%m%d.npy'.format(model))
np.save(os.path.join(result_dir, file_name), adata.obsm['X_umap'])
print(file_name)
