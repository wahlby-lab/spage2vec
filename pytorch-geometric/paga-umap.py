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
sc.set_figure_params(dpi_save=1200)

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')

adata_name = 'spage2vec_27.h5ad'
adata = sc.read(os.path.join(result_dir, adata_name))
print(adata)

adata.obs['louvain'] = adata.obs['louvain'].astype('category')
adata_s = adata[adata.obs['louvain'].astype('int') < 27]

sc.tl.paga(adata_s, groups='louvain')
sc.pl.paga(
    adata_s, 
    threshold=0.2, 
    fontsize=5, 
    node_size_scale=0.5, 
    node_size_power=0.5, 
    edge_width_scale=0.5, 
    save='.svg')
print(adata_s)

sc.tl.umap(adata_s, min_dist=0.5, init_pos='paga', random_state=42)
sc.pl.umap(
    adata_s, 
    color='louvain', 
    legend_loc='on data', 
    legend_fontsize='xx-small', 
    title='', 
    save='.svg')
print(adata_s)

