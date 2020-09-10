import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')
fig_dir = os.path.join(os.getcwd(), 'figures')
os.makedirs(fig_dir, exist_ok=True)

file_name = '.npy' # umap embedding
model = 'SAGE'
X_embedding = np.load(os.path.join(result_dir, file_name))
print(X_embedding.shape)
embedding_rgb = (X_embedding-np.min(X_embedding, axis=0)) / np.ptp(X_embedding, axis=0)

df = {}
for name in ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '9.5_1', '9.5_2', '9.5_3']:
    df[name] = pd.read_csv(os.path.join(data_dir, 'spots_PCW{}.csv'.format(name)))
    df[name]['pcw'] = int(name[0])
    df[name]['section'] = int(name[-1])
    df[name] = df[name].set_index('{}_'.format(name) + df[name].index.astype(str))
df_heart = pd.concat(df.values())
df_nodes = pd.read_csv(os.path.join(result_dir, 'nodes.csv'), index_col=0)
df_heart = df_heart.loc[df_nodes.index]

def umap_plot(pcw):
    n_section = 2 if pcw == 6 else 3
    fig, axes = plt.subplots(n_section, 3, figsize=(21, 7 * n_section))
    for s in range(n_section):
        for i in range(3):
            subset = (df_heart['pcw'] == pcw) & (df_heart['section'] == s+1)
            axes[s,i].scatter(
                embedding_rgb[subset,i], 
                embedding_rgb[subset,(i+1)%3], 
                s=1, 
                c=embedding_rgb[subset], 
                marker='.', 
                alpha=1.0, 
                linewidths=0
            )
            axes[s,i].set_xlabel('UMAP' + str(i+1))
            axes[s,i].set_ylabel('UMAP' + str((i+1)%3+1))
            axes[s,i].set_xticks([])
            axes[s,i].set_yticks([])
            axes[s,i].set_title('{}.5_{}'.format(pcw, s+1))
    fig.suptitle('Umap RGB ({})'.format(model), y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, '{}-umap-{}.png'.format(model, pcw)))
    plt.close()

    fig, axes = plt.subplots(1, n_section, figsize=(7 * n_section, 7))
    for s in range(n_section):
        subset = (df_heart['pcw'] == pcw) & (df_heart['section'] == s+1)
        spot_x = df_heart.loc[subset,'spotX']
        spot_y = df_heart.loc[subset,'spotY']
        if pcw == 6:
            spot_x = spot_x.max() - spot_x
        spot_y = spot_y.max() - spot_y
        axes[s].scatter(
            spot_x, 
            spot_y, 
            s=0.5, 
            c=embedding_rgb[subset], 
            marker='.', 
            alpha=0.7, 
            linewidths=0
        )
        axes[s].set_title('{}.5_{}'.format(pcw, s+1))
    fig.suptitle('Spatial ({})'.format(model), y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'spatial-{}-umap-{}.png'.format(model, pcw)), dpi=500)
    plt.close()

for pcw in [4, 6, 9]:
    umap_plot(pcw)
