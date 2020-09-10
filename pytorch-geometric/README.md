Install pytorch and pytorch-geometric
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip install torch-geometric
```
Install scanpy
```
pip install scanpy[louvain,leiden]
```
- `preprocess.py`: construct graph
- `pyg-*.py`: train graph neural network
- `cluster.py`: perform UMAP and clustering
- `hm.ipynb`: process the clusters and make plots