<!--[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-10.1101/765842-red.svg)](https://doi.org/10.1101/765842)
[![DOI](https://zenodo.org/badge/199853991.svg)](https://zenodo.org/badge/latestdoi/199853991)-->

# Spage2vec: Unsupervised detection of spatial gene expression constellations


This repository contains a collection of python notebooks for reproducing analyses and results from the original publication [1]. The **`notebooks`** folder contains code for:
  - Generate spatial gene expression network from in situ transcriptomic data and train an unsupervised graph representation model for producing a node embedding (`spage2vec_*.ipynb`)
  - Visualize and cluster the learned representations in subcelluar funcional domain (`*_embedding.ipynb`)

### System requirement
The sorce code presented in this repository has been developed and tested on a Linux machine running Ubuntu 16.04 operating system with 64GB RAM, Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz cpu, and nvidia TITAN X gpu.

### Python Library Requirements
The following python packages are required for running the notebooks:
  - `numpy==1.17.2`
  - `tensorflow==1.12.0`
  - `tensorboard==1.12.2`
  - `networkx==2.4`
  - `pandas==0.25.2`
  - `matplotlib==3.0.3`
  - `stellargraph==0.8.1`
  - `scipy==1.3.1`
  - `scikit-learn>=0.21.3`
  - `tqdm==4.36.1`
  - `umap-learn==0.3.10`
  - `scanpy==1.4.4`
  - `leidenalg==0.7.0`
  - `seaborn==0.9.0`
  - `h5py==2.10.0`
  - `loompy==3.0.6`

### Data Download
Spatial gene expression data for the three analyzed assays can be downloaded at: https://doi.org/10.5281/zenodo.3664723. Please extract the content of the zipped archive in this repository local folder before running the notebooks.

## Citation
[1] Partel, G., and Wählby C. Spage2vec: Unsupervised detection of spatial gene expression constellations. BioRxiv, , (2019).
