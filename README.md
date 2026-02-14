# Mangrove Project

## Overview

This project leverages satellite imagery embeddings from Google's AlphaEarth Foundation Model to improve the accuracy of mangrove forest mapping. It combines 64-band AlphaEarth embeddings from Google Earth Engine with Global Mangrove Watch (GMW) labels and trains deep learning models using PyTorch Lightning to predict and refine mangrove coverage.

## Pipeline

The workflow follows three main steps, each documented in numbered notebooks:

1. **Data Acquisition** — Authenticate with Google Earth Engine, download Sentinel-2 imagery and AlphaEarth embeddings for regions of interest.
2. **Data Processing** — Create input coordinate files, compute mangrove coverage labels from GMW shapefiles, and prepare training datasets.
3. **Analysis & Training** — Analyze embedding quality (class separation, spatial continuity), then train and evaluate classification models via PyTorch Lightning.

## Project Structure

- `mangroves/` : Core Python package (GEE wrappers, geospatial utilities, embedding extraction)
- `mangroves/training/` : (PyTorch Lightning pipeline — dataset, data module, config loaders, CLI)
- `notebooks/1_data_acquisition/` : Download & explore GMW shapefiles, Sentinel-2, and AlphaEarth embeddings 
- `notebooks/2_data_processing/` : Create input coordinates, compute coverage labels, build toy dataset |
- `notebooks/3_analysis/` : Analyze embeddings (class separation, cosine similarity, spatial continuity) 
- `notebooks/4_utilities/` : Validate geometry & math utilities 
- `config/` : YAML configuration files for training 
- `output/` : Generated outputs 
- `Writing_Sample/` : Research paper (PDF) 

## Installation

### 1. Install Mamba (Recommended)

```bash
$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
$ ~/miniforge3/bin/conda init
```

### 2. Clone & Install

```bash
$ git clone https://github.com/redd-gs/Mangroves-Project.git
$ cd Mangroves-Project
$ mamba env create -f environment.yml
$ conda activate mangroves
$ pip install -e .
```

