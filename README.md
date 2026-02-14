# Mangrove Project

## Overview

This project leverages satellite imagery embeddings from Google's AlphaEarth Foundation Model to improve the accuracy of mangrove forest mapping. It combines 64-band AlphaEarth embeddings from Google Earth Engine with Global Mangrove Watch (GMW) labels and trains deep learning models using PyTorch Lightning to predict and refine mangrove coverage.

## Pipeline

The workflow follows three main steps, each documented in numbered notebooks:

1. **Data Acquisition** — Authenticate with Google Earth Engine, download Sentinel-2 imagery and AlphaEarth embeddings for regions of interest.
2. **Data Processing** — Create input coordinate files, compute mangrove coverage labels from GMW shapefiles, and prepare training datasets.
3. **Analysis & Training** — Analyze embedding quality (class separation, spatial continuity), then train and evaluate classification models via PyTorch Lightning.

## Project Structure

```
Mangroves-Project/
│
├── mangroves/                          # Core Python package
│   ├── collection.py                   #   Google Earth Engine API wrapper
│   ├── constants.py                    #   Project-wide constants (resolution, bands, etc.)
│   ├── embeddings.py                   #   Download & store embedding patches (.npz)
│   ├── geometry.py                     #   Geospatial region computation (bounding boxes)
│   ├── utils.py                        #   Math utilities (haversine, geodesic circles)
│   └── training/                       #   ML training pipeline (PyTorch Lightning)
│       ├── data.py                     #     Dataset & DataModule definitions
│       ├── load.py                     #     YAML config loaders
│       ├── transforms.py              #     Custom parametric transforms
│       └── main.py                     #     CLI entry point for training/testing
│
├── notebooks/                          # Step-by-step analysis notebooks
│   ├── 1_data_acquisition/             #   Step 1 — Download & explore raw data
│   │   ├── 01_explore_gmw_data         #     Explore Global Mangrove Watch shapefiles
│   │   ├── 02_download_sentinel2       #     Download Sentinel-2 imagery from GEE
│   │   ├── 03_download_embeddings      #     Download AlphaEarth embedding patches
│   │   └── 04_batch_download_embeddings#     Batch download for all sample locations
│   │
│   ├── 2_data_processing/              #   Step 2 — Prepare inputs & labels
│   │   ├── 01_create_inputs            #     Generate sample coordinate CSV
│   │   ├── 02_create_labels            #     Compute mangrove coverage labels from GMW
│   │   └── 03_create_toy_dataset       #     Generate synthetic test dataset
│   │
│   ├── 3_analysis/                     #   Step 3 — Analyze embeddings & evaluate
│   │   ├── 01_embedding_class_separation#    Embedding distances between land-cover classes
│   │   ├── 02_cosine_similarity_map    #     Cosine similarity visualization (Pulau Ubin)
│   │   ├── 03_spatial_continuity       #     Spatial continuity of embeddings
│   │   └── 04_test_training_pipeline   #     Smoke-test the training data pipeline
│   │
│   └── 4_utilities/                    #   Supporting validation notebooks
│       └── 01_validate_geometry        #     Validate haversine & circle utilities
│
├── config/                             # YAML configuration files for training
├── data/                               # Data files (not tracked by git)
├── output/                             # Generated outputs (not tracked by git)
│
├── Writing_Sample/                     # Research paper
│   └── Writing_Sample.pdf
│
├── environment.yml                     # Conda environment specification
├── setup.py                            # Package installation
└── README.md
```

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

