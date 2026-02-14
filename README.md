# Mangrove Project

### Mamba Installatation (Recommended)

Install mamba following the installation procedure at https://github.com/conda-forge/miniforge.

```bash
$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
$ ~/miniforge3/bin/conda init
```

### Conda Installation

```bash
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init
```

### Download the project from GitHub

```bash
$ git clone https://github.com/redd-gs/Mangroves-Project.git
```

### Install the project

```bash
$ cd Mangroves-Project
$ mamba env create -f environment.yml
$ conda activate mangroves
$ pip install -e .
```

To ensure the project is correctly installed, run the following command.
```bash
$ conda activate mangroves
$ run
```


### Explication

This project leverages advanced satellite imagery embeddings from Google's AlphaEarth Foundation Model to improve the accuracy of mangrove forest mapping. Mangroves are critical coastal ecosystems that provide numerous environmental services, including carbon sequestration, coastal protection, and biodiversity support. The Global Mangrove Watch (GMW) project provides valuable baseline data on mangrove distribution worldwide, but these labels can benefit from refinement using state-of-the-art deep learning techniques and high-resolution satellite embeddings.

The framework extracts 64-band AlphaEarth embeddings from Google Earth Engine for specified geographic regions, processes them alongside GMW labels, and trains deep learning models using PyTorch Lightning to predict and refine mangrove coverage. The system handles the complete pipeline from data collection (embeddings extraction, label generation) to model training and evaluation. By combining the rich semantic information captured in AlphaEarth embeddings with supervised learning, this approach aims to produce more accurate and detailed mangrove classification maps that can support conservation efforts and environmental monitoring. 