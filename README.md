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
$ git clone https://github.com/mdellaiera/mangroves.git
```

### Install the project

```bash
$ cd mangroves
$ mamba env create -f environment.yml
$ conda activate mangroves
$ pip install -e .
```

To ensure the project is correctly installed, run the following command.
```bash
$ conda activate mangroves
$ run
```