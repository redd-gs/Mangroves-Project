import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Dict, List, Optional
from pathlib import Path
import logging
import os
import re
import glob


# Coverage category label mapping
CATEGORY_BINS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CATEGORY_NAMES = ['0%', '1-20%', '21-40%', '41-60%', '61-80%', '81-100%']


def _parse_npz_filename(filename: str) -> Dict:
    """Extract category and coverage from filename like cat1_id100.0_cov0.07.npz."""
    match = re.match(r'cat(\d+)_id[\d.]+_cov([\d.]+)\.npz', filename)
    if match:
        cat = int(match.group(1))
        cov = float(match.group(2))
        return {'category': cat, 'coverage': cov}
    return None


class MangroveDataset(Dataset):
    """
    PyTorch Dataset loading AlphaEarth embedding .npz files.

    Each .npz file contains a (64, 244, 244) embedding tensor.
    The coverage category (label) is parsed from the filename.

    Supports two modes:
    - npz_dir mode: scans a directory of .npz files (output/dataset/)
    - csv mode (legacy): reads a data.csv with file paths
    """

    def __init__(self,
                 path: str,
                 train: bool = True,
                 max_samples: int = -1,
                 categories: Optional[List[int]] = None):
        """
        Args:
            path: Path to directory containing .npz files, or directory with data.csv.
            train: Ignored when loading .npz files (split handled by DataModule).
            max_samples: Maximum number of samples to load (-1 = all).
            categories: List of category ints to include (e.g. [1,2,3,4,5]).
                        None means all categories.
        """
        self.path = Path(path)
        self.samples: List[Dict] = []

        # .npz directory mode 
        npz_files = sorted(glob.glob(str(self.path / '*.npz')))
        if npz_files:
            for fpath in npz_files:
                meta = _parse_npz_filename(Path(fpath).name)
                if meta is None:
                    continue
                if categories and meta['category'] not in categories:
                    continue
                label = meta['category'] - 1  # 0-indexed class label
                self.samples.append({'path': fpath, 'label': label,
                                     'coverage': meta['coverage']})
        else:
            # ---- Legacy CSV mode ----
            csv_path = self.path / 'data.csv'
            if csv_path.exists():
                data = pd.read_csv(csv_path)
                data = data[data['train'] == int(train)]
                for _, row in data.iterrows():
                    self.samples.append({
                        'path': str(self.path / row['embeddings']),
                        'label': int(row['category']) - 1,  # 0-indexed class label
                        'coverage': row.get('ratio', 0.0)
                    })

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        logging.info(f'MangroveDataset: loaded {len(self.samples)} samples from {self.path}')

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        fpath = sample['path']

        if fpath.endswith('.npz'):
            data = np.load(fpath)
            # .npz files store embeddings under key 'embeddings' or first key
            if 'embeddings' in data:
                embeddings = data['embeddings'].astype(np.float32)
            else:
                embeddings = data[list(data.keys())[0]].astype(np.float32)
        elif fpath.endswith('.tif'):
            import rasterio
            with rasterio.open(fpath, 'r') as f:
                embeddings = f.read().astype(np.float32)
        else:
            raise ValueError(f'Unsupported file format: {fpath}')

        embeddings = torch.from_numpy(embeddings)
        label = torch.tensor(sample['label'], dtype=torch.long)

        return {'embeddings': embeddings, 'label': label}

    def __len__(self) -> int:
        return len(self.samples)


class MangroveDataModule(LightningDataModule):
    """
    This class is used to create a DataModule for a dataset.
    """
    def __init__(self,
                 dataset: MangroveDataset,
                 batch_size: int = 32,
                 num_processes: int = 1,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 pin_memory: bool = False,
                 shuffle: bool = False):
        """
        Args:
            dataset(MangroveDataset): The dataset to be used.
            batch_size (int, optional): Batch size for the dataloaders.
            num_processes (int, optional): Number of processes to use for loading the dataset.
            val_split (float, optional): Validation split ratio.
            test_split (float, optional): Test split ratio.
            pin_memory (bool, optional): Pin memory for faster GPU transfer.
            shuffle (bool, optional): Shuffle the dataset.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.val_split = val_split
        self.test_split = test_split
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - test_size - val_size
        assert train_size + val_size + test_size == len(dataset), 'Split sizes do not add up to dataset size'
        randperm = torch.randperm(len(dataset))
        self.train_index = randperm[:train_size]
        self.val_index = randperm[train_size:train_size + val_size]
        self.test_index = randperm[train_size + val_size:]
        self.train_dataset = Subset(dataset, self.train_index)
        self.val_dataset = Subset(dataset, self.val_index)
        self.test_dataset = Subset(dataset, self.test_index)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_processes, 
                          pin_memory=self.pin_memory, 
                          shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_processes, 
                          pin_memory=self.pin_memory, 
                          shuffle=self.shuffle)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_processes, 
                          pin_memory=self.pin_memory, 
                          shuffle=self.shuffle)
