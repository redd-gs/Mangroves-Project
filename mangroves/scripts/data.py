import torch
from torch.utils.data import Dataset, Subset, DataLoader
from pytorch_lightning import LightningDataModule
import rasterio
from typing import Dict
from pathlib import Path
import logging
import os
import pandas as pd


class MangroveDataset(Dataset):

    def __init__(self, 
                 path: Path, 
                 train: bool = True,
                 max_samples: int = -1):
        """
        Args:
            path (Path): Path to the directory containing the datasets.
            train (bool, optional): Whether to load the training set (True) or the test set (False).
            max_samples (int, optional): Maximum number of samples to load.
        """
        self.path = Path(path)

        data = pd.read_csv(self.path / 'data.csv')
        data = data[data['train'] == train]
        self.data = data.iloc[:max_samples] if max_samples > -1 else data
        logging.debug('Number of files: {}'.format(len(self.data)))

    def __getitem__(self, index: int) -> Dict:
        # sample = self.data.iloc[index]
        # with rasterio.open(sample['embeddings'], 'r') as f:
        #     embeddings = f.read()
        # labels = {'ratio': sample['ratio']}

        # return {'embeddings': embeddings, 'label': labels}
        return index

    def __len__(self) -> int:
        return len(self.data)


class MangroveDataModule(LightningDataModule):
    """
    Based on PyTorch Lightning DataModule, this class is used to create a DataModule for a dataset.
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
