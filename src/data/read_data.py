import numpy as np
import pandas as pd

import torch
from torch import nn, utils
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # read_data.py -> data/ -> src/ -> root
TRAIN_DIR = _PROJECT_ROOT / "data" / "train"
TEST_DIR = _PROJECT_ROOT / "data" / "val"

#TRAIN_DIR = "./data/train"
#TEST_DIR = "./data/val"

DATA_TRANSFORM = transforms.Compose([
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

class TrainValSplitLoader(datasets.ImageFolder):
    def __init__(self, set_type: str, train_val_split=(0.9, 0.1), seed=1336, n_classes=10, **kwargs):
        super().__init__(**kwargs)
        # yell at the user for being wrong
        if set_type not in ["train", "validation"]:
            raise ValueError("set_type must be one of the following: [train, validation]")
        length = 10_000
        all_indexes = list(range(length))
        class_distribution = [i % length//n_classes for i in range(length)]
        train_size = train_val_split[0]/(train_val_split[0] + train_val_split[1])
        train_idx, validation_idx = train_test_split(all_indexes, train_size=train_size, stratify=class_distribution, random_state=seed)
        if set_type == "train":
            self.indexes = train_idx
        else:
            self.indexes = validation_idx

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        # convert the index from an index of indexes to an index the dataloader can use to get an image
        index = self.indexes[index]
        return super().__getitem__(index)

def get_dataloaders(transformation, seed: int=1336, batch_size: int=128, n_workers: int=0):
    """Returns dataloader for train, validation and test data."""
    np.random.seed(seed)
    training_dataset = TrainValSplitLoader(set_type="train", seed=seed,root=TRAIN_DIR, transform=transformation)
    validation_dataset = TrainValSplitLoader(set_type="validation", seed=seed, root=TRAIN_DIR, transform=transformation)

    test_dataset = datasets.ImageFolder(root=TEST_DIR, 
                                 transform=transformation,
                                 target_transform=None)
    
    training_dataloader = utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    validation_dataloader = utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    
    return training_dataloader, validation_dataloader, test_dataloader

