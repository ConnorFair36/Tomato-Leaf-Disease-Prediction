import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook contains my experiments with creating the dataloader.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    import torch
    from torch import nn, utils
    from torchvision import transforms, datasets
    from sklearn.model_selection import train_test_split

    import random

    return datasets, mo, np, plt, train_test_split, transforms, utils


@app.cell
def _(transforms):
    data_transform = transforms.Compose([
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    return (data_transform,)


@app.cell
def _(datasets, train_test_split):
    # this class assumes that the dataset is 10,000 images and there are 1,000 images per class and everything is in-order
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

    return (TrainValSplitLoader,)


@app.cell
def _(TrainValSplitLoader, data_transform, utils):
    test_dataset = TrainValSplitLoader(set_type="train", root="./data/train", transform=data_transform)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)
    for d, f in test_dataloader:
        print("yes")
    return


@app.cell
def _(data_transform, datasets):
    train_dir = "./data/train"
    test_dir = "./data/val"

    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=data_transform,
                                     target_transform=None) # transforms to perform on labels

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    return (train_data,)


@app.cell
def _(np, train_data, utils):
    import time
    dataloader = utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    record = [time.monotonic()]
    for a, b in dataloader:
        record.append(time.monotonic())

    record = np.array(record)
    record[1:] - record[:-1]
    return (dataloader,)


@app.cell
def _(dataloader, plt):
    # displays a random image from the dataset
    images, labels = next(iter(dataloader))
    print(labels[0])
    plt.imshow(images[0].permute((1,2,0)))
    return


if __name__ == "__main__":
    app.run()
