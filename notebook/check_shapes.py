import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    # Walk up one level from notebooks/ to reach the project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import torch.nn.functional as F

    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn import metrics
    from src.data.read_data import get_dataloaders

    return ConfusionMatrixDisplay, get_dataloaders, metrics, mo, plt, torch


@app.cell
def _(get_dataloaders):
    train_dataset, val_dataset, test_dataset = get_dataloaders()
    return (train_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Dataset Shape Layout

    Input:
    - (batch size, RGB, Width, Height)

    Output:
    - (batch size)
    - Each is an integer from 0 to 9
    - The index to label mapping:
     - Bacterial_spot: 0
     - Early_blight: 1
     - Late_blight: 2
     - Leaf_Mold: 3
     - Septoria_leaf_spot: 4
     - Spider_mitesTwo-spotted_spider_mite": 5
     - Target_Spot: 6
     - Tomato_Yellow_Leaf_Curl_Virus: 7
     - Tomato_mosaic_virus: 8
     - healthy: 9
    """)
    return


@app.cell
def _(train_dataset):
    test = iter(train_dataset)
    images, label = next(test)
    return images, label


@app.cell
def _(train_dataset):
    list(train_dataset.dataset.class_to_idx.keys())[0]
    return


@app.cell
def _(images, label, plt, train_dataset):
    idx = 127
    plt.imshow(images[idx,:,:,:].permute((1,2,0)))
    plt.title(list(train_dataset.dataset.class_to_idx.keys())[label[idx]][9:])
    plt.show()
    return


@app.cell
def _(torch):
    # use argmax to bring the one-hot encoded output to indexes
    ex_output = torch.randint(10, (128,)).to(dtype=torch.float)
    return (ex_output,)


@app.cell
def _(ConfusionMatrixDisplay, ex_output, label, metrics, plt, torch):
    ConfusionMatrixDisplay(metrics.confusion_matrix(ex_output, label.to(dtype=torch.float))).plot()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
