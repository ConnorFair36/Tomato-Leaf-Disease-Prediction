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
    from torch import nn

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, matthews_corrcoef, accuracy_score, precision_score, recall_score, classification_report
    from sklearn import metrics
    from src.data.read_data import get_dataloaders, DATA_TRANSFORM

    return (
        ConfusionMatrixDisplay,
        DATA_TRANSFORM,
        F,
        accuracy_score,
        confusion_matrix,
        get_dataloaders,
        matthews_corrcoef,
        mo,
        nn,
        np,
        plt,
        precision_score,
        recall_score,
        torch,
    )


@app.cell
def _(DATA_TRANSFORM, get_dataloaders):
    train_dataset, val_dataset, test_dataset = get_dataloaders(DATA_TRANSFORM)
    return test_dataset, train_dataset


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
    test2 = iter(train_dataset)
    images, label = next(test2)
    return images, label


@app.cell
def _(train_dataset):
    min(train_dataset.dataset.indexes)
    return


@app.cell
def _(test_dataset):
    class_to_idx = test_dataset.dataset.class_to_idx
    {key[9:]: value for key, value in class_to_idx.items()}
    return


@app.cell
def _(images, label, plt, train_dataset):
    idx = 127
    plt.imshow(images[idx,:,:,:].permute((1,2,0)))
    plt.title(list(train_dataset.dataset.class_to_idx.keys())[label[idx]][9:])
    plt.show()
    return


@app.cell
def _():
    254*254*20
    return


@app.cell
def _(nn):
    # hot garbage
    idiot_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(254*254*20, 10)
    )
    return (idiot_model,)


@app.cell
def _(idiot_model, images, torch):
    with torch.no_grad():
        test_output = idiot_model(images)
    return (test_output,)


@app.cell
def _(F, test_output, torch):
    pred = torch.argmax(F.log_softmax(test_output,dim=1), dim=1)
    return (pred,)


@app.cell
def _(label):
    label.shape
    return


@app.cell
def _(confusion_matrix, label, pred):
    matrix = confusion_matrix(label, pred)
    return (matrix,)


@app.cell
def _(ConfusionMatrixDisplay, matrix, plt):
    ConfusionMatrixDisplay(matrix).plot()
    plt.show()
    return


@app.cell
def _(label, matthews_corrcoef, pred):
    matthews_corrcoef(label, pred)
    return


@app.cell
def _(accuracy_score, label, pred):
    accuracy_score(label, pred)
    return


@app.cell
def _(label):
    (label.numpy() == 1).astype(int)
    return


@app.cell
def _(label, matthews_corrcoef, pred):
    for i in range(10):
        curr_label = (label.numpy() == i).astype(int)
        curr_pred = (pred.numpy() == i).astype(int)
        print(f"class {i}: {matthews_corrcoef(curr_label, curr_pred)}")
    return


@app.cell
def _(label, precision_score, pred):
    precision_score(label, pred, average=None)
    return


@app.cell
def _(label, pred, recall_score):
    recall_score(label, pred, average=None)
    return


@app.cell
def _(ConfusionMatrixDisplay, matrix, plt):
    ConfusionMatrixDisplay(matrix).plot()
    plt.show()
    return


@app.cell
def _(matrix, np):
    new_pred = np.array([])
    new_gt = np.array([])
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            m_value = matrix[j, k]
            new_pred = np.append(new_pred, np.full((m_value,),k))
            new_gt = np.append(new_gt, np.full((m_value,),j))
    return new_gt, new_pred


@app.cell
def _(ConfusionMatrixDisplay, confusion_matrix, new_gt, new_pred, plt):
    ConfusionMatrixDisplay(confusion_matrix(new_gt, new_pred)).plot()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
