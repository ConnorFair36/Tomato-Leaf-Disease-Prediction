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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.data.read_data import get_dataloaders
    from src.data.preprocessors import DATA_TRANSFORM
    from src.models.CNN import CNN
    from src.engine.train_loop import train_model

    return CNN, DATA_TRANSFORM, get_dataloaders, nn, optim, torch, train_model


@app.cell
def _(CNN, DATA_TRANSFORM, get_dataloaders, nn, torch):
    # if you have a saved model that you are loading in, make sure to load it BEFORE running compile
    train_data, val_data, _ = get_dataloaders(DATA_TRANSFORM, n_workers=2)
    model = CNN()
    model = torch.compile(model)
    loss_function = nn.CrossEntropyLoss()
    return loss_function, model, train_data, val_data


@app.cell
def _(loss_function, model, optim, train_data, train_model, val_data):
    train_model(
        model=model,
        train_loader=train_data,
        val_loader=val_data,
        epochs=1,
        loss_fn=loss_function,
        optimizer=optim.Adam(params=model.parameters())
    )
    return


@app.cell
def _(model, optimizer):
    # saving the model
    checkpoint = {
        'model_state_dict': model._orig_mod.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # torch.save(checkpoint, 'path/to/checkpoint/model.pt')

    return


if __name__ == "__main__":
    app.run()
