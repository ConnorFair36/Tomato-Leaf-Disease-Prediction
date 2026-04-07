import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.models.CNN import CNN
    from src.data.read_data import get_dataloaders
    from src.data.preprocessors import DATA_TRANSFORM, DATA_TRANSFORM_DOWNSAMPLE
    from src.engine.train_loop import train_model

    return CNN, DATA_TRANSFORM, get_dataloaders, nn, optim, torch, train_model


@app.cell
def _(DATA_TRANSFORM, get_dataloaders):
    train_loader, val_loader, _ = get_dataloaders(DATA_TRANSFORM)
    return train_loader, val_loader


@app.cell
def _(CNN, nn, optim):
    model = CNN(num_classes=10, dropout_rate=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) #learning rate 
    return criterion, model, optimizer


@app.cell
def _(criterion, model, optimizer, train_loader, train_model, val_loader):
    num_epochs = 50  #change as needed, paper mentioned up to 100 :O

    losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=num_epochs,
            loss_fn=criterion,
            optimizer=optimizer
        )
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "../src/weights/CNN_model_weights.pth")
    return


if __name__ == "__main__":
    app.run()
