import marimo

__generated_with = "0.20.2"
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
    from src.engine.CNN import CNN
    from src.data.read_data import get_dataloaders, DATA_TRANSFORM

    return CNN, DATA_TRANSFORM, get_dataloaders, nn, optim, torch


@app.cell
def _(DATA_TRANSFORM, get_dataloaders):
    train_loader, val_loader, _ = get_dataloaders(DATA_TRANSFORM)
    return (train_loader,)


@app.cell
def _(CNN, nn, optim):
    model = CNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate 
    return criterion, model, optimizer


@app.cell
def _(criterion, model, optimizer, train_loader):
    num_epochs = 5  #change as needed, paper mentioned up to 105 :O

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images, labels

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}") #prints out corss-entropy loss per epoch
    return


@app.cell
def _(model, torch):
    torch.save(model, "./CNN_model.pth")
    return


if __name__ == "__main__":
    app.run()
