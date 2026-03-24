import numpy as np
import torch
from torch import nn

from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, precision_score, recall_score

def evaluate_model(model: torch.Module, dataset: torch.utils.data.dataloader.DataLoader) -> dict:
    """Returns a dictionary containing all of the values used for evaluating model performance based on the given dataset."""
    # get the device that the model is currently on
    device = model.parameters().__next__().device
    class_to_idx = dataset.dataset.class_to_idx
    idx_to_class = {value: key[9:] for key, value in class_to_idx.items()}
    matrix = np.zeros((10,10))
    # loop throught the given evaluation dataset
    for images, ground_truth in dataset:
        # put the images onto the same device as the model
        images = images.to(device)
        with torch.no_grad():
            pred = model(images)
        # take the argmax of predictions if needed
        if pred.dim() == 2:
            pred = torch.argmax(pred, dim=1)
        pred = pred.to(device="cpu")
        matrix += confusion_matrix(ground_truth, pred)
    # reverse the confusion matrix because I'm lazy and want scikit learn to do all the work, lol
    pred = np.array([])
    ground_truth = np.array([])
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            m_value = matrix[j, k]
            pred = np.append(pred, np.full((m_value,),k))
            ground_truth = np.append(ground_truth, np.full((m_value,),j))
    # calculate per-class metrics
    prec_values = {f"precision_{idx_to_class[idx]}": value for idx, value in enumerate(precision_score(ground_truth, pred, average=None))}
    recall_values = {f"recall_{idx_to_class[idx]}": value for idx, value in enumerate(recall_score(ground_truth, pred, average=None))}
    mcc_values = dict()
    for i in range(matrix.shape[0]):
        curr_label = (ground_truth == i).astype(int)
        curr_pred = (pred == i).astype(int)
        mcc_values[f"mcc_{idx_to_class[i]}"] = matthews_corrcoef(curr_label, curr_pred)
    # combine everything to create the final evaluation dictionary
    evaluation = {
        "confusion_matrix": matrix,
        "accuracy":         accuracy_score(ground_truth, pred)
    }
    evaluation = evaluation | prec_values | recall_values | mcc_values
    return evaluation
        