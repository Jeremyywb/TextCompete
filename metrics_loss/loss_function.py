import torch
import numpy as np
from torch import nn
from sklearn.metrics import mean_squared_error

class RMSELoss(nn.Module):
    def __init__(self, reduction='none', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred.float(), y_true.float()) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class CommonLitLoss(nn.Module):
    def __init__(self, loss_name='RMSELoss',loss_param = {},reduction="mean",weights=None,device=""):
        super().__init__()
        self.loss_func = eval(loss_name)(**loss_param)
        self.eps = 1e-9
        self.reduction = reduction
        self.weights = torch.tensor(weights).to(device) if weights else None

    def forward(self, y_pred, y_true):
        loss = self.loss_func(y_pred.float(), y_true.float())
        if self.weights is not None:
            loss = loss * self.weights
            if self.reduction == 'sum':
                loss = loss.sum()
            else:
                loss = loss.sum() / self.weights.sum()
        else:
            if self.reduction == 'sum':
                loss = loss.sum()
            else:
                loss = loss.mean()
        return loss


# def mcrmse(targets, predictions):
#     error = targets - predictions
#     squared_error = np.square(error)
#     colwise_mse = np.mean(squared_error, axis=0)
#     root_colwise_mse = np.sqrt(colwise_mse)
#     return np.mean(root_colwise_mse, axis=0)


# def comp_metric(outputs, targets):
#     colwise_rmse = torch.sqrt(torch.mean(torch.square(targets - outputs), dim=0))
#     metric = torch.mean(colwise_rmse, dim=0)
#     return metric, colwise_rmse

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(args, _name,y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    met = {_name:mcrmse_score}
    for i,col in enumerate(args.model['target_cols']):
        met[f'{_name[0].upper()}{col}'] = scores[i]
    return met