import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, pos_e_w):
        super().__init__()
        self.edge_loss = CrossEntropyMetric(pos_e_w)
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_E = self.edge_loss(flat_pred_E, flat_true_E, loss_type = 'edge') if true_E.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": loss_E.detach(),
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                }
            if wandb.run:
                wandb.log(to_log, commit=True)
        # return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y
        return self.lambda_train[0] * loss_E

    def reset(self):
        self.edge_loss.reset()

    def log_epoch_metrics(self):
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        to_log = {
            "train_epoch/E_CE": epoch_edge_loss
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



