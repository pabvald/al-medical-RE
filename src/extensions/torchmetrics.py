# Base Dependencies
# -----------------
import numpy as np
from typing import Optional

# 3rd-Party Dependencies
# -----------------
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.checks import _input_format_classification
from sklearn.metrics import precision_score, recall_score, f1_score


def _make_binary(preds: torch.Tensor, target: torch.Tensor):

    # obtain decimal values from one-hot encoding
    preds2 = preds.argmax(axis=1).int()
    target2 = target.argmax(axis=1).int()
    
    # replace positive classes by 1
    preds2[preds2 != 0] = 1
    target2[target2 != 0] = 1

    return preds2, target2


class DetectionF1Score(Metric):
    def __init__(self, ) -> None:
        super().__init__()
        self.add_state("y_true", default=torch.Tensor([]).int())
        self.add_state("y_pred", default=torch.Tensor([]).int())

    def update(self, preds: torch.Tensor, target: torch.Tensor): 
        preds, target, datatype = _input_format_classification(preds, target)
        p, t = _make_binary(preds, target)
        self.y_pred = torch.cat((self.y_pred, p))
        self.y_true = torch.cat([self.y_true, t])

    def compute(self) -> torch.Tensor:
        """Computes f-beta over state."""
        score =  f1_score(y_true=self.y_true.cpu().numpy(), y_pred=self.y_pred.cpu().numpy(), average="binary")
        return torch.tensor(score)


class DetectionPrecision(Metric):
    def __init__(self, ) -> None:
        super().__init__()
        self.add_state("y_true", default=torch.Tensor([]).int())
        self.add_state("y_pred", default=torch.Tensor([]).int())

    def update(self, preds: torch.Tensor, target: torch.Tensor): 
        preds, target, datatype = _input_format_classification(preds, target)
        p, t = _make_binary(preds, target)
        self.y_pred = torch.cat((self.y_pred, p))
        self.y_true = torch.cat([self.y_true, t])

    def compute(self) -> torch.Tensor: 
        score = precision_score(y_true=self.y_true.cpu().numpy(), y_pred=self.y_pred.cpu().numpy(), average="binary")
        return torch.tensor(score)


class DetectionRecall(Metric):
    def __init__(self, ) -> None:
        super().__init__()
        self.add_state("y_true", default=torch.Tensor([]).int())
        self.add_state("y_pred", default=torch.Tensor([]).int())

    def update(self, preds: torch.Tensor, target: torch.Tensor): 
        preds, target, datatype = _input_format_classification(preds, target)
        p, t = _make_binary(preds, target)
        self.y_pred = torch.cat((self.y_pred, p))
        self.y_true = torch.cat([self.y_true, t])

    def compute(self) -> torch.Tensor:
        score = recall_score(y_true=self.y_true.cpu().numpy(), y_pred=self.y_pred.cpu().numpy(), average="binary")
        return torch.tensor(score)
