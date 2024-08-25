# Base Dependencies
# -----------------
from typing import Union, Tuple

# 3-rd Party Dependencies
# -----------------------
import torch
from torch import nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """Custom Transformers' Trainer which uses a weighted cross entropy loss"""

    class_weights: torch.Tensor = None

    def compute_loss(
        self, model: nn.Module, inputs: dict, return_outputs: bool = False
    ) -> Union[Tuple[float, dict], float]:
        """Computes the weighted cross entropy loss of a model on the batch of inputs.

        Args:
            model (nn.Module): Transformers model
            inputs (dict): batch of inputs
            return_outputs(bool):

        Returns:
            Union[Tuple[float, dict], float]: loss and outputs of the model, or loss
        """
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
