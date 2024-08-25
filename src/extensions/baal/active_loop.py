# Base Dependencies
# -----------------
import os
import pickle
import types
import numpy as np
import structlog

from typing import Tuple, Callable


# 3rd-Party Dependencies
# ----------------------
import torch.utils.data as torchdata

from baal.active import ActiveLearningLoop
from baal.active.heuristics import heuristics
from baal.active.dataset import ActiveLearningDataset

log = structlog.get_logger(__name__)
pjoin = os.path.join


class MyActiveLearningLoop(ActiveLearningLoop):
    def __init__(
        self,
        dataset: ActiveLearningDataset,
        get_probabilities: Callable,
        heuristic: heuristics.AbstractHeuristic = heuristics.Random(),
        query_size: int = 1,
        max_sample=-1,
        uncertainty_folder=None,
        ndata_to_label=None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset,
            get_probabilities,
            heuristic,
            query_size,
            max_sample,
            uncertainty_folder,
            ndata_to_label,
            **kwargs,
        )
        self.step_acc = []
        self.step_score = []

    def compute_step_metrics(self, probs: np.array, to_label: list):
        """
        Register the accuracy and the avg. prediction score the trained model has on the queried instances

        Args:
            probs (np.array): probabilities of the model on the unlabelled pool
            to_label (list): list of indices to be labelled
        """
        pool = self.dataset.pool

        # obtain true labels of queried examples
        y_true = []
        for idx in to_label[: self.query_size]:
            y_true.append(pool[idx]["label"])
        y_true = np.array(y_true)

        # obtain predicted labels of queried examples
        # 1. avg over MC Dropout iterations to obtain prob per class
        avg_iter_probs = np.mean(probs[: self.query_size], axis=2)
        # 2. get class with highest prob
        y_pred = np.argmax(avg_iter_probs, axis=1)
        assert len(y_pred) == len(y_true)

        # accuracy on the queried examples
        acc = np.mean(y_true == y_pred)
        self.step_acc.append(acc)

        # average predicted score on  true class
        avg_probs = []
        for true_class, classes_probs in zip(y_true, avg_iter_probs):
            avg_probs.append(classes_probs[true_class])

        self.step_score.append(np.mean(avg_probs))

    def step(self, pool=None) -> Tuple[bool, dict]:
        """
        Perform an active learning step.
        Args:
            pool (iterable): Optional dataset pool indices.
                             If not set, will use pool from the active set.
        Returns:
            boolean, Flag indicating if we continue training.
        """
        if pool is None:
            pool = self.dataset.pool
            if len(pool) > 0:
                # Limit number of samples
                if self.max_sample != -1 and self.max_sample < len(pool):
                    indices = np.random.choice(
                        len(pool), self.max_sample, replace=False
                    )
                    pool = torchdata.Subset(pool, indices)
                else:
                    indices = np.arange(len(pool))
        else:
            indices = None

        if len(pool) > 0:
            probs = self.get_probabilities(pool, **self.kwargs)
            if probs is not None and (
                isinstance(probs, types.GeneratorType) or len(probs) > 0
            ):
                to_label, uncertainty = self.heuristic.get_ranks(probs)
                if indices is not None:
                    # re-order the sampled indices based on the uncertainty
                    to_label = indices[np.array(to_label)]
                if self.uncertainty_folder is not None:
                    # We save uncertainty in a file.
                    uncertainty_name = (
                        f"uncertainty_pool={len(pool)}"
                        f"_labelled={len(self.dataset)}.pkl"
                    )
                    pickle.dump(
                        {
                            "indices": indices,
                            "uncertainty": uncertainty,
                            "dataset": self.dataset.state_dict(),
                        },
                        open(pjoin(self.uncertainty_folder, uncertainty_name), "wb"),
                    )
                if len(to_label) > 0:
                    self.compute_step_metrics(probs, to_label)
                    self.dataset.label(to_label[: self.query_size])
                    return True

        return False
