# Base Dependencies
# -----------------
import numpy as np
from abc import ABC, abstractmethod
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, Optional, Union, List

# Package Dependencies
# --------------------
from .config import PLExperimentConfig, ALExperimentConfig
from .utils import compute_metrics

# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection
from utils import ddi_binary_relation

# 3rd-Party Dependencies
# ----------------------
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

# Constants
# ---------
from constants import CHECKPOINTS_CACHE_DIR, DATASETS


# BaseTrainer
# -----------
class BaseTrainer(ABC):
    def __init__(
        self,
        dataset: str,
        train_dataset: Union[RelationCollection, Dataset],
        test_dataset: Union[RelationCollection, Dataset],
        relation_type: Optional[str] = None,
    ):
        """
        Args:
            dataset (str): name of the dataset, e.g., "n2c2".
            train_dataset (Dataset): train split of the dataset.
            test_dataset (Dataset): test split of the dataset.
            relation_type (str, optional): relation type. Defaults to None.

        Raises:
            ValueError: if the name dataset provided is not supported
        """
        if dataset not in DATASETS:
            raise ValueError("unsupported dataset '{}'".format(dataset))

        self.dataset = dataset
        self.relation_type = relation_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # get total number of instances, tokens and characters
        if isinstance(self.train_dataset, RelationCollection):
            self.n_instances = self.train_dataset.n_instances
            self.n_tokens = self.train_dataset.n_tokens
            self.n_characters = self.train_dataset.n_characters
        else:
            self.n_instances = len(self.train_dataset)
            self.n_tokens = self.train_dataset["seq_length"].sum().item()
            self.n_characters = self.train_dataset["char_length"].sum().item()

    @property
    @abstractmethod
    def method_name(self) -> str:
        pass

    @property
    @abstractmethod
    def method_name_pretty(self) -> str:
        pass

    @property
    def use_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def metrics_average(self) -> str:
        if self.dataset == "n2c2":
            avg = "binary"
        else:
            avg = "micro"

        return avg

    @property
    def num_classes(self) -> int:
        if self.dataset == "n2c2":
            n = 2
        else:
            n = 5
        return n

    @property
    def pl_checkpoint_path(self):
        """Pasive Learning checkpoints directory path"""
        directory = Path(
            pjoin(CHECKPOINTS_CACHE_DIR, "pl", self.method_name, self.dataset)
        )
        if self.relation_type:
            directory = Path(pjoin(directory, self.relation_type))

        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=False)

        return directory

    @property
    def al_checkpoint_path(self):
        """Active Learning checkpoints directory path"""
        directory = Path(
            pjoin(CHECKPOINTS_CACHE_DIR, "al", self.method_name, self.dataset)
        )
        if self.relation_type:
            directory = Path(pjoin(directory, self.relation_type))

        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=False)

        return directory

    # Instance Methods
    # ----------------
    @abstractmethod
    def train_passive_learning(
        self, config: PLExperimentConfig, verbose: bool = True, logging: bool = True
    ):
        """Trains the model using Passive Learning"""
        raise NotImplementedError()

    @abstractmethod
    def train_active_learning(
        self,
        query_strategy: str,
        config: ALExperimentConfig,
        verbose: bool = True,
        logging: bool = True,
    ):
        """Trains the model using Active Learning"""
        raise NotImplementedError()
    
    def print_info_passive_learning(self) -> None:
        """Prints information about the Passive Learning training process"""
        print(f"\n\n**** {self.method_name_pretty} - Train Passive Learning ****")
        print(f" - Dataset: {self.dataset}")
        if self.relation_type:
            print(f" - Relation type: {self.relation_type}")

    def print_info_active_learning(
        self, q_strategy: str, pool_size: int, init_q_size: int, q_size: int
    ) -> None:
        """Prints information about the Active Learning training process"""
        print(f"\n\n**** {self.method_name_pretty} - Train Active Learning ****")
        print(f"  - Dataset: {self.dataset}")
        if self.relation_type:
            print(f"  - Relation type: {self.relation_type}")
        print(f"  - Strategy = {q_strategy}")
        print(f"  - Pool size = {pool_size}")
        print(f"  - Initial query size = {init_q_size}")
        print(f"  - Query size = {q_size}")

    def compute_class_weights(self, labels: list) -> Optional[torch.Tensor]:
        """Computes the class weights for the given labels"""
        if len(np.unique(labels)) == self.num_classes:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array(range(self.num_classes)),
                y=labels,
            )
            class_weights = torch.from_numpy(class_weights).float().to(self.device)
        else:
            class_weights = None

        return class_weights

    def compute_init_q_size(self, config: ALExperimentConfig) -> int:
        """Computes the initial pool size for the given configuration"""
        return min(
            config.max_query_size,
            int(round(config.initial_pool_perc * self.n_instances)),
        )

    def compute_q_size(self, config: ALExperimentConfig) -> int:
        """Computes the query size for the given configuration"""
        return min(
            config.max_query_size,
            int(round(config.query_size_perc * self.n_instances)),
        )

    def compute_al_steps(self, config: ALExperimentConfig) -> int:
        """Computes the number of active learning steps for the given configuration"""
        query_size = self.compute_q_size(config)
        return int(round(self.n_instances * config.max_annotation / query_size)) - 1

    def compute_step_accuracy(self, y_true: list, y_pred: list) -> float:
        """Computes the accuracy for the given step"""
        return accuracy_score(y_true, y_pred, normalize=True)
    
    def compute_metrics(
        self,
        y_true: list,
        y_pred: list,
        labels: Optional[List[str]] = None,
        pos_label: int = 1,
    ) -> Dict:
        """Computes metrics

        Args:
            y_true (list): list of ground truths
            y_pred (list): list of predicted values
            labels (Optional[List[str]], optional): list of labels. Defaults to None.
            pos_label (int, optional): positive label. Defaults to 1.

        Returns:
            Dict: precision, recall and F1-score
        """
        if self.dataset == "n2c2":
            metrics = self.compute_metrics_n2c2(y_true, y_pred, labels, pos_label)
        else: # ddi
            metrics = self.compute_metrics_ddi(y_true, y_pred, labels, pos_label)

        return metrics             

    def compute_metrics_n2c2(
        self,
        y_true: list,
        y_pred: list,
        labels: Optional[List[str]] = None,
        pos_label: int = 1,
    ):
        metrics = {}

        # accuracy
        metrics["acc"] = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)

        avg_metrics = compute_metrics(
            y_true=y_true, y_pred=y_pred, average=self.metrics_average, pos_label=1
        )
        for key, value in avg_metrics.items():
            metrics[key] = value

        return metrics

    def compute_metrics_ddi(
        self,
        y_true: list,
        y_pred: list,
        labels: Optional[List[str]] = None,
        pos_label: int = 1,
    ):  
        metrics = {}

        # accuracy
        metrics["acc"] = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)

        # macro
        relevant_classes = [1, 2, 3, 4]
        relevant_indices = np.isin(y_true, relevant_classes)
        micro_metrics = compute_metrics(
            y_true=y_true[relevant_indices],
            y_pred=y_pred[relevant_indices],
            average="micro",
        )
        for key, value in micro_metrics.items():
            metrics[key] = value

        # ddi: "Detection"
        y_true_binary = list(map(lambda x: ddi_binary_relation(x), y_true))
        y_pred_binary = list(map(lambda x: ddi_binary_relation(x), y_pred))

        detection_metrics = compute_metrics(
            y_true=y_true_binary, y_pred=y_pred_binary, average="binary"
        )
        for key, value in detection_metrics.items():
            metrics["detect_" + key] = value

        # ddi: per class
        per_class_metrics = compute_metrics(
            y_true=y_true, y_pred=y_pred, average=None, labels=[0, 1, 2, 3, 4]
        )
        for key, values in per_class_metrics.items():
            for i, value in enumerate(values):
                if labels:
                    class_name = labels[i]
                else:
                    class_name = str(i)
                metrics["class_" + key + "_" + class_name] = value

        return metrics   

    # Class methods
    # --------------
    @classmethod
    def print_al_iteration_metrics(cls, step: int, metrics: Dict[str, float]):
        print("\n** Iteration {} - Metrics **".format(step), flush=True)
        for key, value in metrics.items():
            print("  - {} = {}".format(key, value), flush=True)
        print("")

    @classmethod
    def print_val_metrics(cls, epoch: int, metrics: Dict[str, float]):
        print("\n** Epoch {} - Validation set - Metrics **".format(epoch), flush=True)
        for key, value in metrics.items():
            print("  - {} = {}".format(key, value), flush=True)
        print("")

    @classmethod
    def print_train_metrics(cls, metrics: Dict[str, float]):
        print("\n** Training set - Metrics **", flush=True)
        for key, value in metrics.items():
            print("  - {} = {}".format(key, value), flush=True)
        print("")

    @classmethod
    def print_test_metrics(cls, metrics: Dict[str, float]):
        print("\n** Test set - Metrics **", flush=True)
        for key, value in metrics.items():
            print("  - {} = {}".format(key, value), flush=True)
        print("")
