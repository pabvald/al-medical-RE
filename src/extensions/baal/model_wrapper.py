# Base Dependencies
# -----------------
import sys
import structlog
from math import floor
from typing import Callable, Optional

# PyTorch Dependencies
# --------------------
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# Baal Dependencies
# ------------------
from baal.active.dataset.base import Dataset
from baal.modelwrapper import ModelWrapper
from baal.utils.iterutils import map_on_tensor

log = structlog.get_logger("ModelWrapper")


# Model Wrappers 
# --------------
class MyModelWrapperBilstm(ModelWrapper):
    """
    MyModelWrapper

    Modification of ModelWrapper to allow a transform on a batch from a
    HF Dataset with several inputs (i.e. dictionary of tensors)
    """
    def __init__(self, model, criterion, replicate_in_memory=True, min_train_passes: int = 10):
        super().__init__(model, criterion, replicate_in_memory)
        self.min_train_passes = min_train_passes
        self.batch_sizes = []

    def _compute_batch_size(self, n_labelled: int, max_batch_size: int):
        bs =  min(int(floor(n_labelled / self.min_train_passes)), max_batch_size)
        bs = max(2, bs)
        return bs

    def train_on_dataset(
        self,
        dataset: Dataset,
        optimizer: torch.optim,
        batch_size: int,
        epoch: int,
        use_cuda: bool,
        workers: int = 2,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
    ):
        """
        Train for `epoch` epochs on a Dataset `dataset.
        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            optimizer (optim.Optimizer): Optimizer to use.
            batch_size (int): The batch size used in the DataLoader.
            epoch (int): Number of epoch to train for.
            use_cuda (bool): Use cuda or not.
            workers (int): Number of workers for the multiprocessing.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.
        Returns:
            The training history.
        """
        

        dataset_size = len(dataset)
        actual_batch_size = batch_size #self._compute_batch_size(dataset_size, batch_size)
        self.batch_sizes.append(actual_batch_size) 
        self.train()
        self.set_dataset_size(dataset_size)
        history = []
        log.info("Starting training", epoch=epoch, dataset=dataset_size)
        collate_fn = collate_fn or default_collate
        sampler = BatchSampler(
            RandomSampler(dataset), batch_size=actual_batch_size, drop_last=False
        )
        dataloader = DataLoader(
            dataset, sampler=sampler, num_workers=workers, collate_fn=collate_fn
        )

        for _ in range(epoch):
            self._reset_metrics("train")
            for data, target, *_ in dataloader:
                _ = self.train_on_batch(data, target, optimizer, use_cuda, regularizer)
            history.append(self.get_metrics("train")["train_loss"])

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info(
            "Training complete", train_loss=self.get_metrics("train")["train_loss"]
        )
        self.active_step(dataset_size, self.get_metrics("train"))
        return history

    def test_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        use_cuda: bool,
        workers: int = 2,
        collate_fn: Optional[Callable] = None,
        average_predictions: int = 1,
    ):
        """
        Test the model on a Dataset `dataset`.
        Args:
            dataset (Dataset): Dataset to evaluate on.
            batch_size (int): Batch size used for evaluation.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            average_predictions (int): The number of predictions to average to
                compute the test loss.
        Returns:
            Average loss value over the dataset.
        """
        self.eval()
        log.info("Starting evaluating", dataset=len(dataset))
        self._reset_metrics("test")

        sampler = BatchSampler(
            RandomSampler(dataset), batch_size=batch_size, drop_last=False
        )
        dataloader = DataLoader(
            dataset, sampler=sampler, num_workers=workers, collate_fn=collate_fn
        )

        for data, target, *_ in dataloader:
            _ = self.test_on_batch(
                data, target, cuda=use_cuda, average_predictions=average_predictions
            )

        log.info("Evaluation complete", test_loss=self.get_metrics("test")["test_loss"])
        self.active_step(None, self.get_metrics("test"))
        return self.get_metrics("test")["test_loss"]

    def predict_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 2,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.
        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to display progress
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.
        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.eval()
        if len(dataset) == 0:
            return None

        log.info("Start Predict", dataset=len(dataset))
        collate_fn = collate_fn or default_collate
        sampler = BatchSampler(
            RandomSampler(dataset), batch_size=batch_size, drop_last=False
        )
        loader = DataLoader(
            dataset, sampler=sampler, num_workers=workers, collate_fn=collate_fn
        )

        if verbose:
            loader = tqdm(loader, total=len(loader), file=sys.stdout)
        for idx, (data, *_) in enumerate(loader):

            pred = self.predict_on_batch(data, iterations, use_cuda)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

