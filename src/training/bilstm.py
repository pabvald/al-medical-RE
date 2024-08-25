# Base Dependencies
# -----------------
import numpy as np
import time
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional
from pathlib import Path
from os.path import join

# Package Dependencies
# --------------------
from .base import BaseTrainer
from .config import PLExperimentConfig, BaalExperimentConfig
from .early_stopping import EarlyStopping
from .utils import get_baal_query_strategy

# Local Dependencies
# -------------------
from extensions.baal import (
    MyModelWrapperBilstm,
    MyActiveLearningDatasetBilstm,
    MyActiveLearningLoop,
)
from extensions.torchmetrics import (
    DetectionF1Score,
    DetectionPrecision,
    DetectionRecall,
)
from ml_models.bilstm import (
    HasanModel,
    EmbeddingConfig,
    LSTMConfig,
    RDEmbeddingConfig,
)
from re_datasets.bilstm_utils import pad_and_sort_batch, custom_collate
from vocabulary import Vocabulary, read_list_from_file

# 3rd-Party Dependencies
# ----------------------
import neptune
import torch

from baal.bayesian.dropout import patch_module
from datasets import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torchmetrics import Accuracy
from torchmetrics.classification import F1Score, Precision, Recall

# Constants
# ---------
from constants import (
    N2C2_VOCAB_PATH,
    DDI_VOCAB_PATH,
    N2C2_IOB_TAGS,
    DDI_IOB_TAGS,
    N2C2_RD_MAX,
    DDI_RD_MAX,
    RD_EMB_DIM,
    IOB_EMB_DIM,
    BIOWV_EMB_DIM,
    POS_EMB_DIM,
    DEP_EMB_DIM,
    BIOWORD2VEC_PATH,
    U_POS_TAGS,
    DEP_TAGS,
    BaalQueryStrategy,
)
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT


class BilstmTrainer(BaseTrainer):
    """Trainer for BiLSTM method."""

    def __init__(
        self,
        dataset: str,
        train_dataset: Dataset,
        test_dataset: Dataset,
        relation_type: Optional[str] = None,
    ):
        """
        Args:
            dataset (str): name of the dataset, e.g., "n2c2".
            train_dataset (Dataset): train split of the dataset.
            test_dataset (Dataset): test split of the dataset.
            relation_type (str, optional): relation type.

        Raises:
            ValueError: if the name dataset provided is not supported
        """
        super().__init__(dataset, train_dataset, test_dataset, relation_type)

        # vocabulary
        self.vocab = self._init_vocab()

        # transform datasets
        self.transform = partial(
            pad_and_sort_batch, padding_idx=self.vocab.pad_index, rd_max=self.RD_MAX
        )

    @property
    def method_name(self) -> str:
        return "bilstm"

    @property
    def method_name_pretty(self) -> str:
        return "BiLSTM"

    @property
    def task(self) -> str:
        if self.dataset == "n2c2":
            task = "binary"
        else:
            task = "multiclass"
        return task

    @property
    def model_class(self) -> str:
        return HasanModel

    @property
    def RD_MAX(self) -> str:
        if self.dataset == "n2c2":
            rd_max = N2C2_RD_MAX
        else:
            rd_max = DDI_RD_MAX
        return rd_max

    @property
    def IOB_TAGS(self) -> str:
        if self.dataset == "n2c2":
            iob_tags = N2C2_IOB_TAGS
        else:
            iob_tags = DDI_IOB_TAGS
        return iob_tags

    def _init_optimizer(self, model: Module):
        return Adam(model.parameters(), lr=0.0001)

    def _init_vocab(self):
        """Loads the vocabulary of the dataset"""
        if self.dataset == "n2c2":
            vocab_path = N2C2_VOCAB_PATH
        else:
            vocab_path = DDI_VOCAB_PATH

        return Vocabulary(read_list_from_file(vocab_path))
    
    def _init_model(self, patch: bool = False) -> HasanModel:
        """Builds the BiLSTM model setting the right configuration for the chosen dataset"""
        # word embedding configuration
        biowv_config = EmbeddingConfig(
            embedding_dim=BIOWV_EMB_DIM,
            vocab_size=len(self.vocab),
            emb_path=BIOWORD2VEC_PATH,
            freeze=True,
            padding_idx=self.vocab.pad_index,
        )

        # relative-distance embedding configuration
        rd_config = RDEmbeddingConfig(
            input_dim=self.RD_MAX, embedding_dim=RD_EMB_DIM, freeze=False
        )

        # IOB embedding configuration
        iob_config = EmbeddingConfig(
            embedding_dim=IOB_EMB_DIM, vocab_size=(len(self.IOB_TAGS) + 1), freeze=False
        )

        # Part-of-Speach tag embedding configuration
        pos_config = EmbeddingConfig(
            embedding_dim=POS_EMB_DIM, vocab_size=(len(U_POS_TAGS) + 1), freeze=False
        )

        dep_config = EmbeddingConfig(
            embedding_dim=DEP_EMB_DIM, vocab_size=(len(DEP_TAGS) + 1), freeze=False
        )

        # BiLSTM configuration
        lstm_config = LSTMConfig(
            emb_size=(
                BIOWV_EMB_DIM + 2 * RD_EMB_DIM + POS_EMB_DIM + DEP_EMB_DIM + IOB_EMB_DIM
            )
        )

        model = self.model_class(
            vocab=self.vocab,
            lstm_config=lstm_config,
            bioword2vec_config=biowv_config,
            rd_config=rd_config,
            pos_config=pos_config,
            dep_config=dep_config,
            iob_config=iob_config,
            num_classes=self.num_classes,
        )

        if patch:
            model = patch_module(model)

        return model

    def _reset_trainer(self):
        self.train_dataset.reset_format()
        self.test_dataset.reset_format()

    def create_dataloader(self, dataset: Dataset, batch_size: int = 6) -> DataLoader:
        """Creates a dataloader from a dataset with the adequate configuration

        Args:
            dataset (Dataset): dataset to load

        Returns:
            DataLoader: dataloader for the given dataset
        """
        dataset.set_transform(self.transform)

        # create dataloader
        sampler = BatchSampler(
            RandomSampler(dataset), batch_size=batch_size, drop_last=False
        )
        dataloader = DataLoader(dataset, sampler=sampler, collate_fn=custom_collate)

        return dataloader

    def eval_model(
        self,
        model: Module,
        dataloader: DataLoader,
        criterion: Module,
    ) -> Dict[str, float]:
        """Evaluates the current model on the dev or test set

        Args:
            model (Module): model to use for evaluation.
            dataloader (DataLoader): dataloader of evaluation dataset
        Returns:
            Dict: metrics including loss (`loss`), precision (`p`), recall (`r`) and F1-score (`f1`)
        """

        y_true = np.array([], dtype=np.int8)
        y_pred = np.array([], dtype=np.int8)

        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloader:
                # send (inputs, labels) to device
                labels = labels.to(self.device)
                for key, value in inputs.items():
                    inputs[key] = value.to(self.device)

                # calculate outputs
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += len(inputs) * loss.item()

                # calculate predictions
                _, predicted = torch.max(outputs.data, 1)

                # store labels and predictions
                y_true = np.append(y_true, labels.cpu().detach().numpy())
                y_pred = np.append(y_pred, predicted.cpu().detach().numpy())

        metrics = self.compute_metrics(y_true, y_pred)
        metrics["loss"] = val_loss / len(dataloader)

        return metrics

    def train_passive_learning(
        self, config: PLExperimentConfig, verbose: bool = True, logging: bool = True
    ):
        """Trains the BiLSTM model using passive learning and early stopping

        Args:
            config (PLExperimentConfig): cofiguration
            verbose (bool): determines if information is printed during training. Daults to True.
            logging (bool): log the test metrics on Neptune. Defaults to True.
        """
        self._reset_trainer()

        # setup
        train_val_split = self.train_dataset.train_test_split(
            test_size=config.val_size, stratify_by_column="label"
        )
        labels = np.array(train_val_split["train"]["label"])

        train_dataloader = self.create_dataloader(
            train_val_split["train"], batch_size=config.batch_size
        )

        val_dataloader = self.create_dataloader(
            train_val_split["test"], batch_size=config.batch_size
        )
        test_dataloader = self.create_dataloader(
            self.test_dataset, batch_size=config.batch_size
        )

        if logging:
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        model = self._init_model()
        model = model.to(self.device)
        criterion = CrossEntropyLoss(weight=self.compute_class_weights(labels))
        optimizer = self._init_optimizer(model)

        # print info
        if verbose:
            self.print_info_passive_learning()

        # early stopper
        ES = EarlyStopping(
            patience=config.es_patience,
            verbose=True,
            path=Path(join(self.pl_checkpoint_path, "best_model.pt")),
        )

        # training loop
        for epoch in range(config.max_epoch):
            running_loss = 0.0
            for i, (inputs, labels) in tqdm(enumerate(train_dataloader, 0)):
                # get the inputs; data is a list of [inputs, labels]
                labels = labels.to(self.device)
                for key, value in inputs.items():
                    inputs[key] = value.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            # evaluate model on validation set
            val_metrics = self.eval_model(model, val_dataloader, criterion)
            train_loss = running_loss / len(train_dataloader)
            val_loss = val_metrics["loss"]
            running_loss = 0.0
            if logging:
                run["loss/train"].append(train_loss)
                run["loss/val"].append(val_loss)

                for key, value in val_metrics.items():
                    if key != "loss":
                        run[f"val/{key}"].append(value)

            if verbose:
                self.print_val_metrics(epoch + 1, val_metrics)

            # check early stopping
            ES(val_loss, model)
            if ES.early_stop:
                break

        # load best model
        model.load_state_dict(
            torch.load(Path(join(self.pl_checkpoint_path, "best_model.pt")))
        )

        # evaluate model on test dataset
        test_metrics = self.eval_model(model, test_dataloader, criterion)
        if verbose:
            self.print_test_metrics(test_metrics)
        if logging:
            run["method"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["strategy"] = "passive learning"
            run["config"] = config.__dict__
            run["epochs"] = epoch

            for key, value in test_metrics.items():
                run["test/" + key] = value
            run.stop()

        return model

    def set_al_metrics(self, baal_model: MyModelWrapperBilstm):
        """
        Configures the metrics that are to be computed during the active learning experiment

        Args:
            baal_model (MyModelWrapperBilstm): model wrapper

        """
        # accuracy
        baal_model.add_metric(
            name="acc",
            initializer=lambda: Accuracy(task=self.task, average="micro").to(
                self.device
            ),
        )

        if self.dataset == "n2c2":
            f1 = F1Score(num_classes=self.num_classes, ignore_index=0).to(self.device)
            p = Precision(num_classes=self.num_classes, ignore_index=0).to(self.device)
            r = Recall(num_classes=self.num_classes, ignore_index=0).to(self.device)
            baal_model.add_metric(name="f1", initializer=lambda: f1)
            baal_model.add_metric(name="p", initializer=lambda: p)
            baal_model.add_metric(name="r", initializer=lambda: r)

        else:  # self.dataset == "ddi":
            # detection + classification metrics
            cla_f1_micro = F1Score(
                num_classes=self.num_classes, average="micro", ignore_index=0
            ).to(self.device)

            cla_p_micro = Precision(
                num_classes=self.num_classes, average="micro", ignore_index=0
            ).to(self.device)

            cla_r_micro = Recall(
                num_classes=self.num_classes, average="micro", ignore_index=0
            ).to(self.device)

            cla_f1_macro = F1Score(
                num_classes=self.num_classes, average="macro", ignore_index=0
            ).to(self.device)

            cla_p_macro = Precision(
                num_classes=self.num_classes, average="macro", ignore_index=0
            ).to(self.device)

            cla_r_macro = Recall(
                num_classes=self.num_classes, average="macro", ignore_index=0
            ).to(self.device)

            baal_model.add_metric(name="micro_f1", initializer=lambda: cla_f1_micro)
            baal_model.add_metric(name="micro_p", initializer=lambda: cla_p_micro)
            baal_model.add_metric(name="micro_r", initializer=lambda: cla_r_micro)
            baal_model.add_metric(name="macro_f1", initializer=lambda: cla_f1_macro)
            baal_model.add_metric(name="macro_p", initializer=lambda: cla_p_macro)
            baal_model.add_metric(name="macro_r", initializer=lambda: cla_r_macro)

            # detection metrics
            detect_f1 = DetectionF1Score().to(self.device)
            detect_p = DetectionPrecision().to(self.device)
            detect_r = DetectionRecall().to(self.device)

            baal_model.add_metric(name="detect_f1", initializer=lambda: detect_f1)
            baal_model.add_metric(name="detect_p", initializer=lambda: detect_p)
            baal_model.add_metric(name="detect_r", initializer=lambda: detect_r)

            # per class metrics
            per_class_f1 = F1Score(num_classes=self.num_classes, average="none").to(
                self.device
            )

            per_class_p = Precision(num_classes=self.num_classes, average="none").to(
                self.device
            )

            per_class_r = Recall(num_classes=self.num_classes, average="none").to(
                self.device
            )

            baal_model.add_metric(name="class_f1", initializer=lambda: per_class_f1)
            baal_model.add_metric(name="class_p", initializer=lambda: per_class_p)
            baal_model.add_metric(name="class_r", initializer=lambda: per_class_r)

        return baal_model

    def train_active_learning(
        self,
        query_strategy: BaalQueryStrategy,
        config: BaalExperimentConfig,
        verbose: bool = True,
        logging: bool = True,
    ):
        """Trains the BiLSTM model using active learning

        Args:
            query_strategy (str): name of the query strategy to be used in the experiment.
            config (BaalExperimentConfig): experiment configuration.
            verbose (bool): determines if information is printed during trainig or not. Defaults to True.s
            logging (bool): log the test metrics on Neptune. Defaults to True.
        """
        self._reset_trainer()

        if logging:
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        # setup querying
        INIT_QUERY_SIZE = self.compute_init_q_size(config)
        QUERY_SIZE = self.compute_q_size(config)
        AL_STEPS = 2 # self.compute_al_steps(config)
        
        f_query_strategy = get_baal_query_strategy(
            name=query_strategy.value,
            shuffle_prop=config.shuffle_prop,
            query_size=QUERY_SIZE,
        )   


        if verbose:
            self.print_info_active_learning(
                q_strategy=query_strategy.value,
                pool_size=self.n_instances,
                init_q_size=INIT_QUERY_SIZE,
                q_size=QUERY_SIZE,
            )

        # setup active set
        self.train_dataset.set_transform(self.transform)
        self.test_dataset.set_transform(self.transform)
        active_set = MyActiveLearningDatasetBilstm(self.train_dataset)
        active_set.can_label = False
        active_set.label_randomly(INIT_QUERY_SIZE)

        # setup model
        PATCH =  config.all_bayesian or (query_strategy == BaalQueryStrategy.BATCH_BALD)
        if not PATCH: 
            config.iterations = 1
        model = self._init_model(PATCH)
        model = model.to(self.device)
        criterion = CrossEntropyLoss(self.compute_class_weights(active_set.labels))
        optimizer = self._init_optimizer(model)

        baal_model = MyModelWrapperBilstm(
            model,
            criterion,
            replicate_in_memory=False,
            min_train_passes=config.min_train_passes,
        )
        baal_model = self.set_al_metrics(baal_model)

        # active loop
        active_loop = MyActiveLearningLoop(
            dataset=active_set,
            get_probabilities=baal_model.predict_on_dataset,
            heuristic=f_query_strategy,
            query_size=QUERY_SIZE,
            batch_size=config.batch_size,
            iterations=config.iterations,
            use_cuda=self.use_cuda,
            verbose=False,
            workers=2,
            collate_fn=custom_collate,
        )

        # We will reset the weights at each active learning step so we make a copy.
        init_weights = deepcopy(baal_model.state_dict())

        if logging:
            run["model"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["bayesian"] = config.all_bayesian or (
                query_strategy == BaalQueryStrategy.BATCH_BALD
            )
            run["strategy"] = query_strategy.value
            run["config"] = config.__dict__
            run["annotation/intance_ann"].append(active_set.n_labelled / self.n_instances)
            run["annotation/token_ann"].append(
                active_set.n_labelled_tokens / self.n_tokens
            )
            run["annotation/char_ann"].append(
                active_set.n_labelled_chars / self.n_characters
            )

        step_acc = []

        # Active learning loop
        for step in tqdm(range(AL_STEPS)):
            init_step_time = time.time()

            # Load the initial weights.
            baal_model.load_state_dict(init_weights)

            # Train the model on the currently labelled dataset.
            init_train_time = time.time()
            _ = baal_model.train_on_dataset(
                dataset=active_set,
                optimizer=optimizer,
                batch_size=config.batch_size,
                use_cuda=self.use_cuda,
                epoch=config.max_epoch,
                collate_fn=custom_collate,
            )
            train_time = time.time() - init_train_time

            # test the model on the test set.
            baal_model.test_on_dataset(
                dataset=self.test_dataset,
                batch_size=config.batch_size,
                use_cuda=self.use_cuda,
                average_predictions=config.iterations,
                collate_fn=custom_collate,
            )

            if verbose:
                self.print_al_iteration_metrics(step + 1, baal_model.get_metrics())

            # query new instances to be labelled
            init_query_time = time.time()
            should_continue = active_loop.step()
            query_time = time.time() - init_query_time
            step_time = time.time() - init_step_time

            if logging:
                run["times/step_time"].append(step_time)
                run["times/train_time"].append(train_time)
                run["times/query_time"].append(query_time)
                run["annotation/intance_ann"].append(
                    active_set.n_labelled / self.n_instances
                )
                run["annotation/token_ann"].append(
                    active_set.n_labelled_tokens / self.n_tokens
                )
                run["annotation/char_ann"].append(
                    active_set.n_labelled_chars / self.n_characters
                )

            if not should_continue:
                break

            # adjust class weights
            baal_model.criterion = CrossEntropyLoss(
                self.compute_class_weights(active_set.labels)
            )
        # end of active learning loop

        if logging:
            for metrics in baal_model.active_learning_metrics.values():
                for key, value in metrics.items():
                    f_key = key.replace("test_", "test/").replace("train_", "train/")

                    if "class" in key:
                        for i, class_value in enumerate(value):
                            run[f_key + "_" + str(i)].append(class_value)
                    else:
                        run[f_key].append(value)

            run["train/step_acc"].extend(active_loop.step_acc)
            run["train/step_score"].extend(active_loop.step_score)

            run.stop()
