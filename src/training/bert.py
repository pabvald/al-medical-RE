# Base Dependencies
# -----------------
import numpy as np
import re
import time
from copy import deepcopy
from functools import partial
from os.path import join
from pathlib import Path
from typing import Optional, Dict

# Package Dependencies
# --------------------
from .base import BaseTrainer
from .config import PLExperimentConfig, BaalExperimentConfig
from .utils import get_baal_query_strategy, tokenize, tokenize_pairs

# Local Dependencies
# ------------------
from extensions.baal import my_active_huggingface_dataset, MyActiveLearningLoop
from extensions.transformers import WeightedLossTrainer
from ml_models.bert import ClinicalBERT, ClinicalBERTTokenizer, ClinicalBERTConfig

# 3rd-Party Dependencies
# ----------------------
import neptune

from baal.transformers_trainer_wrapper import BaalTransformersTrainer
from baal.bayesian.dropout import patch_module
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    TrainingArguments,
)

# Constants
# ---------
from constants import BaalQueryStrategy
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT


class BertTrainer(BaseTrainer):
    """Trainer for the BERT method"""

    def __init__(
        self,
        dataset: str,
        train_dataset: Dataset,
        test_dataset: Dataset,
        pairs: bool = False,
        relation_type: Optional[str] = None,
    ):
        """
        dataset (str): name of the dataset, e.g., "n2c2".
        train_dataset (Dataset): train split of the dataset.
        test_dataset (Dataset): test split of the dataset.
        relation_type (str, optional): relation type.

        Raises:
            ValueError: if the name dataset provided is not supported
        """
        super().__init__(dataset, train_dataset, test_dataset, relation_type)

        self.pairs = pairs
        # tokenizer
        self.tokenizer = ClinicalBERTTokenizer()

        # tokenize datasets
        if not pairs:
            self.train_dataset = tokenize(self.tokenizer, self.train_dataset)
            self.test_dataset = tokenize(self.tokenizer, self.test_dataset)
        else:
            self.train_dataset = tokenize_pairs(self.tokenizer, self.train_dataset)
            self.test_dataset = tokenize_pairs(self.tokenizer, self.test_dataset)

    @property
    def method_name(self) -> str:
        if self.pairs:
            name = "bert-pairs"
        else:
            name = "bert"
        return name

    @property
    def method_name_pretty(self) -> str:
        if self.pairs:
            name = "Paired Clinical BERT"
        else:
            name = "Clinical BERT"
        return name

    def _init_model(self, patch: bool = False) -> ClinicalBERT:
        config = ClinicalBERTConfig
        config.num_labels = self.num_classes
        model = ClinicalBERT(config=ClinicalBERTConfig)
        if patch:
            model = patch_module(model)
        return model

    def compute_metrics_transformer(self, eval_preds: EvalPrediction) -> Dict:
        """Computes metrics from a Transformer's prediction.

        Args:
            eval_preds (EvalPrediction): transformer's prediction

        Returns:
            Dict: precision, recall and F1-score
        """
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return self.compute_metrics(y_true=labels, y_pred=predictions)

    def train_passive_learning(
        self, config: PLExperimentConfig, verbose: bool = True, logging: bool = True
    ):
        """Trains the BiLSTM model using passive learning and early stopping

        Args:
            config (PLExperimentConfig): cofiguration
            verbose (bool): determines if information is printed during training. Daults to True.
            logging (bool): log the test metrics on Neptune. Defaults to True.
        """
        if logging:
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        # setup
        train_val_split = self.train_dataset.train_test_split(
            test_size=config.val_size, stratify_by_column="label"
        )
        train_set = train_val_split["train"]
        val_set = train_val_split["test"]
        test_set = self.test_dataset

        model = self._init_model()

        training_args = TrainingArguments(
            output_dir=self.pl_checkpoint_path,  # output directory
            optim="adamw_torch",  # optimizer
            weight_decay=0.01,  # strength of weight decay
            learning_rate=5e-5,  # learning rate
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            num_train_epochs=config.max_epoch,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,  # batch size for evaluation
            log_level="warning",  # logging level
            logging_dir=".logs/n2c2/bert/",  # directory for storing logs
            report_to="none",
            metric_for_best_model="f1",
            load_best_model_at_end=True,
        )

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            seed=config.seed,
            train_dataset=train_set,
            eval_dataset=val_set,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_transformer,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.es_patience)
            ],
        )
        labels = train_set["label"].numpy()
        trainer.class_weights = self.compute_class_weights(labels)

        # print info
        if verbose:
            self.print_info_passive_learning()

        # train model
        trainer.train()
        eval_loss_values = trainer.eval_loss
        train_loss_values = trainer.training_loss

        # evaluate model on test set
        test_metrics = trainer.evaluate(test_set)

        if verbose:
            self.print_test_metrics(test_metrics)

        # log to Neptune
        if logging:
            run["method"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["strategy"] = "passive learning"
            run["config"] = config.__dict__
            run["epoch"] = len(eval_loss_values)

            for loss in train_loss_values:
                run["loss/train"].append(loss)

            for loss in eval_loss_values:
                run["loss/val"].append(loss)

            for key, value in test_metrics.items():
                key2 = re.sub(r"eval_", "", key)
                run["test/" + key2] = value

            run.stop()

        return model

    def train_active_learning(
        self,
        query_strategy: BaalQueryStrategy,
        config: BaalExperimentConfig,
        verbose: bool = True,
        save_models: bool = False,
        logging: bool = True,
    ):
        """Trains the BiLSTM model using active learning

        Args:
            query_strategy (str): name of the query strategy to be used in the experiment.
            config (BaalExperimentConfig): experiment configuration.
            verbose (bool): determines if information is printed during trainig or not. Defaults to True.s
            logging (bool): log the test metrics on Neptune. Defaults to True.
        """

        if logging:
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)
            run["model"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["strategy"] = query_strategy.value
            run["bayesian"] = config.all_bayesian or (
                query_strategy == BaalQueryStrategy.BATCH_BALD
            )
            run["params"] = config.__dict__

        # setup quering 
        INIT_QUERY_SIZE = self.compute_init_q_size(config)
        QUERY_SIZE = self.compute_q_size(config)
        AL_STEPS = self.compute_al_steps(config)

        f_query_strategy = get_baal_query_strategy(
            name=query_strategy.value,
            shuffle_prop=config.shuffle_prop,
            query_size=QUERY_SIZE,
        )

        # setup model
        PATCH = config.all_bayesian or (query_strategy == BaalQueryStrategy.BATCH_BALD)
        if not PATCH:
            config.iterations = 1     

        # setup active set
        active_set = my_active_huggingface_dataset(self.train_dataset)
        active_set.can_label = False
        active_set.label_randomly(INIT_QUERY_SIZE)

        # print info
        if verbose:
            self.print_info_active_learning(
                q_strategy=query_strategy.value,
                pool_size=self.n_instances,
                init_q_size=INIT_QUERY_SIZE,
                q_size=QUERY_SIZE,
            )

        training_args = TrainingArguments(
            output_dir=self.al_checkpoint_path,
            optim="adamw_torch",  # optimizer
            weight_decay=0.01,  # strength of weight decay
            learning_rate=5e-5,  # learning rate
            num_train_epochs=config.max_epoch,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,  # batch size for evaluation
            log_level="warning",  # logging level
            logging_dir=".logs/n2c2/bert/",  # directory for storing logs
            report_to="none",
        )

        # create the trainer through Baal Wrapper
        baal_trainer = BaalTransformersTrainer(
            model_init=partial(self._init_model, PATCH),
            seed=config.seed,
            args=training_args,
            train_dataset=active_set,
            tokenizer=None,
            compute_metrics=self.compute_metrics_transformer,
        )


        # create Active Learning loop
        active_loop = MyActiveLearningLoop(
            dataset=active_set,
            get_probabilities=baal_trainer.predict_on_dataset,
            heuristic=f_query_strategy,
            query_size=QUERY_SIZE,
            iterations=config.iterations,
            max_sample=config.max_sample,
        )

        init_weights = deepcopy(baal_trainer.model.state_dict())

        # Active Learning loop
        for step in range(AL_STEPS):
            init_step_time = time.time()

            # reset the model to the initial state
            baal_trainer.model.load_state_dict(init_weights)

            # train model on current active set
            init_train_time = time.time()
            baal_trainer.train()
            train_time = time.time() - init_train_time

            if save_models:
                # save model
                path = Path(join(self.al_checkpoint_path, "model_{}.ck".format(step)))
                baal_trainer.model.save_pretrained(path)
                
            # evaluate model on test set
            metrics = baal_trainer.evaluate(self.test_dataset)
            metrics["dataset_size"] = active_set.n_labelled

            # print step metrics
            if verbose:
                self.print_al_iteration_metrics(step + 1, metrics)

            # query new instances
            init_query_time = time.time()
            should_continue = active_loop.step()
            query_time = time.time() - init_query_time
            step_time = time.time() - init_step_time

            if logging:
                run["times/step_time"].append(step_time)
                run["times/train_time"].append(train_time)
                run["times/query_time"].append(query_time)
                run["annotation/instance_ann"].append(
                    active_set.n_labelled / self.n_instances
                )
                run["annotation/token_ann"].append(
                    active_set.n_labelled_tokens / self.n_tokens
                )
                run["annotation/char_ann"].append(
                    active_set.n_labelled_chars / self.n_characters
                )
                for key, value in metrics.items():
                    f_key = key.replace("test_", "test/").replace("train_", "train/")
                    run[f_key].append(value)

            if not should_continue:
                break

            # We reset the model weights to relearn from the new train set.
            baal_trainer.load_state_dict(init_weights)
            baal_trainer.lr_scheduler = None

        # log to Neptune
        if logging:
            for step_acc in active_loop.step_acc:
                run["train/step_acc"].append(step_acc)

            for step_score in active_loop.step_score:
                run["train/step_score"].append(step_score)

            run.stop()
