# Base Dependencies
# -----------------
import numpy as np
import time
from typing import Optional
from pathlib import Path
from os.path import join
from joblib import dump, load

# Package Dependencies
# --------------------
from .base import BaseTrainer
from .config import ALExperimentConfig, PLExperimentConfig
from .utils import compute_metrics, random_sampling

# Local Dependencies
# -------------------
from features import RandomForestFeaturesNegation
from models.relation_collection import RelationCollection
from ml_models.rf import (
    RandomForestClassifierOneStage,
)

# 3rd-Party Dependencies
# --------------------
import neptune
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.batch import uncertainty_batch_sampling

# Constants
# ---------
from constants import RFQueryStrategy
from config import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT


# Auxiliar Functions
# ------------------
def _get_query_strategy(q: RFQueryStrategy):
    if q == RFQueryStrategy.RANDOM:
        return random_sampling
    elif q == RFQueryStrategy.LC:
        return uncertainty_sampling
    elif q == RFQueryStrategy.BATCH_LC:
        return uncertainty_batch_sampling
    else:
        raise ValueError("Query strategy not supported")


# Trainer Class
# -------------
class RandomForestTrainer(BaseTrainer):
    """RandomForestTrainer

    Trainer for the Random Forest model
    """

    def __init__(
        self,
        dataset: str,
        train_dataset: RelationCollection,
        test_dataset: RelationCollection,
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
        super().__init__(dataset, train_dataset, test_dataset, relation_type)

        # feature encoder
        self.f_encoder = RandomForestFeaturesNegation(self.dataset)

    def _init_model(self):
        return RandomForestClassifierOneStage(self.dataset)

    @property
    def method_name(self) -> str:
        return "rf"

    @property
    def method_name_pretty(self) -> str:
        return "Random Forest"

    def train_passive_learning(
        self, config: PLExperimentConfig, logging: bool = True, save_model: bool = False
    ) -> RandomForestClassifierOneStage:
        """Trains the RF model using passive learning

        Args:
            logging (bool, optional): determines if logging should be done. Defaults to True.
            save_model (bool, optional): determines if the model should be saved. Defaults to False.

        Returns:
            RandomForestClassifierOneStage: trained model
        """
        if logging:
            # Connect to Neptune and create a run
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        # print info
        self.print_info_passive_learning()

        # init model
        model = self._init_model()

        # fit model
        X_train: np.array = self.f_encoder.fit_transform(self.train_dataset)
        y_train: np.array = self.train_dataset.labels
        model = model.fit(X_train, y_train)

        # predict
        X_test: np.array = self.f_encoder.transform(self.test_dataset)
        y_test: np.array = self.test_dataset.labels
        y_pred_train = model.predict(X_train)
        y_pred = model.predict(X_test)

        # compute metrics
        train_metrics = self.compute_metrics(y_true=y_train, y_pred=y_pred_train)
        test_metrics = self.compute_metrics(y_true=y_test, y_pred=y_pred)

        self.print_train_metrics(train_metrics)
        self.print_test_metrics(test_metrics)

        # save model
        if save_model:
            dump(model, Path(join(self.pl_checkpoint_path, "model.joblib")))

        if logging:
            run["method"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["strategy"] = "passive learning"

            for key, value in train_metrics.items():
                run["train/" + key] = value

            for key, value in test_metrics.items():
                run["test/" + key] = value

            run["model/parameters"] = model.get_params()
            # run["model/file"].upload(Path(join(self.pl_checkpoint_path, "model.joblib")))
            run.stop()

        return model

    def train_active_learning(
        self,
        query_strategy: RFQueryStrategy,
        config: ALExperimentConfig,
        save_models: bool = False,
        verbose: bool = True,
        logging: bool = True,
    ):
        """Trains the RF model using passive learning

        Args:
            query_strategy (str): strategy used to query the most informative instances.
            config (ALExperimentConfig): configuration of the AL experiment.
            logging (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: if `query_strategy` not supported
        """

        if logging:
            run = neptune.init_run(project=NEPTUNE_PROJECT, api_token=NEPTUNE_API_TOKEN)

        # setup
        f_query_strategy = _get_query_strategy(query_strategy)
        INIT_QUERY_SIZE = self.compute_init_q_size(config)
        QUERY_SIZE = self.compute_q_size(config)
        AL_STEPS = self.compute_al_steps(config)

        if verbose:
            self.print_info_active_learning(
                q_strategy=query_strategy.value,
                pool_size=self.n_instances,
                init_q_size=INIT_QUERY_SIZE,
                q_size=QUERY_SIZE,
            )

        # Isolate training examples for labelled dataset
        init_query_indices = np.random.randint(
            low=0, high=self.n_instances, size=INIT_QUERY_SIZE
        )
        active_collection = self.train_dataset[init_query_indices]
        X_active = self.f_encoder.fit_transform(active_collection)
        y_active = active_collection.labels

        # Isolate the non-training examples we'll be querying.
        pool_indices = np.delete(
            np.array(range(self.n_instances)), init_query_indices, axis=0
        )
        pool_collection = self.train_dataset[pool_indices]
        X_pool = self.f_encoder.transform(pool_collection)
        Y_pool = pool_collection.labels

        # Specify the core estimator along with it's active learning model
        learner = ActiveLearner(
            estimator=self._init_model(),
            X_training=X_active,
            y_training=y_active,
            query_strategy=f_query_strategy,
        )

        if save_models:
            dump(
                {
                    "model": learner.estimator,
                    "f_encoder": self.f_encoder,
                    "X_active": X_active,
                    "y_active": y_active,
                },
                Path(join(self.al_checkpoint_path, "model_init.joblib")),
            )

        # evaluate init model
        X_test = self.f_encoder.transform(self.test_dataset)
        y_test = self.test_dataset.labels
        y_pred = learner.predict(X_test)

        init_metrics = compute_metrics(
            y_true=y_test, y_pred=y_pred, average=self.metrics_average
        )

        if verbose:
            self.print_al_iteration_metrics(step=0, metrics=init_metrics)

        if logging:
            run["method"] = self.method_name
            run["dataset"] = self.dataset
            run["relation"] = self.relation_type
            run["strategy"] = query_strategy.value
            for k, v in init_metrics.items():
                run["test/" + k].append(v)

            run["annotation/instance_ann"].append(
                active_collection.n_instances / self.n_instances
            )
            run["annotation/token_ann"].append(active_collection.n_tokens / self.n_tokens)
            run["annotation/char_ann"].append(
                active_collection.n_characters / self.n_characters
            )

        # Active Learning Loop
        for index in range(AL_STEPS):
            init_step_time = time.time()

            # query most informative examples
            init_query_time = time.time()
            n_instances = min(QUERY_SIZE, X_pool.shape[0])
            query_index, _ = learner.query(X_pool, n_instances=n_instances)
            X_query = X_pool[query_index]
            y_query = Y_pool[query_index]
            query_time = time.time() - init_query_time

            # compute accuracy on query
            y_query_pred = learner.predict(X_query)
            step_acc = self.compute_step_accuracy(y_true=y_query, y_pred=y_query_pred)

            # compute average prediction score for true label on query
            scores = []
            query_probs = learner.estimator.predict_proba(X_pool[query_index])
            for i in range(len(y_query)):
                try:
                    scores.append(query_probs[i][y_query[i]])
                except IndexError:
                    scores.append(0.0)
            step_score = np.mean(scores)

            # move queried instances from pool to training
            active_collection = active_collection + pool_collection[query_index]

            # train model on new training data
            init_train_time = time.time()
            X_active = self.f_encoder.fit_transform(active_collection)
            y_active = active_collection.labels
            learner.fit(X=X_active, y=y_active)
            train_time = time.time() - init_train_time

            if save_models:
                dump(
                    {
                        "model": learner.estimator,
                        "f_encoder": self.f_encoder,
                        "X_active": X_active,
                        "y_active": y_active,
                    },
                    Path(
                        join(self.al_checkpoint_path, "model_{}.joblib".format(index))
                    ),
                )

            # remove the queried instance from the unlabeled pool.
            pool_indices = np.delete(
                np.array(range(len(pool_collection))), query_index, axis=0
            )
            if len(pool_indices) == 0:
                break
            pool_collection = pool_collection[pool_indices]
            X_pool = self.f_encoder.transform(pool_collection)

            # calculate and report our model's precision, recall and f1-score.
            X_test = self.f_encoder.transform(self.test_dataset)
            y_pred = learner.predict(X_test)

            # compute metrics
            step_metrics = self.compute_metrics(y_true=y_test, y_pred=y_pred)

            step_time = time.time() - init_step_time

            if verbose:
                self.print_al_iteration_metrics(step=index + 1, metrics=step_metrics)

            if logging:
                run["model/parameters"].append(learner.estimator.get_params())

                for key, value in step_metrics.items():
                    run["test/" + key].append(value)

                run["times/step_time"].append(step_time)
                run["times/train_time"].append(train_time)
                run["times/query_time"].append(query_time)

                run["train/step_acc"].append(step_acc)
                run["train/step_score"].append(step_score)

                run["annotation/instance_ann"].append(
                    active_collection.n_instances / self.n_instances
                )
                run["annotation/token_ann"].append(
                    active_collection.n_tokens / self.n_tokens
                )
                run["annotation/char_ann"].append(
                    active_collection.n_characters / self.n_characters
                )

        # end of active learning loop

        if logging:
            run.stop()
