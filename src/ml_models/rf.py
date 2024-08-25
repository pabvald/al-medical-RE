# Base Dependencies
# ----------------
import numpy as np

from typing import Any

# Local Dependencies
# ------------------
from features import RandomForestFeatures
from utils import ddi_binary_relation

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Constants
# ---------
RF_HYPERPARAM_GRID = {
    "bootstrap": [True, False],
    "max_depth": [2, 5, 10, 20, 30, 40, 50],
    "max_features": ["sqrt", "log2", None],
    "min_samples_leaf": [2, 3, 4],
    "min_samples_split": [2, 5, 10],
}

RF_BINARY_THRESHOLD_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9]


# ML Models
# ---------
class RandomForestClassifierOneStage(BaseEstimator):
    """Random Forest Classifier One Stage

    Random Forest classifier that can be used for both N2C2 and DDI datasets.
    It consideres a single stage of classification. For the n2c2 corpus, it
    is used to classify between positive and negative relations. For the N2C2
    corpus, it is used to classify between the 5 relation types, including
    the NO-REL type.
    """

    def __init__(self, dataset: str) -> None:
        """Initializes the model

        Args:
            dataset (str): dataset's name
        """
        super().__init__()
        self.dataset = dataset
        self.clf = RandomForestClassifier(class_weight="balanced")

    @property
    def scoring(self) -> str:
        """Scoring metric to use for hyperparameter tuning"""
        if self.dataset == "n2c2":
            return "f1"
        else:
            return "f1_micro"

    def score(self, X: np.array, Y: np.array, sample_weight=None) -> float:
        """Scores the model

        Args:
            X (np.array): Feature matrix
            Y (np.array): Label vector

        Returns:
            float: Score
        """
        from sklearn.metrics import f1_score
        return f1_score(Y, self.predict(X), sample_weight=sample_weight)
    
    def fit(self, X: np.array, Y: np.array):
        """Fits the model. It uses 5-fold cross validation to find the best hyperparameters.

        Args:
            X (np.array): Feature matrix
            Y (np.array): Label vector

        Returns:
            RandomForestClassifierOneStage: Fitted model
        """
        assert X.shape[0] == len(Y)
        assert len(X.shape) == 2

        search = RandomizedSearchCV(self.clf, RF_HYPERPARAM_GRID, scoring=self.scoring)
        search = search.fit(X, Y)
        self.clf = search.best_estimator_

        return self

    def predict(self, X: np.array):
        """Predicts the class of a given sample

        Args:
            X (np.array): Feature matrix

        Returns:
            np.array: Predicted class
        """
        return self.clf.predict(X)

    def predict_log_probab(self, X: np.array):
        """Predicts the log probability of a given sample

        Args:
            X (np.array): Feature matrix

        Returns:
            np.array: Predicted log probability
        """
        return self.clf.predict_log_proba(X)

    def predict_proba(self, X: np.array):
        """Predicts the probability of a given sample

        Args:
            X (np.array): Feature matrix

        Returns:
            np.array: Predicted probability
        """
        return self.clf.predict_proba(X)


class RandomForestClassifierTwoStage(BaseEstimator):
    """Random Forest Classifier Two Stage 

    Random Forest Classifier that can be used for the DDI dataset. It considers
    a two stage classification. The first stage is used to classify between posivite
    and negative relations. The second stage is used to classify between the 4
    relation types.
    """

    class BalancedRandomForestClassifierBinary(BaseEstimator):
        """Balanced Random Forest Classifier Binary

        Random Forest Classifier used in the first stage of the two stage classification.
        It classifies between positive and negative relations.

        """

        def __init__(
            self,
            dataset: str,
            threshold: float = 0.7,
        ) -> None:
            """Initializes the model

            Args:
                dataset (str): dataset's name
                threshold (float, optional): classification threshold to classify instances
                    as positive. Defaults to 0.7.
            """
            super().__init__()
            self.dataset = dataset
            self.threshold = threshold
            self.clf = BalancedRandomForestClassifier(class_weight="balanced")

        def make_binary(self, y: Any) -> np.array:
            
            if self.dataset == "ddi":
                return ddi_binary_relation(y)
            else:
                raise NotImplementedError
            
        def fit(self, X: np.array, Y: np.array):
            """Fits the model.

            Args:
                X (np.array): Feature matrix
                Y (np.array): Label vector

            Returns:
                BalancedRandomForestClassifierBinary: Fitted model
            """
            assert X.shape[0] == len(Y)
            assert len(X.shape) == 2

            # fit binary random forest
            self.clf = self.clf.fit(X, Y)

            return self

        def predict(self, X: np.array):
            """Predicts the class of a given sample

            Args:
                X (np.array): Feature matrix

            Returns:
                np.array: Predicted class
            """
            Y = (self.clf.predict_proba(X)[:, 1] >= self.threshold).astype(bool)
            return Y

    def __init__(self) -> None:
        super().__init__()
        # 1st classifier - Detect relations - classify between positive and negative
        self.clf1 = self.BalancedRandomForestClassifierBinary()

        # 2nd classifier - Classifiy relation - classify positive relations into a relation type
        self.clf2 = BalancedRandomForestClassifier()

    def fit(self, X: np.array, Y: np.array):
        """Fits the model.

        Args:
            X (np.array): Feature matrix
            Y (np.array): Label vector

        Returns:
            RandomForestClassifierTwoStage: Fitted model
        """
        assert X.shape[0] == len(Y)
        assert len(X.shape) == 2
        Y_1 = np.array(list(map(lambda y: self.make_binary(y), Y)))

        # fit 1st classifier
        search1 = GridSearchCV(
            estimator=self.clf1,
            param_grid={"threshold": RF_BINARY_THRESHOLD_GRID},
            scoring="f1",
        )
        search1 = search1.fit(X, Y_1)
        self.clf1 = search1.best_estimator_

        # fit 2nd classifier
        search2 = RandomizedSearchCV(
            estimator=self.clf2,
            param_distributions=RF_HYPERPARAM_GRID,
            scoring="f1_micro",
        )
        search2 = search2.fit(X[Y > 0, :], Y[Y > 0])
        self.clf2 = search2.best_estimator_

        return self

    def predict(self, X: np.array):
        """Predicts the class of a given sample

        Args:
            X (np.array): Feature matrix

        Returns:
            np.array: Predicted class
        """
        Y = self.clf1.predict(X)
        Y = np.array(Y, dtype=np.int8)
        Y[Y > 0] = self.clf2.predict(X[Y > 0])
        return Y


# ML Pipelines
# ------------
# RandomForestPipelineN2C2 = Pipeline(
#     [
#         ("encoder", RandomForestFeatures("n2c2")),
#         ("clf", RandomForestClassifierOneStageN2C2()),
#     ]
# )

# RandomForestPipelineDDI = Pipeline(
#     [
#         ("encoder", RandomForestFeatures("ddi")),
#         ("clf", RandomForestClassifierOneStageDDI()),
#     ]
# )
