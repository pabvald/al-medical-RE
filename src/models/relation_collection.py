# coding: utf-8

# Base Dependencies
# -------------------
import numpy
import numpy.typing as npt

from os.path import join as pjoin
from tqdm import tqdm
from typing import List, Dict, Union, Optional

# Package Dependencies
# --------------------
from .entity import Entity
from .relation import Relation, RelationDDI, RelationN2C2

# Local Dependencies
# ------------------
from nlp_pipeline import get_pipeline

# 3r-Party Dependencies
# ---------------------
from datadings.writer import FileWriter
from datadings.reader import MsgpackReader
from spacy.language import Language
from spacy.tokens import Doc

# Constants
# -------------------
from constants import DATASETS_PATHS


# RelationCollection class
# ------------------------
class RelationCollection:
    """RelationCollection"""

    NLP: Language = None  # Spacy's pipeline

    def __init__(self, relations: Union[Relation, List[Relation]]):
        """
        Args:
            relations (Union[Relation, List[Relation]]): relations that form the colction

        Raises:
            ValueError: if an empty list is provided. A collection must contain at least one relation
        """
        if relations is None or len(relations) == 0:
            raise ValueError("collection must contain at least one relation")
        if isinstance(relations, Relation):
            relations = [relations]
        self.relations = numpy.array(relations, dtype="object")
        self._tokens = None
        self._left_tokens = None
        self._middle_tokens = None
        self._right_tokens = None
        self._entities1_tokens = None
        self._entities2_tokens = None

    def __len__(self) -> int:
        """Length of the collection"""
        return len(self.relations)

    def __getitem__(self, indexes: Union[int, List[int]]) -> "RelationCollection":
        """Get a subcollection of the collection

        Args:
            indexes (Union[int, List[int]]): indexes of the relations to be included in the subcollection

        Returns:
            RelationCollection: subcollection of the collection
        """
        subcollection = RelationCollection(self.relations[indexes])

        if self._tokens is not None:
            subcollection._tokens = self._tokens[indexes]
        if self._left_tokens is not None:
            subcollection._left_tokens = self._left_tokens[indexes]
        if self._entities1_tokens is not None:
            subcollection._entities1_tokens = self._entities1_tokens[indexes]
        if self._middle_tokens is not None:
            subcollection._middle_tokens = self._middle_tokens[indexes]
        if self._entities2_tokens is not None:
            subcollection._entities2_tokens = self._entities2_tokens[indexes]
        if self._right_tokens is not None:
            subcollection._right_tokens = self._right_tokens[indexes]

        return subcollection

    def __add__(self, other: "RelationCollection") -> "RelationCollection":
        """Add two RelationCollection objects

        Args:
            other (RelationCollection): other RelationCollection object

        Returns:
            RelationCollection: new RelationCollection object with the relations of both objects
        """

        add_collection = RelationCollection(
            numpy.concatenate([self.relations, other.relations])
        )
        if (self._tokens is not None) or (other._tokens is not None):
            if self._tokens is None:
                self.compute_tokens()
            if other._tokens is None:
                other.compute_tokens()
            add_collection._tokens = numpy.concatenate([self._tokens, other._tokens])

        if (self._left_tokens is not None) or (other._left_tokens is not None):
            if self._left_tokens is None:
                self.compute_left_tokens()
            if other._left_tokens is None:
                other.compute_left_tokens()
            add_collection._left_tokens = numpy.concatenate(
                [self._left_tokens, other._left_tokens]
            )

        if (self._middle_tokens is not None) or (other._middle_tokens is not None):
            if self._middle_tokens is None:
                self.compute_middle_tokens()
            if other._middle_tokens is None:
                other.compute_middle_tokens()
            add_collection._middle_tokens = numpy.concatenate(
                [self._middle_tokens, other._middle_tokens]
            )

        if (self._right_tokens is not None) or (other._right_tokens is not None):
            if self._right_tokens is None:
                self.compute_right_tokens()
            if other._right_tokens is None:
                other.compute_right_tokens()
            add_collection._right_tokens = numpy.concatenate(
                [self._right_tokens, other._right_tokens]
            )

        if (self._entities1_tokens is not None) or (
            other._entities1_tokens is not None
        ):
            if self._entities1_tokens is None:
                self.compute_entities1_tokens()
            if other._entities1_tokens is None:
                other.compute_entities1_tokens()
            add_collection._entities1_tokens = numpy.concatenate(
                [self._entities1_tokens, other._entities1_tokens]
            )

        if (self._entities2_tokens is not None) or (
            other._entities2_tokens is not None
        ):
            if self._entities2_tokens is None:
                self.compute_entities2_tokens()
            if other._entities2_tokens is None:
                other.compute_entities2_tokens()
            add_collection._entities2_tokens = numpy.concatenate(
                [self._entities2_tokens, other._entities2_tokens]
            )

        return add_collection

    # Properties
    # ----------
    @property
    def ids(self):
        """Ids of the relations"""
        return numpy.array(list(map(lambda r: r.id, self.relations)))

    @property
    def labels(self):
        """Labels of the relations"""
        return numpy.array(list(map(lambda r: r.label, self.relations)))

    @property
    def tokens(self):
        """Tokens of the relations"""
        if self._tokens is None:
            self.compute_tokens()
        return self._tokens

    @property
    def left_tokens(self):
        """Tokens of the left context of the relations"""
        if self._left_tokens is None:
            self.compute_left_tokens()
        return self._left_tokens

    @property
    def middle_tokens(self):
        """Tokens of the middle context of the relations"""
        if self._middle_tokens is None:
            self.compute_middle_tokens()
        return self._middle_tokens

    @property
    def right_tokens(self):
        """Tokens of the right context of the relations"""
        if self._right_tokens is None:
            self.compute_right_tokens()
        return self._right_tokens

    @property
    def entities1_tokens(self):
        """Tokens of the first target entity of the relations"""
        if self._entities1_tokens is None:
            self.compute_entities1_tokens()
        return self._entities1_tokens

    @property
    def entities2_tokens(self):
        """Tokens of the second target entiy of the relations"""
        if self._entities2_tokens is None:
            self.compute_entities2_tokens()
        return self._entities2_tokens

    @property
    def n_instances(self) -> int:
        """Number of relations contained in the collection"""
        return len(self.relations)

    @property
    def n_tokens(self) -> int:
        """Number of tokens contained in the collection"""
        n = 0
        for doc in self.tokens:
            n += len(doc)
        return n

    @property
    def n_characters(self) -> int:
        """Number of characters (including spaces) contained in the collection"""
        n = 0
        for r in self.relations:
            n += len(r.text)
        return n

    # Class Methods
    # -------------
    @classmethod
    def from_datading(cls, dataset: str, filepath: str) -> "RelationCollection":
        """Creates a RelationCollection from a datading

        Args:
            dataset (str): name of the dataset
            filepath (str): path to the datading

        Returns:
            RelationCollection: the RelationCollection
        """
        if dataset == "n2c2":
            RelationClass = RelationN2C2
        elif dataset == "ddi":
            RelationClass = RelationDDI
        else:
            raise ValueError("unsupported dataset '{}'")

        relations = []
        with MsgpackReader(filepath) as reader:
            for sample in tqdm(reader):
                rel = sample["relation"]
                rel["entity1"] = Entity(**rel["entity1"])
                rel["entity2"] = Entity(**rel["entity2"])
                rel["middle_entities"] = [
                    Entity(**ent) for ent in rel["middle_entities"]
                ]
                relations.append(RelationClass(**rel))

        return cls(relations)

    @classmethod
    def set_nlp(cls, nlp: Language):
        """Sets the Entity Class' Spacy's pipeline

        Args:
            nlp (Language): NLP pipeline
        """
        cls.NLP = nlp

    @classmethod
    def tokenize_texts(cls, texts: List[str], disable: List[str] = ["parser", "negex"]):
        """Tokenizes a text fragment with the configured Spacy's pipeline

        Args:
            text (str): text fragment
            disable (List[str], optional): pipes of the Spacy's pipeline to be disabled. Defaults to ["parser"].

        Returns:
            Doc: tokenized text
        """
        if cls.NLP is None:
            cls.NLP = get_pipeline()

        with cls.NLP.select_pipes(disable=disable):
            docs = list(cls.NLP.pipe(texts, batch_size=32))

        if len(docs) > 1:
            docs = numpy.array(docs, dtype="object")
        return docs

    # Instance methods
    # ----------------
    def valid_indexes(self) -> List[int]:
        """Computes the indixes of the relations that are valid, i.e., whose tokenization
        is the same whether it is done on the whole text or by parts

        Returns:
            List[int]: indexes of valid relations
        """
        self.compute_all()

        indexes = []
        for i in range(self.__len__()):
            if len(self.tokens[i]) == (
                len(self.left_tokens[i])
                + len(self.entities1_tokens[i])
                + len(self.middle_tokens[i])
                + len(self.entities2_tokens[i])
                + len(self.right_tokens[i])
            ):
                indexes.append(i)

        return numpy.array(indexes)

    def compute_all(self) -> None:
        """Computes all tokens"""
        self.compute_tokens()
        self.compute_left_tokens()
        self.compute_entities1_tokens()
        self.compute_middle_tokens()
        self.compute_entities2_tokens()
        self.compute_right_tokens()

    def compute_tokens(self) -> None:
        """Compute the tokens of the relations' text"""
        self._tokens = self.tokenize_texts(list(map(lambda r: r.text, self.relations)))

    def compute_left_tokens(self) -> None:
        """Compute tokens of the relations' left context"""
        self._left_tokens = self.tokenize_texts(
            list(map(lambda r: r.left_context.strip(), self.relations))
        )

    def compute_middle_tokens(self) -> None:
        """Compute tokens of the relations' middle context"""
        self._middle_tokens = self.tokenize_texts(
            list(map(lambda r: r.middle_context.strip(), self.relations))
        )

    def compute_right_tokens(self) -> None:
        """Compute tokens of the relations' right context"""
        self._right_tokens = self.tokenize_texts(
            list(map(lambda r: r.right_context.strip(), self.relations))
        )

    def compute_entities1_tokens(self) -> None:
        """Compute tokens of the relations' entities"""
        self._entities1_tokens = self.tokenize_texts(
            list(map(lambda r: r.entity1.text.strip(), self.relations))
        )

    def compute_entities2_tokens(self) -> None:
        """Compute tokens of the relations' entities"""
        self._entities2_tokens = self.tokenize_texts(
            list(map(lambda r: r.entity2.text.strip(), self.relations))
        )

    def type_subcollection(self, rtype: str) -> "RelationCollection":
        """Obtains a subcollection containing relations of a type

        Args:
            rtype (str): type of the relations to be included in the subcollection

        Returns:
            RelationCollection: subcollection
        """
        indexes = []
        for idx, r in enumerate(self.relations):
            if r.type == rtype:
                indexes.append(idx)

        return self.__getitem__(indexes)

    def generate_samples(self) -> Dict:
        """Generates datading samples from relations"""
        for rel in self.relations:
            yield {"key": rel.id, "relation": rel.todict()}

    def to_datading(self, filepath: str, overwrite: bool = True):
        """Stores a RelationCollection in a datading

        Args:
            filepath (str): path to the datading
            overwrite (bool, optional): whether to overwrite the datading if it already exists. Defaults to True.
        """
        with FileWriter(filepath, overwrite=overwrite) as writer:
            for sample in self.generate_samples():
                writer.write(sample)

    # Static Methods
    # --------------
    @staticmethod
    def load_collections(
        dataset: str, splits: List[str] = ["train", "test"]
    ) -> Dict[str, "RelationCollection"]:
        """Loads the relation collections of a given dataset

        Args:
            dataset (str): name of the dataset
            splits (List[str]): splits to be loaded. Defaults ot `["train", "test"]`.
        """

        print("Loading {} dataset...".format(dataset))
        dataset_path = DATASETS_PATHS[dataset]

        collections = {}

        for split in splits:
            path = pjoin(dataset_path, "{}_datading".format(split), "relations.msgpack")
            collections[split] = RelationCollection.from_datading(dataset, path)

        return collections
