# coding: utf-8

# Base Dependencies
# -----------------
import numpy as np
from tqdm import tqdm
from typing import Dict
from os.path import join as pjoin
from pathlib import Path

# Local Dependencies
# ------------------
from features import BertFeatures
from constants import N2C2_PATH, N2C2_REL_TYPES, DDI_PATH, DDI_ALL_TYPES
from models import RelationCollection
from utils import ddi_binary_relation

# 3rd-party Dependencies
# ----------------------
from datasets import Dataset as HFDataset
from datasets import ClassLabel, Value, Features


class BertDatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_datasets(
        dataset: str, collections: Dict[str, RelationCollection]
    ): 
        if dataset == "n2c2":
            return BertDatasetFactory.create_datasets_n2c2(collections)
        elif dataset == "ddi": 
            return BertDatasetFactory.create_datasets_ddi(collections)
        else:
            raise ValueError("unsupported dataset '{}'".format(dataset))
        
        
    @staticmethod
    def create_datasets_n2c2(collections: Dict[str, RelationCollection]):
        """Generates the n2c2 datasets for the BERT model

        Args:
            collections (Dict[str, RelationCollection]): collections of the n2c2 corpus
        """
        print("Creating n2c2 datasets for BERT model...")

        for split, collection in collections.items():
            print(split, ": ")
            for rel_type in tqdm(N2C2_REL_TYPES):
                # path
                dataset_path = Path(pjoin(N2C2_PATH, split + ".hf", "bert", rel_type))

                # extract subcollection
                subcollection = collection.type_subcollection(rel_type)
                
                # generate features
                features = BertFeatures().fit_transform(subcollection)

                # build dataset
                dataset = HFDataset.from_dict(
                    mapping={
                        "sentence": features["sentence"],
                        "text": features["text"],
                        "char_length": features["char_length"],
                        "seq_length": features["seq_length"],
                        "label": subcollection.labels,
                    },
                    features=Features(
                        {
                            "label": ClassLabel(
                                num_classes=2,
                                names=["negative", "positive"],
                                names_file=None,
                                id=None,
                            ),
                            "sentence": Value(dtype="string", id=None),
                            "text":  Value(dtype="string", id=None),
                            "char_length": Value(dtype="int32"),
                            "seq_length": Value(dtype="int32"),
                        }
                    ),
                )
                dataset = dataset.with_format("torch")

                # store dataset
                dataset.save_to_disk(dataset_path=dataset_path)

    @staticmethod
    def create_datasets_ddi(collections: Dict[str, RelationCollection]):
        """Generates the DDI datasets for the BERT model

        Args:
            collections (Dict[str, RelationCollection]): collections of the DDI corpus
        """
        print("Creating DDI datasets for BERT model...")

        for split, collection in tqdm(collections.items()):
            # path
            dataset_path = Path(pjoin(DDI_PATH, split + ".hf", "bert"))

            # generate features
            features = BertFeatures().fit_transform(collection)

            # build dataset
            dataset = HFDataset.from_dict(
                mapping={
                    "sentence": features["sentence"],
                    "text": features["text"],
                    "char_length": features["char_length"],
                    "seq_length": features["seq_length"],
                    "label": collection.labels,
                    "label2": np.array(
                        list(map(ddi_binary_relation, collection.labels))
                    ),
                },
                features=Features(
                    {
                        "label": ClassLabel(
                            num_classes=len(DDI_ALL_TYPES),
                            names=DDI_ALL_TYPES,
                            names_file=None,
                            id=None,
                        ),
                        "label2": ClassLabel(
                            num_classes=2,
                            names=["negative", "positive"],
                            names_file=None,
                            id=None,
                        ),
                        "sentence": Value(dtype="string", id=None),
                        "text": Value(dtype="string", id=None),
                        "char_length": Value(dtype="int32"),
                        "seq_length": Value(dtype="int32"),
                    }
                ),
            )
            dataset = dataset.with_format("torch")

            # store dataset
            dataset.save_to_disk(dataset_path=dataset_path)
