# coding: utf-8

# Base Dependencies
# -----------------
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict
from os.path import join as pjoin

# Local Dependencies
# ------------------
from features import BilstmFeatures
from vocabulary import Vocabulary
from models import RelationCollection

# 3rd-party Dependencies
# ----------------------
from datasets import Dataset as HFDataset
from datasets import ClassLabel, Value, Features, Sequence

# Constants
# ---------
from constants import (
    N2C2_PATH,
    N2C2_REL_TYPES,
    N2C2_VOCAB_PATH,
    DDI_PATH,
    DDI_ALL_TYPES,
    DDI_NO_REL,
    DDI_VOCAB_PATH,
)


class BilstmDatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def create_datasets(
        dataset: str, collections: Dict[str, RelationCollection], vocab: Optional[Vocabulary]
    ): 
        if dataset == "n2c2":
            return BilstmDatasetFactory.create_datasets_n2c2(collections, vocab)
        elif dataset == "ddi": 
            return BilstmDatasetFactory.create_datasets_ddi(collections, vocab)
        else:
            raise ValueError("unsupported dataset '{}'".format(dataset))
        
    @staticmethod
    def create_datasets_n2c2(
        collections: Dict[str, RelationCollection], vocab: Optional[Vocabulary]
    ):
        """Generates the n2c2 datasets for the BiLSTM model

        Args:
            collections (Dict[str, RelationCollection]): collections of the n2c2 corpus
        """
        print("Creating n2c2 dataset for LSTM model...")

        if vocab is None:
            vocab_config = {"voc_file": N2C2_VOCAB_PATH}
            vocab = Vocabulary.build_vocab(vocab_config)

        for split, collection in collections.items():
            print(split, ": ")
            for rel_type in tqdm(N2C2_REL_TYPES):
                # output path
                dataset_path = Path(pjoin(N2C2_PATH, split + ".hf", "bilstm", rel_type))

                # create subcollection
                subcollection = collection.type_subcollection(rel_type)    
                
                # generate features
                features = BilstmFeatures(dataset="n2c2", vocab=vocab).fit_transform(
                    subcollection
                )

                # build dataset
                dataset = HFDataset.from_dict(
                    mapping={
                        # "id": collection.ids,
                        "e1": features["e1"],
                        "e2": features["e2"],
                        "sent": features["sent"],
                        "rd1": features["rd1"],
                        "rd2": features["rd2"],
                        "pos": features["pos"],
                        "dep": features["dep"],
                        "iob": features["iob"],
                        "seq_length": features["seq_length"],
                        "char_length": features["char_length"],
                        "label": subcollection.labels,
                    },
                    features=Features(
                        {
                            # "id": Value(dtype="string", id=None),
                            "e1": Sequence(Value(dtype="int32"), length=-1),
                            "e2": Sequence(Value(dtype="int32"), length=-1),
                            "rd1": Sequence(Value(dtype="int32"), length=-1),
                            "rd2": Sequence(Value(dtype="int32"), length=-1),
                            "sent": Sequence(Value(dtype="int32"), length=-1),
                            "pos": Sequence(Value(dtype="int8"), length=-1),
                            "dep": Sequence(Value(dtype="int8"), length=-1),
                            "iob": Sequence(Value(dtype="int8"), length=-1),
                            "seq_length": Value(dtype="int32"),
                            "char_length": Value(dtype="int32"),
                            "label": ClassLabel(
                                num_classes=2,
                                names=["negative", "positive"],
                                names_file=None,
                                id=None,
                            ),
                        }
                    ),
                )
                dataset = dataset.with_format("torch")

                # store dataset
                dataset.save_to_disk(dataset_path=dataset_path)

    @staticmethod
    def create_datasets_ddi(
        collections: Dict[str, RelationCollection], vocab: Optional[Vocabulary]
    ):
        """Generates the DDI datasets for the BiLSTM model

        Args:
            collections (Dict[str, RelationCollection]): collections of the ddi corpus
        """
        print("Creating DDI dataset for LSTM model...")

        if vocab is None:
            vocab_config = {"voc_file": DDI_VOCAB_PATH}
            vocab = Vocabulary.build_vocab(vocab_config)

        for split, collection in tqdm(collections.items()):
            # output path
            dataset_path = Path(pjoin(DDI_PATH, split + ".hf", "bilstm"))
                
            # generate features
            features = BilstmFeatures(dataset="ddi", vocab=vocab).fit_transform(
                collection
            )

            # build dataset
            dataset = HFDataset.from_dict(
                mapping={
                   # "id": collection.ids,
                    "e1": features["e1"],
                    "e2": features["e2"],
                    "sent": features["sent"],
                    "rd1": features["rd1"],
                    "rd2": features["rd2"],
                    "pos": features["pos"],
                    "dep": features["dep"],
                    "iob": features["iob"],
                    "seq_length": features["seq_length"],
                    "char_length": features["char_length"],
                    "label": collection.labels,
                },
                features=Features(
                    {
                        # "id": Value(dtype="string", id=None),
                       "e1": Sequence(Value(dtype="int32"), length=-1),
                        "e2": Sequence(Value(dtype="int32"), length=-1),
                        "rd1": Sequence(Value(dtype="int32"), length=-1),
                        "rd2": Sequence(Value(dtype="int32"), length=-1),
                        "sent": Sequence(Value(dtype="int32"), length=-1),
                        "pos": Sequence(Value(dtype="int8"), length=-1),
                        "dep": Sequence(Value(dtype="int8"), length=-1),
                        "iob": Sequence(Value(dtype="int8"), length=-1),
                        "seq_length": Value(dtype="int32"),
                        "char_length": Value(dtype="int32"),
                        "label": ClassLabel(
                            num_classes=len(DDI_ALL_TYPES),
                            names=DDI_ALL_TYPES,
                            names_file=None,
                            id=None,
                        ),
                    }
                ),
            )
            dataset = dataset.with_format("torch")

            # store dataset
            dataset.save_to_disk(dataset_path=dataset_path)
