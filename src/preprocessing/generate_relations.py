# coding: utf-8

# Base Dependencies
# ------------------
import json
import xml.etree.ElementTree as ET

from os.path import join as pjoin
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Set, Dict

# Local Dependencies
# --------------------
from models import Document, Entity, RelationN2C2, RelationDDI, RelationCollection
from utils import files_ddi, files_n2c2, doc_id_n2c2, make_dir

# 3rd-Party Dependencies
# ----------------------
from PyRuSH import PyRuSHSentencizer

# Constants
# ---------
from constants import N2C2_PATH, DDI_PATH


# Auxiliar Functions
# ------------------
def read_txt(file: Path) -> str:
    """Reads a .txt file

    Args:
        file (Path): path to the .txt file
    """
    # read text file
    with open(file, "r", encoding="utf-8") as fin:
        text = fin.read()

    return text


def read_json(file: Path) -> str:
    """Reads a .json file

    Args:
        file (Path): path to the .json file
    """
    return json.loads(read_txt(file))


def read_annotations_n2c2(file: Path) -> Tuple[List[Entity], Set[str]]:
    """Reads a n2c2 .ann file and extracts the entities and the relations

    Args:
        file (Path): path to the n2c2 annotation file
    """

    # read file
    with open(file, "r", encoding="utf-8") as fin:
        annotations: List[str] = fin.readlines()

    # process file
    doc_id: str = doc_id_n2c2(file)
    entities: List[Entity] = list()
    gt_relations: Set[str] = set()  # ground-truth relations

    for line in annotations:
        if line.startswith("T"):  # process entity
            entities.append(Entity.from_n2c2_annotation(doc_id, line))

        elif line.startswith("R"):  # process relation
            id, definition = line.strip().split("\t")
            type, entity1_id, entity2_id = definition.split()
            entity1_id = entity1_id.split(":")[1]
            entity2_id = entity2_id.split(":")[1]
            gt_relations.add("{}-{}".format(entity1_id, entity2_id))

        else:  # ignore annotator's note
            continue

    # sort entities by their end character
    entities.sort(key=lambda ent: ent.end)

    return entities, gt_relations


# Main Functions
# ---------------
def generate_relations(
    dataset: str, save_to_disk: bool = True
) -> Dict[str, RelationCollection]:
    """Generates relations of a given dataset and saves them to disk

    Args:
        dataset (str): dataset's name
        save_to_disk (bool, optional): the relation collections are saved to disk in a datading or not. Defaults to True.

    Raises:
        ValueError: unsupported dataset

    Returns:
        Dict[str, RelationCollection]: train and test relation collections
    """
    if dataset == "n2c2":
        return generate_relations_n2c2(save_to_disk=save_to_disk)
    elif dataset == "ddi":
        return generate_relations_ddi(save_to_disk=save_to_disk)
    else:
        raise ValueError("unsupported dataset '{}'".format(dataset))


def generate_relations_n2c2(save_to_disk: bool = True) -> Dict[str, RelationCollection]:
    """Generates relations of the n2c2 dataset
        1. Per document
        2. Read all entities, all true relations
        3. Separate in to drugs and per attribute
        4. For each relation type, combine each drug with each attribute within the same sentence

    Args:
        save_to_disk (bool): the relation collections are saved to disk in a datading or not. Default to True.

    Returns:
        Dict[str, RelationCollection]: train and test relation collections
    """
    print("Generating relations for the n2c2 dataset...\n")

    dataset = files_n2c2()
    collections = {}

    for split, files in dataset.items():

        print(split, ": ")
        split_entities = []
        split_relations = []

        for basepath in tqdm(files):
            # process clinical text, split in sentences
            document: Document = Document.from_json(read_txt(basepath + ".json"))

            # read annotation file
            entities, gt_relations = read_annotations_n2c2(basepath + ".ann")

            # generate relations
            relations = RelationN2C2.generate_relations_n2c2(
                document, entities, gt_relations, (split == "test")
            )

            split_entities.extend(entities)
            split_relations.extend(relations)

        # create collection
        collection = RelationCollection(split_relations)

        # remove invalid relations
        collection = collection[collection.valid_indexes()]

        # write to databing
        if save_to_disk:
            make_dir(pjoin(N2C2_PATH, "{}_datading".format(split)))
            collection.to_datading(
                pjoin(N2C2_PATH, "{}_datading".format(split), "relations.msgpack")
            )
        
        collections[split] = collection

    return collections


def generate_relations_ddi(save_to_disk: bool = True) -> Dict[str, RelationCollection]:
    """Generates relations of the ddi dataset

    Args:
        save_to_disk (bool): the relation collections are saved to disk in a datading or not. Default to True.

    Returns:
        Dict[str, RelationCollection]: train and test relation collections
    """
    print("Generating relations for the DDI Extraction corpus...")

    dataset = files_ddi()
    collections = {}

    for split, files in dataset.items():
        print(split, ": ")
        split_relations = []

        for file in tqdm(files):
            xml_tree = ET.parse(file)
            relations = RelationDDI.generate_relations_ddi(xml_tree)
            split_relations.extend(relations)

        # create collection
        collection = RelationCollection(split_relations)

        # remove invalid relations
        collection = collection[collection.valid_indexes()]

        # write to databing
        if save_to_disk:
            make_dir(pjoin(DDI_PATH, "{}_datading".format(split)))
            collection.to_datading(
                pjoin(DDI_PATH, "{}_datading".format(split), "relations.msgpack")
            )
            
        collections[split] = collection

    return collections
