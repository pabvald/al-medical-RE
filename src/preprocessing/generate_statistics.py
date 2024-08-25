# coding: utf-8

# Base Dependencies
# ------------------
import numpy as np
from collections import Counter
from typing import Dict
from pathlib import Path
from os.path import join as pjoin
from tqdm import tqdm

# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection

# 3rd-Party Dependencies
# ----------------------
import pandas as pd
from tabulate import tabulate

# Constants
# ---------
from constants import N2C2_REL_TYPES, DDI_ALL_TYPES, N2C2_PATH, DDI_PATH

TABLE_FORMAT = "latex"


# Main Functions
# ---------------
def generate_statistics(dataset: str, collections: Dict[str, RelationCollection]):
    if dataset == "n2c2":
        return generate_statistics_n2c2(collections)
    elif dataset == "ddi":
        return generate_statistics_ddi(collections)
    else:
        raise ValueError("unsupported dataset '{}'".format(dataset))


def generate_statistics_n2c2(collections: Dict[str, RelationCollection]):
    """Generates the statistics for the n2c2 dataset"""

    df_counts = {
        "relation": [],
        "train_positive": [],
        "train_negative": [],
        "test_positive": [],
        "test_negative": [],
    }
    df_seq_lengths = {
        "relation": [],
        "train_min": [],
        "train_avg": [],
        "train_max": [],
        "test_min": [],
        "test_avg": [],
        "test_max": [],
    }

    # number of relations per type of relation
    for rel_type in tqdm(N2C2_REL_TYPES):
        df_counts["relation"].append(rel_type)
        df_seq_lengths["relation"].append(rel_type)

        for split, collection in collections.items():
            subcollection = collection.type_subcollection(rel_type)

            # add counts to data
            count_labels = Counter(subcollection.labels)

            df_counts[split + "_negative"].append(count_labels[0])
            df_counts[split + "_positive"].append(count_labels[1])

            # add sequence length to dataframe
            seq_lengths = list(
                map(lambda rel: len(rel.text.split()), subcollection.relations)
            )

            df_seq_lengths[split + "_min"].append(min(seq_lengths))
            df_seq_lengths[split + "_avg"].append(sum(seq_lengths) / len(subcollection))
            df_seq_lengths[split + "_max"].append(max(seq_lengths))

    df_counts = pd.DataFrame(df_counts)
    df_seq_lengths = pd.DataFrame(df_seq_lengths)

    # add totals to counts
    df_counts["train_total"] = df_counts["train_positive"] + df_counts["train_negative"]
    df_counts["test_total"] = df_counts["test_positive"] + df_counts["test_negative"]
    df_counts["total_positive"] = (
        df_counts["train_positive"] + df_counts["test_positive"]
    )
    df_counts["total_negative"] = (
        df_counts["train_negative"] + df_counts["test_negative"]
    )
    df_counts["total"] = df_counts["total_positive"] + df_counts["total_negative"]
    df_counts.loc[len(df_counts)] = ["Total"] + [
        df_counts[col].sum() for col in df_counts.columns[1:]
    ]

    all_train_seq_lengths = list(map(lambda rel: len(rel.text.split()), collections["train"].relations))
    all_test_seq_lengths = list(map(lambda rel: len(rel.text.split()), collections["test"].relations))
    df_seq_lengths = df_seq_lengths.append(
        {
            "relation": "Overall",
            "train_min": min(all_train_seq_lengths),
            "train_avg": sum(all_train_seq_lengths) / len(all_train_seq_lengths),
            "train_max": max(all_train_seq_lengths),
            "test_min": min(all_test_seq_lengths),
            "test_avg": sum(all_test_seq_lengths) / len(all_test_seq_lengths),
            "test_max": max(all_test_seq_lengths),
        },
        ignore_index=True,
    )

    # select and reorder columns
    df_counts = df_counts.loc[:, ["relation", "train_positive", "train_negative", "train_total", "test_positive", "test_negative", "test_total", "total"]]

    # save data to csv
    df_counts.to_csv(Path(pjoin(N2C2_PATH, "counts.csv")), index=False)
    df_seq_lengths.to_csv(Path(pjoin(N2C2_PATH, "seq_length.csv")), index=False)

    # print statistics
    print("\n **** Statistics of the N2C2 Dataset ****")
    print("Counts:")
    print(tabulate(df_counts, headers="keys", tablefmt=TABLE_FORMAT))
    print("Seq Length:")
    print(tabulate(df_seq_lengths, headers="keys", tablefmt=TABLE_FORMAT))


def generate_statistics_ddi(collections: Dict[str, RelationCollection]) -> None:
    """Generates the statistics of the DDI dataset"""

    df_counts = {"relation": [], "train": [], "test": []}
    df_seq_lengths = {
        "relation": [],
        "train_min": [],
        "train_avg": [],
        "train_max": [],
        "test_min": [],
        "test_avg": [],
        "test_max": [],
    }

    for rel_type in DDI_ALL_TYPES:
        df_counts["relation"].append(rel_type)
        df_seq_lengths["relation"].append(rel_type)

        for split, collection in collections.items():
            subcollection = collection.type_subcollection(rel_type)

            df_counts[split].append(len(subcollection))

            seq_lengths = list(
                map(lambda rel: len(rel.text.split()), subcollection.relations)
            )
            df_seq_lengths[split + "_min"].append(min(seq_lengths))
            df_seq_lengths[split + "_avg"].append(sum(seq_lengths) / len(subcollection))
            df_seq_lengths[split + "_max"].append(max(seq_lengths))

    # convert to dataframes
    df_counts = pd.DataFrame(df_counts)
    df_seq_lengths = pd.DataFrame(df_seq_lengths)

    # add totals
    train_negative = df_counts.loc[(df_counts["relation"] == "NO-REL"), "train"].values[
        0
    ]
    train_positive = df_counts.loc[(df_counts["relation"] != "NO-REL"), "train"].sum()
    test_negative = df_counts.loc[(df_counts["relation"] == "NO-REL"), "test"].values[0]
    test_positive = df_counts.loc[(df_counts["relation"] != "NO-REL"), "test"].sum()
    train_total = train_positive + train_negative
    test_total = test_positive + test_negative
    total = train_total + test_total

    # add positive row
    df_counts.loc[len(df_counts)] = ["Total Positive", train_positive, test_positive]

    # add totals
    df_counts["total"] = df_counts["train"] + df_counts["test"]
    df_counts.loc[len(df_counts)] = [" Total", train_total, test_total, total]

    all_train_seq_lengths = list(map(lambda rel: len(rel.text.split()), collections["train"].relations))
    all_test_seq_lengths = list(map(lambda rel: len(rel.text.split()), collections["test"].relations))
    df_seq_lengths = df_seq_lengths.append(
        {
            "relation": "Overall",
            "train_min": min(all_train_seq_lengths),
            "train_avg": sum(all_train_seq_lengths) / len(all_train_seq_lengths),
            "train_max": max(all_train_seq_lengths),
            "test_min": min(all_test_seq_lengths),
            "test_avg": sum(all_test_seq_lengths) / len(all_test_seq_lengths),
            "test_max": max(all_test_seq_lengths),
        },
        ignore_index=True,
    )

    # save data to csv
    df_counts.to_csv(Path(pjoin(DDI_PATH, "counts.csv")), index=False)
    df_seq_lengths.to_csv(Path(pjoin(DDI_PATH, "seq_length.csv")), index=False)

    # print statistics
    print("\n **** Statistics of the DDI Dataset ****")
    print("Counts:")
    print(tabulate(df_counts, headers="keys", tablefmt=TABLE_FORMAT))
    print("Seq Length:")
    print(tabulate(df_seq_lengths, headers="keys", tablefmt=TABLE_FORMAT))
