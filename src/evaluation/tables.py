# Base Dependencies
# -----------------
import re
import numpy as np
from pathlib import Path
from os.path import join as pjoin

# Local Dependencies
# ------------------
from evaluation.io import (
    collect_step_times,
    collect_annotation_rates,
    collect_step_times_sum,
)

# 3rd-Party Dependencies
# ----------------------
import pandas as pd
from tabulate import tabulate

# Constants
# ---------
from constants import (
    N2C2_REL_TYPES,
    N2C2_REL_TEST_WEIGHTS,
    METHODS_NAMES,
)
from evaluation.explainability.random_forest import FEATURE_LABELS

FORMAT_STRATEGY = {"random": 1, "LC": 2, "BatchLC": 3, "BatchBALD": 4}


def _fcell_ar(mean: float, std: float, decimals: int = 3) -> str:
    """Formats a cell of an Annotation Rate table"""
    value = "{:2.2f} +- {:2.2f}".format(round(mean, decimals), round(std, decimals))
    value = re.sub(r"^0\.", ".", value)
    return value


def _fcell_pl(mean: float, std: float, decimals: int = 3) -> str:
    """Formats a cell of a Passive Learning table

    Args:
        mean (float): mean value
        std (float): standard deviation
        decimals (int, optional): number of decimals to represent the values. Defaults to 3.

    Returns:
        str: formatted cell content
    """
    value = "{:0.3f}+-{:0.2f}".format(round(mean, decimals), round(std, decimals))
    value = re.sub(r"^0\.", ".", value)
    return value


def _fcell_al(mean: float, std: float, strategy: str, decimals: int = 3) -> str:
    """Formats a cell of an Active Learning table

    Args:
        mean (float): mean value
        std (float): standard deviation
        strategy (str): query strategy that obtain the (best) performance
        decimals (int, optional): number of decimals to represent the values. Defaults to 3.

    Returns:
        str: formatted cell content
    """
    value = "{:0.3f}+-{:0.2f} superscript{}".format(
        round(mean, decimals), round(std, decimals), FORMAT_STRATEGY[strategy]
    )
    value = re.sub(r"0\.", ".", value)
    return value


def rename_strategy(strategy: str) -> str:
    """Renames the strategy to be displayed in the table"""
    if strategy in ["BatchLC", "BatchBALD"]:
        return "BatchLC / BatchBALD"
    else:
        return strategy


# Main Functions
# --------------
def pl_table_ddi():
    """Generates the results table for passive learning training on the DDI Extraction corpus"""
    TABLE_HEADERS = [
        "Method",
        "Detection",
        "Effect",
        "Mechanism",
        "Advise",
        "Interaction",
        "Macro",
        "Micro",
    ]

    # table with related work
    table = [
        ["Chowdhury et al.", ".800", ".628", ".679", ".692", ".547", ".648", ".651"],
        ["Quan et al.", ".790", ".682", ".722", ".780", ".510", ".674", ".702"],
    ]

    for i, method in enumerate(METHODS_NAMES.keys()):
        # read method's results
        path = Path(pjoin("results", "ddi", method, "passive learning", "results.csv"))
        df = pd.read_csv(path)

        df = df[
            [
                "DETECT_f1",
                "EFFECT_f1",
                "MECHANISM_f1",
                "ADVISE_f1",
                "INT_f1",
                "Macro_f1",
                "Micro_f1",
            ]
        ]
        # compute mean and standard deviation of experiments
        means = df.mean(axis=0)
        stds = df.std(axis=0)

        row = [METHODS_NAMES[method]]
        for i in range(len(means)):
            row.append(_fcell_pl(means[i], stds[i]))
        table.append(row)

    print(tabulate(table, headers=TABLE_HEADERS, tablefmt="latex"))


def pl_table_n2c2():
    """Generates the results table for the passive learning training on the n2c2 corpus"""
    TABLE_HEADERS = [
        "Method",
        "Strength",
        "Duration",
        "Route",
        "Form",
        "ADE",
        "Dosage",
        "Reason",
        "Frequency",
        "Macro",
        "Micro",
    ]

    # table with the related work
    table = [
        ["Xu et al.", "-", "-", "-", "-", "-", "-", "-", "-", "-", ".965"],
        [
            "Alimova et al.",
            ".875",
            ".769",
            ".896",
            ".843",
            ".696",
            ".874",
            ".716",
            ".843",
            ".814",
            ".852",
        ],
        [
            "Wei et al. ",
            ".985",
            ".892",
            ".972",
            ".975",
            ".812",
            ".971",
            ".767",
            ".964",
            ".917",
            "-",
        ],
    ]

    for i, method in enumerate(METHODS_NAMES.keys()):
        all_experiments = pd.DataFrame()
        df_method = pd.read_csv(
            Path(
                pjoin(
                    "results", "n2c2", "all", method, "passive learning", "results.csv"
                )
            )
        )

        # get results for each relation type
        for rel_type in N2C2_REL_TYPES + ["Macro", "Micro"]:
            df_relation = df_method[df_method["relation"] == rel_type]
            relation_column = pd.DataFrame({rel_type: df_relation["f1"].values})
            all_experiments = pd.concat([all_experiments, relation_column], axis=1)

        # add method's row to  latex table
        means = list(all_experiments.mean(axis=0))
        stds = list(all_experiments.std(axis=0))

        row = [METHODS_NAMES[method]]
        for j in range(len(means)):
            row.append(_fcell_pl(means[j], stds[j]))
        table.append(row)

    print(tabulate(table, headers=TABLE_HEADERS, tablefmt="latex"))


def al_table_ddi():
    """Generates the results table for the active learning training on the DDI Extraction corpus"""
    TABLE_HEADERS = [
        "Method",
        "Detection",
        "Effect",
        "Mechanism",
        "Advise",
        "Interaction",
        "Macro",
        "Micro",
    ]

    metrics = [
        "DETECT_f1 (max)",
        "EFFECT_f1 (max)",
        "MECHANISM_f1 (max)",
        "ADVISE_f1 (max)",
        "INT_f1 (max)",
        "Macro_f1 (max)",
        "Micro_f1 (max)",
    ]
    table = []
    for i, method in enumerate(METHODS_NAMES.keys()):
        # load results
        path = Path(pjoin("results", "ddi", method, "active learning", "results.csv"))
        if not path.is_file():
            continue
        df = pd.read_csv(path)

        # sort resutls by creation time
        df = df.sort_values(by=["Creation Time"])

        # discard unnecessary columns
        df = df[["strategy"] + metrics]

        # get means and stds of the runs
        means = df.groupby(["strategy"], as_index=False).mean()
        stds = df.groupby(["strategy"], as_index=False).std()

        # add method's row to  latex table
        row = [METHODS_NAMES[method]]
        for metric in metrics:
            try:
                idxmax = means[metric].idxmax()
                mean = means.iloc[idxmax][metric]
                std = stds.iloc[idxmax][metric]
                strategy = means.iloc[idxmax]["strategy"]

                row.append(_fcell_al(mean, std, strategy))
            except TypeError:
                row.append("-")
        table.append(row)

    # print table
    print(tabulate(table, headers=TABLE_HEADERS, tablefmt="latex"))


def al_table_n2c2():
    """Generates the results table for the active learning training on the n2c2 corpus"""

    TABLE_HEADERS = ["Method"] + N2C2_REL_TYPES + ["Macro", "Micro"]
    metric = "f1 (max)"
    table = []
    for i, method in enumerate(METHODS_NAMES.keys()):
        # load results
        path = Path(
            pjoin("results", "n2c2", "all", method, "active learning", "results.csv")
        )

        if not path.is_file():
            continue
        df = pd.read_csv(path)

        # sort resutls by creation time and relation type
        df = df.sort_values(by=["relation", "Creation Time"])

        # discard unnecessary columns
        df = df[["strategy", "relation", metric]]

        # get means and stds of the runs
        means = df.groupby(["relation", "strategy"], as_index=False).mean()
        stds = df.groupby(["relation", "strategy"], as_index=False).std()

        # select the best value for each relation
        row_means = []
        row_stds = []
        row_strategies = []
        for relation in N2C2_REL_TYPES + ["Macro", "Micro"]:
            idxmax = means[means["relation"] == relation][metric].idxmax()
            mean = means.iloc[idxmax][metric]
            std = stds.iloc[idxmax][metric]
            strategy = means.iloc[idxmax]["strategy"]

            row_means.append(mean)
            row_stds.append(std)
            row_strategies.append(strategy)

        # add method's row to  latex table
        row = [METHODS_NAMES[method]]
        for mean, std, strategy in zip(row_means, row_stds, row_strategies):
            row.append(_fcell_al(mean, std, strategy))
        table.append(row)

    # print table
    print(tabulate(table, headers=TABLE_HEADERS, tablefmt="latex"))


def al_improvements_table_n2c2():
    """Generates the improvements table for the active learning training on the n2c2 corpus"""
    TABLE_HEADERS = [
        "Strategy",
        "Strength",
        "Duration",
        "Route",
        "Form",
        "ADE",
        "Dosage",
        "Reason",
        "Frequency",
        "Macro",
        "Micro",
    ]

    for i, method in enumerate(METHODS_NAMES.keys()):
        table = []
        all_experiments = pd.DataFrame()
        pl_results = pd.read_csv(
            Path(
                pjoin(
                    "results", "n2c2", "all", method, "passive learning", "results.csv"
                )
            )
        )
        al_results = pd.read_csv(
            Path(
                pjoin(
                    "results", "n2c2", "all", method, "active learning", "results.csv"
                )
            )
        )

        # sort resutls by creation time and relation type
        al_results = al_results.sort_values(by=["relation", "Creation Time"])

        # discard unnecessary columns
        al_results = al_results[["strategy", "relation", "f1 (max)"]]

        # get results for each relation type
        for strategy in al_results["strategy"].unique():
            row = [strategy]
            for rel_type in N2C2_REL_TYPES + ["Macro", "Micro"]:
                pl_score = pl_results.loc[
                    pl_results["relation"] == rel_type, "f1"
                ].mean()
                al_score = al_results.loc[
                    (al_results["relation"] == rel_type)
                    & (al_results["strategy"] == strategy),
                    "f1 (max)",
                ].mean()
                improvement = (al_score - pl_score) * 100
                row.append(improvement)

            table.append(row)

        print("Method: ", METHODS_NAMES[method])
        print(tabulate(table, headers=TABLE_HEADERS, tablefmt="markdown"))
        print("\n\n")


def step_time_table():
    """Generates the results table for the AL step times"""

    ddi_data = collect_step_times(Path(pjoin("results", "ddi")))
    n2c2_data = collect_step_times(Path(pjoin("results", "n2c2", "all")))
    ddi_data["Corpus"] = "DDI"
    n2c2_data["Corpus"] = "n2c2"
    data = pd.concat([ddi_data, n2c2_data])

    # edit columns
    data["strategy"] = data["strategy"].apply(lambda x: rename_strategy(x))

    for column in [
        "iter_time (average)",
        "iter_time (max)",
        "iter_time (min)",
    ]:
        data[column] = data[column].apply(lambda x: x / 60)
        data[column] = data[column].apply(lambda x: round(x, 2))

    # create table
    HEADERS = ["Method", "Strategy", "n2c2", "n2c2", "n2c2", "DDI", "DDI", "DDI"]
    table = [["Method", "Strategy", "Min.", "Avg.", "Max.", "Min.", "Avg.", "Max."]]
    for method in METHODS_NAMES.keys():
        for q_strategy in ["random", "LC", "BatchLC / BatchBALD"]:
            row = [
                METHODS_NAMES[method],
                q_strategy,
            ]
            for corpus in ["n2c2", "DDI"]:
                for column in [
                    "iter_time (min)",
                    "iter_time (average)",
                    "iter_time (max)",
                ]:
                    index = (
                        (data["method"] == method)
                        & (data["strategy"] == q_strategy)
                        & (data["Corpus"] == corpus)
                    )

                    mean = data.loc[index, column].mean()
                    std = data.loc[index, column].std()
                    row.append(_fcell_ar(mean, std))
            table.append(row)

    print(tabulate(table, headers=HEADERS, tablefmt="latex"))


def step_time_sum_table():
    """Generates the results table for the total AL step time"""

    ddi_data = collect_step_times_sum(Path(pjoin("results", "ddi")))
    n2c2_data = collect_step_times_sum(Path(pjoin("results", "n2c2", "all")))
    data = dict()
    data["DDI"] = ddi_data
    data["n2c2"] = n2c2_data

    # create table
    HEADERS = ["Method", "Strategy", "n2c2", "DDI"]
    table = []
    for method in METHODS_NAMES.keys():
        if method == "rf":
            strategies = ["random", "LC", "BatchLC"]
        else:
            strategies = ["random", "LC", "BatchBALD"]
        for q_strategy in strategies:
            row = [
                METHODS_NAMES[method],
                q_strategy,
            ]
            for corpus in ["n2c2", "DDI"]:
                mean = data[corpus][method][q_strategy]["mean"]
                std = data[corpus][method][q_strategy]["std"]
                row.append(_fcell_ar(mean, std))
            table.append(row)

    print(tabulate(table, headers=HEADERS, tablefmt="latex"))


def ar_table():
    ar_ddi = collect_annotation_rates(Path(pjoin("results", "ddi")))
    ar_n2c2 = collect_annotation_rates(Path(pjoin("results", "n2c2", "all")))
    ar_ddi["Corpus"] = "DDI"
    ar_n2c2["Corpus"] = "n2c2"
    ar_results = pd.concat([ar_ddi, ar_n2c2])

    # edit columns
    ar_results["CAR"] = ar_results["CAR"].apply(lambda x: x * 100)
    ar_results["TAR"] = ar_results["TAR"].apply(lambda x: x * 100)
    ar_results["IAR"] = ar_results["IAR"].apply(lambda x: x * 100)

    # table
    HEADERS = [
        "Method",
        "Strategy",
        "TAR (%)",
        "CAR (%)",
        "TAR (%)",
        "CAR (%)",
    ]
    table = []

    for method in METHODS_NAMES.keys():
        for q_strategy in ["random", "LC", "BatchLC / BatchBALD"]:
            row = [
                METHODS_NAMES[method],
                q_strategy,
            ]
            for corpus in ["n2c2", "DDI"]:
                for metric in ["TAR", "CAR"]:
                    index = (
                        (ar_results["method"] == method)
                        & (ar_results["strategy"] == q_strategy)
                        & (ar_results["Corpus"] == corpus)
                    )

                    mean = ar_results.loc[index, metric].mean()
                    std = ar_results.loc[index, metric].std()
                    row.append(_fcell_ar(mean, std))
            table.append(row)

    print(tabulate(table, headers=HEADERS, tablefmt="latex"))
