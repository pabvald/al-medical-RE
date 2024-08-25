# Base Dependencies
# -----------------
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
from joblib import load
from os.path import join as pjoin

# Package Dependencies
# --------------------
from evaluation.io import (
    collect_pl_results_ddi,
    collect_pl_results_n2c2,
    collect_step_times,
)

# 3rd-Party Dependencies
# ----------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
# ---------
from constants import N2C2_REL_TYPES, METHODS_NAMES

COLOR_PALETTE = "Set2"


# Auxiliar Functions
# ------------------
def __boxplot_pl(
    results: pd.DataFrame,
    metric_name: str = "Micro_f1",
    title: Optional[str] = None,
    output_file: Optional[Path] = None,
):
    """Boxplot of the results of the experiments.

    Args:
        results (pd.DataFrame): results of the experiments
        title (Optional[str], optional): title of the plot. Defaults to None.
        output_file (Optional[Path], optional): path of the output file. Defaults to None.
    """
    sns.set()
    sns.set_style("white")  # grid style
    colors = sns.color_palette(COLOR_PALETTE)  # set color palette

    sns.boxplot(x="method", y=metric_name, data=results, palette=colors, orient="v")
    plt.xlabel("Method")
    plt.ylabel("Micro F1 Score")

    plt.xticks(
        ticks=list(range(len(METHODS_NAMES))),
        labels=list(METHODS_NAMES.values()),
        rotation=45,
        ha="right",
    )

    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()


# Main Function
# -------------
def pl_boxplot_both_corpus():
    """Plots a box plot showing the passive learning results for both corpora"""

    title = None  # "Passive Learning Results"
    output_file = Path(pjoin("results", "pl_boxplot_both_corpus.png"))
    n2c2_folder = Path(pjoin("results", "n2c2", "all"))
    ddi_folder = Path(pjoin("results", "ddi"))

    # collect results
    n2c2_pl_results: pd.DataFrame = collect_pl_results_n2c2(n2c2_folder)
    ddi_pl_results: pd.DataFrame = collect_pl_results_ddi(ddi_folder)

    # combine results into single dataframe
    n2c2_pl_results["corpus"] = "n2c2"
    n2c2_pl_results = n2c2_pl_results[n2c2_pl_results["relation"] == "Micro"]
    n2c2_pl_results = n2c2_pl_results[["f1", "method"]]
    n2c2_pl_results = n2c2_pl_results.rename(columns={"f1": "Micro_f1"})
    n2c2_pl_results["corpus"] = "n2c2"
    ddi_pl_results["corpus"] = "DDI"
    results = pd.concat([n2c2_pl_results, ddi_pl_results])

    # plot
    sns.set()
    sns.set_style("whitegrid")  # grid style
    # colors = sns.color_palette(COLOR_PALETTE)  # set color palette
    colors = ["#a1dab4", "#FEFEBB"]
    plt.figure(figsize=(6, 8), dpi=1200)

    ax = sns.boxplot(
        x="method", y="Micro_f1", data=results, palette=colors, orient="v", hue="corpus"
    )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
        title=None,
        frameon=True,
    )
    plt.xlabel("Method")
    plt.ylabel("Micro F1 Score")
    plt.xticks(
        ticks=list(range(len(METHODS_NAMES))),
        labels=["RF", "BiLSTM", "CBERT", "CBERT-pairs"],
        rotation=0,
        ha="right",
    )

    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()


def pl_boxplot_n2c2():
    """Plots a box plot showing the passive learning results for the N2C2 corpus"""
    folder = Path(pjoin("results", "n2c2", "all"))
    pl_results = collect_pl_results_n2c2(folder)
    __boxplot_pl(
        pl_results,
        title="Corpus: n2c2",
        metric_name="Micro_f1",
        output_file=Path(pjoin("results", "n2c2", "all", "pl_boxplot_n2c2.png")),
    )


def pl_boxplot_ddi():
    """Plots a box plot showing the passive learning results for the DDI Extraction corpus"""
    folder = Path(pjoin("results", "ddi"))
    pl_results = collect_pl_results_ddi(folder)
    __boxplot_pl(
        pl_results,
        title="Corpus: DDI",
        metric_name="Micro_f1",
        output_file=Path(pjoin("results", "ddi", "pl_boxplot_ddi.png")),
    )
