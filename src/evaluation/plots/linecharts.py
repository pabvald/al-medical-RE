# Base Dependencies
# -----------------
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from os.path import join as pjoin


# Local Dependencies
# ------------------
from evaluation.io import (
    collect_al_series,
    collect_pl_series_ddi,
    collect_pl_series_n2c2,
    collect_step_times_series,
)

# 3rd-Party Dependencies
# ----------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
# ---------
from constants import METHODS_NAMES, N2C2_REL_TYPES

COLOR_PALETTE = "Set2"


def _iter_time_linechart_with_error_bands(
    results: Dict,
    legend: bool = True,
    title: Optional[str] = None,
    legend_title: Optional[str] = None,
    output_file: Optional[Path] = None,
):
    ALPHA = 0.1
    DDI_TRAIN_SIZE = 27705
    sns.set()
    sns.set_style("ticks")  # grid style
    colors = sns.color_palette(COLOR_PALETTE)
    markers = ["x", "v", "o", "P"]
    linestyles = [
        "solid",
        "dotted",
        "dashed",
        "dashdot",
    ]

    # plot active learning performance
    N = 0
    for x in results.keys():
        if len(results[x]["mean"]) > N:
            N = len(results[x]["mean"])

    x = np.linspace(2.5, 50, num=N)
    for i, q_strategy in enumerate(results.keys()):
        mean = results[q_strategy]["mean"]
        std = results[q_strategy]["std"]

        plt.plot(
            x,
            mean,
            linestyle=linestyles[i],
            color=colors[i],
            marker=markers[i],
            label=q_strategy,
        )
        plt.fill_between(x, mean - std, mean + std, color=colors[i], alpha=ALPHA)

    sns.despine()  # remove top and right axis
    plt.ylabel("Step Time (minutes)")
    plt.xlabel("# of annotated samples (out of 27,705)")

    # set axis to 0%, 10%, 20% 30%, 40%, 50% of annotated dtaset
    xticks = [0, 5, 10, 15, 20, 25,  30,  35, 40, 45, 50]
    xlabels = [f"{str(int(round(x * DDI_TRAIN_SIZE / 100)))} ({x}%)" for x in xticks]
    plt.xticks(
        xticks,
        labels=xlabels,
        rotation=45,
    )

    plt.legend(title=legend_title, loc="best")

    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()


def _al_linechart_with_error_bands(
    al_results: Dict,
    pl_results: Dict,
    y_label: str = "F1 Score",
    legend: bool = True,
    title: Optional[str] = None,
    output_file: Optional[Path] = None,
):
    """Plots a line chart with error bands

    Args:
        results (Dict): dicitonary containing the results for each query strategy
        output_file (Optional[str], optional): path of the output file. Defaults to None.
    """
    ALPHA = 0.1
    sns.set()
    sns.set_style("white")  # grid style
    colors = sns.color_palette(COLOR_PALETTE)
    markers = ["x", "v", "P", ]
    linestyles = [
        "-",
        "--",
        "-.",
    ]

    # plot active learning performance
    N = 0
    for x in al_results.keys():
        if len(al_results[x]["mean"]) > N:
            N = len(al_results[x]["mean"])

    x = np.linspace(2.5, 50, num=N)
    for i, q_strategy in enumerate(al_results.keys()):
        mean = al_results[q_strategy]["mean"]
        std = al_results[q_strategy]["std"]

        plt.plot(
            x,
            mean,
            linestyle=linestyles[i],
            color=colors[i],
            marker=markers[i],
            label=q_strategy,
        )
        plt.fill_between(x, mean - std, mean + std, color=colors[i], alpha=ALPHA)

    # plot passive learning performance
    plt.axhline(
        y=pl_results["mean"], color="black", linestyle="dashed", label="100% Data"
    )

    sns.despine()  # remove top and right axis
    plt.ylabel(y_label)
    plt.xlabel("% annotated dataset")

    # set axis to 0%, 10%, 20% 30%, 40%, 50% of annotated dtaset
    plt.xticks(
        [0, 10, 20, 30, 40, 50],
        labels=["0", "10", "20", "30", "40", "50"],
    )

    plt.legend(loc="lower right")

    if title:
        plt.title(title)
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()


# Main Functions
# --------------
def al_linecharts_n2c2():
    """Plots the line charts for the N2C2 corpus"""
    corpus = "n2c2"

    for method in METHODS_NAMES.keys():
        # load passive learning results
        pl_results = collect_pl_series_n2c2(
            Path(pjoin("results", corpus, "all", method))
        )

        # for each relation type
        for rel_type in N2C2_REL_TYPES:
            output_file = Path(
                pjoin("results", corpus, rel_type, f"al_{method}_n2c2_{rel_type}.png")
            )
            title = f"Relation = {rel_type}, Method = {METHODS_NAMES[method]}"

            # load AL results
            al_results = collect_al_series(
                Path(pjoin("results", corpus, rel_type, method)), metric="f1"
            )

            # plot results
            _al_linechart_with_error_bands(
                al_results=al_results,
                pl_results=pl_results[rel_type],
                title=title,
                output_file=output_file,
            )

        # micro average
        # load AL results
        al_results = collect_al_series(
            Path(pjoin("results", corpus, "all", method)), metric="Micro_f1"
        )

        output_file = Path(pjoin("results", corpus, "all", f"al_{method}_n2c2.png"))
        title = f"Method = {METHODS_NAMES[method]}"
        # plot results
        _al_linechart_with_error_bands(
            al_results=al_results,
            pl_results=pl_results["Micro"],
            title=title,
            y_label="Micro F1 Score",
            output_file=output_file,
        )


def al_linecharts_ddi():
    """Plots the line charts for the DDI Extraction corpus"""
    corpus = "ddi"
    for method in METHODS_NAMES.keys():
        title = f"Method = {METHODS_NAMES[method]}"
        results_path = Path(pjoin("results", corpus, method))
        output_file = Path(pjoin("results", corpus, f"al_{method}_ddi.png"))

        # load results
        al_results = collect_al_series(results_path, metric="Micro_f1")
        pl_results = collect_pl_series_ddi(results_path)

        # plot results
        _al_linechart_with_error_bands(
            al_results=al_results,
            pl_results=pl_results,
            y_label="Micro F1 Score",
            legend=(method == "bert"),
            title=title,
            output_file=output_file,
        )


def iter_time_linecharts():
    results = collect_step_times_series(Path(pjoin("results", "ddi")))

    for method in METHODS_NAMES.keys():
        # all strtategies with one method
        output_file = Path(
            pjoin("results", "ddi", method, f"iter_time_{method}_ddi.png")
        )
        _iter_time_linechart_with_error_bands(
            results[method],
            title=f"Method = {METHODS_NAMES[method]}",
            legend_title="Strategy",
            output_file=output_file,
        )

    for strategy in ["random", "LC", "BatchBALD"]:
        # LC strategy with all methods
        output_file = Path(pjoin("results", "ddi", f"iter_time_{strategy}_ddi.png"))
        lc_results = {}
        for method in METHODS_NAMES.keys():
            if strategy == "BatchBALD" and method == "rf": 
                continue
            lc_results[method] = results[method][strategy]

        _iter_time_linechart_with_error_bands(
            lc_results,
            title=f"Query Strategy = {strategy}",
            legend_title="Method",
            output_file=output_file,
        )
