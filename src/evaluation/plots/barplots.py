# Base Dependencies
# -----------------
from pathlib import Path
from os.path import join as pjoin

# Local Dependencies
# ------------------
from evaluation.io import collect_step_times

# 3rd-Party Dependencies
# ----------------------
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
# ---------
COLOR_PALETTE = "Set2"


# Auxiliar Functions
# ------------------
def rename_strategies(x):
    if x == "BatchLC" or x == "BatchBALD":
        return "BatchLC / BatchBALD"
    else:
        return x


# Main Functions
# --------------
def step_time_barplot():
    """Plots a strip plot showing the step time results for the DDI Extraction corpus"""
    title = "Average Step Time per Method and Query Strategy"
    output_file = Path(pjoin("results", "step_time_barplot.png"))

    # collect results
    n2c2_results = collect_step_times(Path(pjoin("results", "n2c2", "all")))
    ddi_results = collect_step_times(Path(pjoin("results", "ddi")))
    n2c2_results["Corpus"] = ["n2c2"] * len(n2c2_results)
    ddi_results["Corpus"] = ["DDI"] * len(ddi_results)
    results = pd.concat([n2c2_results, ddi_results], ignore_index=True)

    # convert step time to minutes
    results["iter_time (average)"] = results["iter_time (average)"].apply(
        lambda x: x / 60
    )
    results["strategy"] = results["strategy"].apply(rename_strategies)

    # plot
    sns.set()
    sns.set_style("whitegrid")  # grid style
    colors = sns.color_palette(COLOR_PALETTE)
    fig, axes = plt.subplots(2)
    g = sns.FacetGrid(results, row="Corpus", aspect=2, legend_out=True, sharex=False)
    g.map(sns.barplot, "iter_time (average)", "method", "strategy", palette=colors)
    g.add_legend(title="Query Strategy")
    g.set_ylabels("Method")
    g.set_xlabels("Step Time (minutes)")
    g.set_yticklabels(["RF", "BiLSTM", "CBERT", "CBERT-pairs"])

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.45, 1),
        ncol=3,
        title=None,
        frameon=True,
    )

    # ax.bar_label(ax.containers[0], fmt='%.f%%')

    if title:
        # plt.title(title)
        pass
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()
