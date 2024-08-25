"""Statistical tests for the experiments of the N2C2 and DDI corpora."""

# Base Dependencies
# -----------------
import numpy as np
from pathlib import Path
from os.path import join as pjoin
from typing import List

# Package Dependencies
# --------------------
from .io import *

# 3rd Party Dependencies
# ----------------------
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from glob import glob
from scipy.stats import wilcoxon, levene, shapiro, f_oneway, f, kruskal, chi2
from statsmodels.stats.multitest import multipletests
from Orange.evaluation import compute_CD, graph_ranks
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tabulate import tabulate

# Constants
# ---------
from constants import METHODS_NAMES, DATASETS, DATASETS_PATHS

ALPHA = 0.05
STRATEGIES = {
    "rf": ["LC", "BatchLC"],
    "bilstm": ["LC", "BatchBALD"],
    "bert": ["LC", "BatchBALD"],
    "bert-pairs": ["LC", "BatchBALD"],
}

BASELINE_NAME = "random"


# ANOVA Tests
# ----------
def one_way_anova_pl(df: pd.DataFrame, corpus: str, alpha: float):
    """Performs a one-way ANOVA test to determine if there are significant differences between the methods"""
    # group results by method
    grouped = []
    for method in df["method"].unique():
        grouped.append(list(df[df["method"] == method]["Micro_f1"]))

    # Perform a one-way ANOVA test
    _, pval = f_oneway(*grouped)

    print("\n\n**** One-way ANOVA test  ****")
    print("  - Corpus: ", corpus)
    print(
        "  - Description: determines if there are significant differences between the methods"
    )
    print("  - p-value: ", pval)
    print("  - Significant difference: ", bool(pval < alpha))

    # if there is a signficant difference between the methods, permform Tukey's HSD test
    if pval < alpha:
        tukey_hsd(df, corpus, alpha)


def two_way_anova_pl(df: pd.DataFrame):
    """Performs a two-way ANOVA test to determine if there are significant differences between the methods and corpora"""
    # two-way ANOVA test
    formula = "Micro_f1 ~ C(method) + C(corpus) + C(method):C(corpus)"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Check the normality of residuals using Shapiro-Wilk test
    residuals = model.resid
    w, p_value = shapiro(residuals)

    print("\n\n**** Two-way ANOVA assumptions  ****")
    print(" Normality of residuals - Shapiro-Wilk test:")
    print("     W:", w)
    print("     p-value:", p_value, "(", (p_value > ALPHA), ")")

    # create Q-Q plot of residuals
    sm.qqplot(model.resid, line="s")
    plt.show()

    # Check the homogeneity of variances using Levene's test
    grouped_data = df.groupby(["method", "corpus"])["Micro_f1"]
    w, p_value = levene(*[group.values for name, group in grouped_data])
    print(" Homogeneity of variances - Levene's test:")
    print("     W:", w)
    print("     p-value:", p_value, "(", p_value > ALPHA, ")")

    #
    print("\n\n**** Two-way ANOVA test  ****")
    print(anova_table)


# Post-hoc tests
# --------------
def tukey_hsd(df: pd.DataFrame, alpha: float = ALPHA):
    """Performs Tukey's HSD test to determine which pairs of methods have significantly different means"""
    corpora = list(df["corpus"].unique())

    # Perform Tukey's HSD test
    tukey_results_method = pairwise_tukeyhsd(df["Micro_f1"], df["method"], alpha=alpha)
    if len(corpora) > 1:
        tukey_resutls_corpus = pairwise_tukeyhsd(
            df["Micro_f1"], df["corpus"], alpha=alpha
        )

    print("\n\n**** Tukey's HDS test  ({})****".format(corpora))
    print(
        "  - Description: determines which pairs of methods have significantly different means"
    )
    print(tukey_results_method)
    if len(corpora) > 1:
        print(tukey_resutls_corpus)


def nemenyi_test(avg_ranks: List[float], methods: List[str], N: int):
    """Performs Nemenyi's test to determine which pairs of methods have significantly different means"""
    cd_nemenyi = compute_CD(
        avranks=list(avg_ranks), n=N, alpha=str(ALPHA), test="nemenyi"
    )
    print("\n\n**** Nemenyi's test  ****")
    print(" - critical distance (alpha = {}) = ".format(ALPHA), cd_nemenyi)
    print(" - avg. ranks:")
    for i in range(len(methods)):
        print("     - {}: {}".format(methods[i], avg_ranks[i]))
    graph_ranks(
        avg_ranks,
        methods,
        cd=cd_nemenyi,
        width=9,
        textspace=1,
    )
    plt.show()


def bonferroni_test(avg_ranks: List[float], methods: List[str], N: int):
    """Performs Bonferroni's test to determine which pairs of methods have significantly different means"""
    cd_bonferroni = compute_CD(
        avranks=list(avg_ranks), n=N, alpha=str(ALPHA), test="bonferroni"
    )
    print("\n\n**** Bonferroni's test  ****")
    print(" - critical distance (alpha = {}) = ".format(ALPHA), cd_bonferroni)
    print(" - avg. ranks:")
    for i in range(len(methods)):
        print("     - {}: {}".format(methods[i], avg_ranks[i]))
    graph_ranks(
        avg_ranks,
        methods,
        cd=cd_bonferroni,
        width=9,
        textspace=1,
    )
    plt.show()


# Non-parametric tests
# --------------------
def kruskal_wallis_method(df: pd.DataFrame):
    """Performs the Kruskal-Wallis test to determine if there are significant differences between the methods"""
    metric = "Micro_f1"
    corpora = list(df["corpus"].unique())
    methods = list(METHODS_NAMES.keys())

    # Create a list of the data for each method
    method1_data = df[df["method"] == methods[0]][metric]
    method2_data = df[df["method"] == methods[1]][metric]
    method3_data = df[df["method"] == methods[2]][metric]
    method4_data = df[df["method"] == methods[3]][metric]

    # Perform the Kruskal-Wallis test
    H, p = kruskal(method1_data, method2_data, method3_data, method4_data)

    # Print the test results
    print("\n\n**** Kruskal-Wallis test  ({})****".format(corpora))
    print("Kruskal-Wallis H statistic: ", H)
    print("p-value: ", p)


def ivan_and_davenport_test(df: pd.DataFrame):
    """Performs Ivan and Davenport test to determine which pairs of methods have significantly different means"""
    N = len(df["corpus"].unique())  # number of corpora
    K = len(df["method"].unique())  # number of methods

    # calculate mean of f1 for each combination of method and corpus
    df_mean = df.groupby(["method", "corpus"], as_index=False).mean()

    # pivot the dataframe
    df_pivot = df_mean.pivot(index="corpus", columns="method", values="Micro_f1")
    print(df_pivot)

    # compute square ranks of each method
    df_ranks = df_pivot.rank(axis=1, ascending=False)

    avg_ranks = df_ranks.mean(axis=0).to_dict()
    sqr_avg_ranks = np.array(list(map(lambda x: x**2, avg_ranks.values())))

    # perform Friedman test
    friedman = (
        (12 * N) / (K * (K + 1)) * (sum(sqr_avg_ranks) - (K / 4 * ((K + 1) ** 2)))
    )
    F_f = (N - 1) * friedman / (N * (K - 1) - friedman)
    p_value = 1 - f.cdf(F_f, (K - 1), (K - 1) * (N - 1))

    print("\n\n**** Ivan and Davenport test  ****")
    print("F_F statistic: ", F_f)
    print("p-value: ", p_value)

    print()

    # perform Nemenyi test if Friedman test is significant
    if p_value < ALPHA:
        nemenyi_test(list(avg_ranks.values()), list(avg_ranks.keys()), N)
        bonferroni_test(list(avg_ranks.values()), list(avg_ranks.keys()), N)


def friedman_test(df: pd.DataFrame):
    """Performs Friedman's test to determine if there is a significant difference between the methods"""
    N = len(df["corpus"].unique())  # number of corpora
    K = len(df["method"].unique())  # number of methods

    # calculate mean of f1 for each combination of method and corpus
    df_mean = df.groupby(["method", "corpus"], as_index=False).mean()

    # pivot the dataframe
    df_pivot = df_mean.pivot(index="corpus", columns="method", values="Micro_f1")

    # compute square ranks of each method
    df_ranks = df_pivot.rank(axis=1, ascending=False)
    avg_ranks = df_ranks.mean(axis=0).to_dict()
    sqr_avg_ranks = np.array(list(map(lambda x: x**2, avg_ranks.values())))
    assert len(sqr_avg_ranks) == K

    # perform Friedman test
    friedman = (
        (12 * N) / (K * (K + 1)) * (sum(sqr_avg_ranks) - (K / 4 * ((K + 1) ** 2)))
    )
    p_value = 1 - chi2.cdf(friedman, K - 1)

    print("\n\n**** Friedman test  ****")
    print("Friedman statistic: ", friedman)
    print("p-value: ", p_value)

    print()

    # perform Nemenyi test if there is a significant difference
    if p_value < ALPHA:
        nemenyi_test(list(avg_ranks.values()), list(avg_ranks.keys()), N)
        bonferroni_test(list(avg_ranks.values()), list(avg_ranks.keys()), N)


def load_method_data(method: str, strategy: str):
    """Loads the data for a given method and strategy."""
    data = []

    for corpus in DATASETS:
        if corpus == "n2c2":
            path = Path(
                pjoin("results", "n2c2", "all", method, "active learning", strategy)
            )
        else:
            path = Path(pjoin("results", "ddi", method, "active learning", strategy))

        for exp in glob(str(pjoin(path, "*f1.csv"))):
            df = pd.read_csv(exp)
            data = data + df["Micro_f1"].tolist()

    return data


# Main
# ----
def pl_statistical_tests():
    """Statistical tests for the passive learning experiments."""

    # load results
    n2c2_results = collect_pl_results_n2c2(Path(pjoin("results", "n2c2", "all")))
    ddi_results = collect_pl_results_ddi(Path(pjoin("results", "ddi")))

    # concatenate results
    n2c2_results = n2c2_results[n2c2_results["relation"] == "Micro"]
    n2c2_results = n2c2_results[["f1", "method"]]
    n2c2_results.columns = ["Micro_f1", "method"]
    n2c2_results["corpus"] = "n2c2"
    ddi_results["corpus"] = "ddi"
    results = pd.concat([n2c2_results, ddi_results])

    # friedman_test(results)
    ivan_and_davenport_test(results)


def al_performance_statistical_tests():
    """Statistical tests for the active learning experiments."""

    # create an array to store the p-values for each strategy and scenario
    p_values = np.zeros((2, 4))

    for j, method in enumerate(METHODS_NAMES.keys()):
        method_p_values = []
        strategies = STRATEGIES[method]

        for i, strategy_name in enumerate(strategies):
            # iterate over each scenario
            strategy_data = load_method_data(method=method, strategy=strategy_name)
            baseline_data = load_method_data(method=method, strategy=BASELINE_NAME)

            assert len(strategy_data) == len(baseline_data)

            # calculate the Wilcoxon signed-rank test p-value for the pairs
            _, p_value = wilcoxon(
                x=strategy_data, y=baseline_data, alternative="greater"
            )

            # store the p-value for the current strategy and method
            method_p_values.append(p_value)

        # perform Bonferroni correction on the p-values for the current scenario
        rejected, corrected_p_values, _, _ = multipletests(
            method_p_values, alpha=ALPHA, method="bonferroni"
        )

        # store the corrected p-values in the array
        p_values[:, j] = corrected_p_values

    # print the corrected p-values and indicate whether the null hypothesis is rejected or not
    for j, method in enumerate(METHODS_NAMES.keys()):
        strategies = STRATEGIES[method]

        for i, strategy_name in enumerate(strategies):
            is_rejected = p_values[i, j] <= ALPHA
            print(
                f"Strategy {strategy_name} vs. {BASELINE_NAME} with method {method}: "
                f"p-value = {p_values[i, j]:.10f}, "
                f"null hypothesis is {'rejected' if is_rejected else 'not rejected'}"
            )
            print()


def ar_statistical_test():
    ar_ddi = collect_annotation_rates(Path(pjoin("results", "ddi")))
    ar_n2c2 = collect_annotation_rates(Path(pjoin("results", "n2c2", "all")))
    ar_ddi["Corpus"] = "DDI"
    ar_n2c2["Corpus"] = "n2c2"
    ar_results = pd.concat([ar_ddi, ar_n2c2])

    # create an array to store the p-values for each strategy and scenario

    for metric in ["TAR", "CAR"]:
        p_values = np.zeros((2, 4))

        for j, method in enumerate(METHODS_NAMES.keys()):
            method_p_values = []

            for i, strategy_name in enumerate(["LC", "BatchLC / BatchBALD"]):
                # iterate over each scenario
                strategy_data = ar_results.loc[
                    (ar_results["method"] == method)
                    & (ar_results["strategy"] == strategy_name)
                ][metric]
                baseline_data = ar_results.loc[
                    (ar_results["method"] == method)
                    & (ar_results["strategy"] == BASELINE_NAME)
                ][metric]

                assert len(strategy_data) == len(baseline_data)

                # calculate the Wilcoxon signed-rank test p-value for the pairs
                _, p_value = wilcoxon(
                    x=strategy_data, y=baseline_data, alternative="two-sided"
                )

                # store the p-value for the current strategy and method
                method_p_values.append(p_value)

            # perform Bonferroni correction on the p-values for the current scenario
            rejected, corrected_p_values, _, _ = multipletests(
                method_p_values, alpha=ALPHA, method="bonferroni"
            )

            # store the corrected p-values in the array
            p_values[:, j] = corrected_p_values

        # print the corrected p-values and indicate whether the null hypothesis is rejected or not
        print("\nMetric: ", metric)
        for j, method in enumerate(METHODS_NAMES.keys()):
            strategies = STRATEGIES[method]

            for i, strategy_name in enumerate(strategies):
                is_rejected = p_values[i, j] <= ALPHA
                if is_rejected:
                    print(
                        f"Strategy {strategy_name} vs. {BASELINE_NAME} with method {method}: "
                        f"p-value = {p_values[i, j]:.10f}, "
                        f"null hypothesis is {'rejected' if is_rejected else 'not rejected'}"
                    )
                    print()
