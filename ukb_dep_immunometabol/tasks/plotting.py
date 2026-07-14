from pathlib import Path
import json

from pycirclize import Circos
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from scipy.stats import pearsonr, ttest_ind, f_oneway
from PIL import Image
from pydra.compose import python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ukb_dep_immunometabol.utils import load_resource


@python.define
class PlotClusterEval(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Plot evaluation metrics of clustering outcomes

    Args:
        output_dir: Output directory
        evals_file: Clustering evaluation metrics
    """
    output_dir: Path
    evals_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            eval_plot: Plot of clustering evaluation
        """
        eval_plot: Path

    @staticmethod
    def function(output_dir, evals_file):
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(evals_file)
        with sns.plotting_context(context="paper", font_scale=1.5):
            sns.catplot(
                kind="bar", data=data, y="Method", x="Score", col="Metric", orient="h",
                fill=False, color="gray", linewidth=1.5, sharex=False)
        eval_plot = Path(output_plot_dir, "ukb_cluster_eval.png")
        plt.savefig(eval_plot, bbox_inches="tight", dpi=500)

        return eval_plot


@python.define
class PlotHieraCluster(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Plot the clustering outcome of hierarchical clustering

    Args:
        output_dir: Output directory
        link_file: linkage matrix from hierarchical clustering
    """
    output_dir: Path
    link_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            cluster_plot: Plot of clustering outcome
        """
        cluster_plot: Path

    @staticmethod
    def function(output_dir, link_file):
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        hiera_link = np.load(link_file, allow_pickle=True)

        # Depressive symptom score labels
        labels = []
        field_cols = {"Type": str, "Field Description": str, "Notes": str}
        field_file = load_resource("ukb_selected_fields.csv")
        field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
        for _, field_row in field_data.iterrows():
            if field_row["Type"] == "Dep sympt":
                labels.append(f"{field_row['Notes']} {field_row['Field Description']}")

        # Dendrogram
        set_link_color_palette(["gold", "pink", "darkseagreen"])
        _, ax = plt.subplots(figsize=(8, 4))
        dendrogram(
            hiera_link, orientation="left", labels=np.array(labels), ax=ax, leaf_font_size=12,
            above_threshold_color="k", color_threshold=0.8 * max(hiera_link[:, 2]))
        plt.tight_layout()
        cluster_plot = Path(output_plot_dir, "ukb_hiera_cluster.png")
        plt.savefig(cluster_plot, bbox_inches="tight", dpi=500)

        return cluster_plot


@python.define
class PlotSociodemo(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Plot sociodemographic relations with depressive symptom dimensions

    Args:
        output_dir: Output directory
        sdem_file: Sociodemographic data for plotting
    """
    output_dir: Path
    sdem_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            sdem_plot: Plot of sociodemographic relations
        """
        sdem_plot: Path

    @staticmethod
    def function(output_dir, sdem_file):
        raw_plot_dir = Path(output_dir, "raw_plots")
        raw_plot_dir.mkdir(parents=True, exist_ok=True)
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        col_dtypes = (
                {col: str for col in ["eid", "educ", "income", "sex"]}
                | {col: float for col in ["age", "mood", "energy"]})
        data = pd.read_csv(
            sdem_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")

        xticks = [range(4, 25, 2), range(4, 13, 2)]
        yticks = {
            "age": ["50", "60", "70", "80"], "sex": ["Female", "Male"],
            "income": [
                "Less than 18,000", "18,000 to 30,999", "31,000 to 51,999", "52,000 to 100,000",
                "Greater than 100,000"],
            "educ": [
                "College or University degree", "A levels/AS levels or equivalent",
                "O levels/GCSEs or equivalent", "CSEs or equivalent",
                "NVQ or NHD or HNC or equivalent", "Other professional qualifications",
                "None of the above"]}

        # Plot for each sociodemographic factor
        raw_plots = {}
        for y in ["age", "educ", "income", "sex"]:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(6, 2))
            with sns.plotting_context(context="paper", font_scale=1.5):
                for x_i, x in enumerate(["mood", "energy"]):
                    if y == "age":
                        sns.regplot(
                            data, y=y, x=x, ax=ax[x_i], marker="x",
                            scatter_kws={"color": ".2"}, line_kws={"color": "red"})
                        ax[x_i].set_yticks([50, 60, 70, 80], labels=yticks[y])
                        r, p = pearsonr(data[y].astype(float), data[x].astype(float))
                        ax_title = f"R = {r: .3f}, p = {p: .2E}"
                    else:
                        sns.violinplot(
                            data, y=y, x=x, ax=ax[x_i], hue=y, orient="h", linecolor="k",
                            dodge=False, order=yticks[y], palette="Greys", hue_order=yticks[y])
                        if y == "sex":
                            res = ttest_ind(
                                data[x].loc[data[y] == "Female"].astype(float),
                                data[x].loc[data[y] == "Male"].astype(float), equal_var=False)
                            ax_title = f"T = {res.statistic: .3f}, p = {res.pvalue: .2E}"
                        elif y == "income" or y == "educ":
                            sample = [data[x].loc[data[y] == col] for col in yticks[y]]
                            f, p = f_oneway(*sample, equal_var=False)
                            ax_title = f"F = {f: .3f}, p = {p: .2E}"
                    ax[x_i].set_title(ax_title, fontdict={"fontsize": 10})
                    ax[x_i].set_xlabel(f"{x} dimension")
                    ax[x_i].set_xticks(xticks[x_i])
                    ax[x_i].set_ylabel("")
            raw_plot = Path(raw_plot_dir, f"ukb_dep_{y}.png")
            fig.savefig(raw_plot, bbox_inches="tight", dpi=500)
            raw_plots[y] = Image.open(raw_plot)
            plt.close(fig)

        # Combine into one plot
        w = raw_plots["sex"].size[0] + raw_plots["educ"].size[0]
        h = raw_plots["sex"].size[1] + raw_plots["age"].size[1] + 100
        sdem_im = Image.new("RGBA", (w, h))
        sdem_im.paste(raw_plots["sex"])
        sdem_im.paste(raw_plots["age"], (0, raw_plots["sex"].size[1]+100))
        sdem_im.paste(raw_plots["income"], (
            raw_plots["sex"].size[0]+raw_plots["educ"].size[0]-raw_plots["income"].size[0], 0))
        sdem_im.paste(raw_plots["educ"], (raw_plots["sex"].size[0], raw_plots["sex"].size[1]+100))
        sdem_plot = Path(output_plot_dir, "ukb_dep_sdem.png")
        sdem_im.save(sdem_plot)

        return sdem_plot


@python.define
class PlotPLSCorr(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Bar plot of component-y correlations from PLS analysis

    Args:
        output_dir: Output directory
        pls_corr_file: PLS component-y correlation results
        diagn: Diagnosis (MDD, nonMDD, or mixed)
    """
    output_dir: Path
    pls_corr_file: Path
    diagn: str

    class Outputs(python.Outputs):
        """
        Args:
            pls_corr_plot: Plot of PLS component-y correlation
        """
        pls_corr_plot: Path

    @staticmethod
    def function(output_dir, pls_corr_file, diagn):
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        corr = pd.read_csv(pls_corr_file)

        hues = {"Energy sum score": "pink", "Mood sum score": "lawngreen"}
        f = plt.figure(figsize=(12, 12), dpi=500)
        f.subplots(1, 2, subplot_kw={"polar": True})
        f.subplots_adjust(wspace=0.7)
        for ax, sex in zip(f.axes, ["female", "male"]):
            corr_curr = corr.loc[corr["Sex"] == sex]
            corr_plot = corr_curr.pivot(
                index="Depressive score", columns="Feature type", values="Absolute correlation")
            circos = Circos.radar_chart(
                corr_plot, vmax=1, cmap=hues, bg_color=None, grid_interval_ratio=0.2)
            circos.text(f"{diagn} {sex}", r=130, size=16)
            circos.plotfig(ax=ax)

        plt.savefig(
            Path(output_plot_dir, f"ukb_{diagn}_pls_corr.svg"), bbox_inches="tight", format="svg")
        pls_corr_plot = Path(output_plot_dir, f"ukb_{diagn}_pls_corr.png")
        plt.savefig(pls_corr_plot, bbox_inches="tight")
        plt.close()

        return pls_corr_plot


@python.define
class PlotPLSVIP(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Radar bar plots for PLS results of body and blood features

    Args:
        output_dir: Output directory
        pls_vip_file: PLS VIP score results
        cat_pheno: List of phenotype categories
        diagn: Diagnosis (MDD, nonMDD, or mixed)
    """
    output_dir: Path
    pls_vip_file: Path
    cat_pheno: list
    diagn: str

    class Outputs(python.Outputs):
        """
        Args:
            pls_vip_plots: Radar plots of PLS results
        """
        pls_vip_plots: list

    @staticmethod
    def function(output_dir, pls_vip_file, cat_pheno, diagn):
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        data_vip = pd.read_csv(pls_vip_file)
        data_vip = data_vip.loc[data_vip["Feature type"].isin(cat_pheno)]

        # Data field short names and sectors for plotting
        field_names = dict.fromkeys(data_vip["Data field"].unique())
        sectors = {col_type: 0 for col_type in cat_pheno}
        field_cols = {"Field ID": str, "Instance": "Int64", "Short Name": str, "Type": str}
        field_file = load_resource("ukb_selected_fields.csv")
        field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
        for _, field_row in field_data.iterrows():
            col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
            if col_id in field_names.keys():
                field_names[col_id] = field_row["Short Name"]
            if field_row["Type"] in sectors.keys():
                sectors[field_row["Type"]] += 1

        # Add derivative features for Blood count
        sectors["Blood count"] += 4
        field_names.update({"sii": "SII", "nlr": "NLR", "plr": "PLR", "lmr": "LMR"})

        # Blood proteomic marker columns
        pro_cols = data_vip["Data field"].loc[data_vip["Feature type"] == "Proteomics"].unique()
        sectors["Proteomics"] = len(pro_cols)
        field_names.update({col: col for col in pro_cols})

        # Plotting set-ups
        col_type_names = {
            "Body fat": "Body fat", "Blood metabol": "Blood metabolic markers",
            "Blood count": "Blood cell count", "Proteomics": "Blood proteomic markers"}
        hues = {"Mood sum score": "lawngreen", "Energy sum score": "pink"}
        yticks = [0, 1, 2, 3, 4, 5]
        sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

        pls_vip_plots = []
        for sex in ["female", "male"]:
            circos = Circos(sectors, space=3, start=5, end=355)

            for col_type, sector in zip(cat_pheno, circos.sectors):
                sector.axis(fc="none", lw=0)
                sector.text(col_type_names[col_type], size=12, r=170)
                tracks = [sector.add_track((35, 65)), sector.add_track((70, 100))]

                # Plot VIP scores
                for track, (dep, hue) in zip(tracks, hues.items()):
                    track.axis(fc="none", lw=0)
                    data_plot = data_vip.loc[
                        (data_vip["Sex"] == sex) & (data_vip["Feature type"] == col_type)
                        & (data_vip["Depressive score"] == dep)]
                    track.bar(
                        np.arange(0, sector.size), data_plot["VIP score"], width=1, align="edge",
                        vmin=yticks[0], vmax=yticks[-1], color=hue)

                    if dep == "Energy sum score":
                        xticklabels = [field_names[col] for col in data_plot["Data field"]]
                        track.xticks(
                            np.arange(0, sector.size)+0.5, xticklabels,
                            label_orientation="vertical", label_size=10)

                    track.grid(y_grid_num=len(yticks), ls="dashed", color="gray", lw=0.5)
                    if col_type == "Body fat":
                        track.yticks(yticks, yticks, vmin=yticks[0], vmax=yticks[-1], side="left")

            circos.plotfig()

            pls_plot_file = Path(output_plot_dir, f"ukb_{diagn}_pls_vip_{sex}.png")
            plt.savefig(pls_plot_file, bbox_inches="tight", dpi=500)
            pls_vip_plots.append(pls_plot_file)

            pls_plot_file = Path(output_plot_dir, f"ukb_{diagn}_pls_vip_{sex}.svg")
            plt.savefig(pls_plot_file, bbox_inches="tight", format="svg")

            plt.close()

        return pls_vip_plots
