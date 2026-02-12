from pathlib import Path

from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_glass_brain, plot_surf_stat_map
from nilearn.surface import load_surf_data
from pycirclize import Circos
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from scipy.stats import pearsonr, ttest_ind, f_oneway
from PIL import Image
from pydra.compose import python
import matplotlib.pyplot as plt
import nibabel as nib
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
        set_link_color_palette(["gold", "darkseagreen", "pink"])
        _, ax = plt.subplots(figsize=(12, 8))
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

        dep_desc = ["Mood dimension", "Energy dimension"]
        xticks = [range(4, 21, 2), range(4, 17, 2)]
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
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 2.5))
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
                    ax[x_i].set_xticks(xticks[x_i])
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
class PlotPLSRadar(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Radar bar plots for PLS results of body and blood features

    Args:
        output_dir: Output directory
        pls_file: PLS results
    """
    output_dir: Path
    pls_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            pls_radar_plots: Radar plots of PLS results
        """
        pls_radar_plots: list

    @staticmethod
    def function(output_dir, pls_file):
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        col_type_pheno = ["Body fat", "Blood metabol", "Blood count"]
        data_pls = pd.read_csv(pls_file)
        data_plot = data_pls.loc[data_pls["Feature type"].isin(col_type_pheno)]

        # Data field short names and sectors for plotting
        field_names = dict.fromkeys(data_plot["Data field"].unique())
        sectors = {col_type: 0 for col_type in col_type_pheno}
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

        # Plotting set-ups
        col_type_names = {
            "Body fat": "Body fat", "Blood metabol": "Blood metabolic markers",
            "Blood count": "Blood cell count"}
        colour_energy = {
            "Body fat": "#ff96ff", "Blood count": "#96c8ff", "Blood metabol": "#fab46e"}
        colour_mood = {"Body fat": "#ffdcff", "Blood count": "#bedcff", "Blood metabol": "#ffdcb4"}
        yticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

        pls_radar_plots = []
        for sex in ["female", "male"]:
            data_curr = data_plot.loc[data_plot["Sex"] == sex]
            circos = Circos(sectors, space=3, start=5, end=355)

            for col_type, sector in zip(col_type_pheno, circos.sectors):
                data_sector = data_curr.loc[data_curr["Feature type"] == col_type]
                data_energy = data_sector.loc[data_sector["Depressive score"] == "Sum score 1"]
                data_mood = data_sector.loc[data_sector["Depressive score"] == "Sum score 2"]

                sector.axis(fc="none", lw=0)
                sector.text(col_type_names[col_type], size=12, r=155)
                track = sector.add_track((20, 100))
                track.axis(fc="none", lw=0)

                track.bar(
                    np.arange(0, sector.size), data_energy["VIP score"], width=0.5,
                    vmin=yticks[0], vmax=yticks[-1], color=colour_energy[col_type])
                track.bar(
                    np.arange(0, sector.size)+0.5, data_mood["VIP score"], width=0.5,
                    vmin=yticks[0], vmax=yticks[-1], color=colour_mood[col_type])

                xticklabels = [field_names[col] for col in data_energy["Data field"]]
                track.xticks(
                    np.arange(0, sector.size)+0.25, xticklabels, label_orientation="vertical",
                    label_size=10)

                track.grid(y_grid_num=len(yticks), ls="dashed", color="gray")
                if col_type == "Body fat":
                    track.yticks(yticks, yticks, vmin=yticks[0], vmax=yticks[-1], side="left")

            circos.plotfig()

            pls_plot_file = Path(output_plot_dir, f"ukb_pls_radar_{sex}.png")
            plt.savefig(pls_plot_file, bbox_inches="tight", dpi=500)
            pls_radar_plots.append(pls_plot_file)

            pls_plot_file = Path(output_plot_dir, f"ukb_pls_radar_{sex}.svg")
            plt.savefig(pls_plot_file, bbox_inches="tight", format="svg")
            pls_radar_plots.append(pls_plot_file)

            plt.close()

        return pls_radar_plots


@python.define
class PlotPLSBrain(python.Task["CanonicalPythonTask.Outputs"]):
    """
    Radar bar plots for PLS results of brain features

    Args:
        output_dir: Output directory
        pls_file: PLS results
    """
    output_dir: Path
    pls_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            pls_brain_plots: Plots of PLS results for brain features
        """
        pls_brain_plots: list

    @staticmethod
    def function(output_dir, pls_file):
        raw_plot_dir = Path(output_dir, "raw_plots")
        raw_plot_dir.mkdir(parents=True, exist_ok=True)
        output_plot_dir = Path(output_dir, "plots")
        output_plot_dir.mkdir(parents=True, exist_ok=True)

        data_pls = pd.read_csv(pls_file)

        # Data field description for white matter features
        field_desc = dict.fromkeys(
            data_pls["Data field"].loc[data_pls["Feature type"] == "Brain WM"].unique())
        field_cols = {"Field ID": str, "Instance": "Int64", "Field Description": str, "Type": str}
        field_file = load_resource("ukb_selected_fields.csv")
        field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
        for _, field_row in field_data.iterrows():
            col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
            if col_id in field_desc.keys():
                field_desc[col_id] = field_row["Field Description"]

        # JHU atlas
        jhu_img = nib.load(load_resource("JHU-ICBM-labels-1mm.nii.gz"))
        jhu_vol = jhu_img.get_fdata()
        jhu_lut = pd.read_csv(load_resource("JHU_labels.csv"))

        # Schaefer 200-parcel atlas
        fs_meshes = load_fsaverage(mesh="fsaverage")
        fs_bg = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="sulcal")
        l_annot = load_surf_data(load_resource("lh.Schaefer2018_200Parcels_17Networks_order.annot"))
        r_annot = load_surf_data(load_resource("rh.Schaefer2018_200Parcels_17Networks_order.annot"))
        fs_lut = pd.read_csv(load_resource("Schaefer200_17Networks_labels.csv"))

        # Melbourne S2 subcortex atlas
        mel_img = nib.load(load_resource("Tian_Subcortex_S2_7T.nii"))
        mel_vol = mel_img.get_fdata()
        mel_lut = pd.read_table(load_resource("Tian_Subcortex_S2_7T.csv"), delimiter=";")

        pls_brain_plots = []
        for sex in ["female", "male"]:
            for dep_score, dep in zip(["Sum score 1", "Sum score 2"], ["energy", "mood"]):
                data_curr = data_pls.loc[
                    (data_pls["Sex"] == sex) & (data_pls["Depressive score"] == dep_score)]

                # White matter features
                for feature_type in ["FA", "MD", "ICVF", "ISOVF", "OD"]:
                    feature_cols = [
                        col for col, desc in field_desc.items() if f"Mean {feature_type}" in desc]
                    vol_curr = np.zeros(jhu_vol.shape)
                    for col in feature_cols:
                        data_plot = data_curr.loc[data_curr["Data field"] == col]
                        wm_val = data_plot["VIP score"].values[0]
                        wm_region = field_desc[col].split("in ")[1]
                        wm_parc = jhu_lut["Atlas value"].loc[jhu_lut["Brain region"] == wm_region]
                        vol_curr[np.where(jhu_vol == wm_parc.values[0])] = wm_val

                    wm_img = nib.Nifti1Image(vol_curr, jhu_img.affine)
                    pls_wm_plot = Path(output_plot_dir, f"ukb_pls_{feature_type}_{sex}_{dep}.png")
                    plot_glass_brain(
                        wm_img, output_file=pls_wm_plot, colorbar=True, cmap="OrRd",
                        vmin=0, vmax=3, symmetric_cbar=False, plot_abs=False, threshold=None)
                    pls_brain_plots.append(pls_wm_plot)

                # Gray matter volume (subcortical)
                vol_curr = np.zeros(mel_vol.shape)
                for _, mel_row in mel_lut.iterrows():
                    data_plot = data_curr.loc[data_curr["Data field"] == mel_row["ROIabbr"]]
                    gmv_val = data_plot["VIP score"].values[0]
                    vol_curr[np.where(mel_vol == mel_row["ROIid"])] = gmv_val

                gmv_img = nib.Nifti1Image(vol_curr, mel_img.affine)
                pls_gmv_plot = Path(output_plot_dir, f"ukb_pls_gmv_subcort_{sex}_{dep}.png")
                plot_glass_brain(
                    gmv_img, output_file=pls_gmv_plot, colorbar=True, cmap="OrRd",
                    vmin=0, vmax=3, symmetric_cbar=False, plot_abs=False, threshold=None)
                pls_brain_plots.append(pls_gmv_plot)

                # Gray matter volume (cortical)
                gmv_images = []
                for hemi, annot in zip(["left", "right"], [l_annot, r_annot]):
                    fs_surf = np.zeros(annot.shape)
                    for _, fs_row in fs_lut.iterrows():
                        data_plot = data_curr.loc[data_curr["Data field"] == fs_row[hemi]]
                        gmv_val = data_plot["VIP score"].values[0]
                        fs_surf[np.where(annot == fs_row["Atlas value"])] = gmv_val

                    for view in ["laterl", "medial"]:
                        pls_gmv_plot = Path(
                            raw_plot_dir, f"ukb_pls_gmv_cort_{sex}_{dep}_{hemi}_{view}.png")
                        plot_surf_stat_map(
                            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf, bg_map=fs_bg,
                            hemi=hemi, view=view, cmap="OrRd", colorbar=True, vmin=0, vmax=3,
                            symmetric_cbar=False, threshold=None, output_file=pls_gmv_plot)
                        gmv_images.append(Image.open(pls_gmv_plot))

                # Combine GMV cortical plots into one plot
                # Individual image sizes should be (470, 500).
                # To cover the colorbars, we paste each plot at (400, 0)
                gmv_image = gmv_images[0]
                for i in range(1, len(gmv_images)):
                    gmv_image.paste(gmv_images[i], (400, 0))
                pls_gmv_plot = Path(output_plot_dir, f"ukb_pls_gmv_cort_{sex}_{dep}.png")
                gmv_image.save(pls_gmv_plot)
                pls_brain_plots.append(pls_gmv_plot)

        return pls_brain_plots
