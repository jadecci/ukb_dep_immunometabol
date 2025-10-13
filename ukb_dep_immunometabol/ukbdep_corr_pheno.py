from pathlib import Path
import argparse

from pycirclize import Circos
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def pcorr(data: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    covar = ["21003-2.0", "age2"]
    x_resid = data[x] - np.dot(data[covar], np.linalg.lstsq(data[covar], data[x], rcond=-1)[0])
    r, p = pearsonr(x_resid, data[y])
    return r, p


def corr_analysis(
        data: pd.DataFrame, pheno_desc: pd.DataFrame, pheno_cols: dict, pheno: str,
        out_dir: Path) -> pd.DataFrame:
    out_name = col_type.replace(" ", "-")
    genders = {0: "female", 1: "male"}
    data_corr = {}
    for gender_i in [0, 1]:
        data_curr = data.loc[data["31-0.0"] == gender_i]
        for pheno_col in pheno_cols[pheno]:
            for dep_col in field_dict["Dep score"]:
                pheno_curr = pheno_desc.loc[pheno_desc["Field ID"] == pheno_col.split("-")[0]]
                pheno_desc_curr = pheno_curr["Field Description"].values[0]
                ind = f"{pheno_col}-{dep_col}-{genders[gender_i]}"
                r, p = pcorr(data_curr, pheno_col, dep_col)
                data_corr[ind] = {
                    "Type": pheno, "Data field": pheno_col, "r": r, "p": p,
                    "Absolute r": np.abs(r), "Gender": genders[gender_i],
                    "Depressive score field": dep_col, "Depressive score": dep_desc[dep_col],
                    "Field description": pheno_desc_curr}

    data_corr = pd.DataFrame(data_corr).T
    data_corr.to_csv(Path(out_dir, f"ukb_dep_corr_pheno_{out_name}.csv"))
    return data_corr


parser = argparse.ArgumentParser(
        description="Compute correlations between depressive sum scores and brain/body biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

dep_desc = {
    "Sum score (cluster 6)": f"Depressive mood\nsymptoms",
    "Sum score (cluster 5)": f"Depressive energy\nsymptoms"}
field_dict = {"Dep score": ["Sum score (cluster 5)", "Sum score (cluster 6)"]}
col_dtypes = {
    "eid": str, "31-0.0": float, "21003-2.0": float, "Sum score (cluster 5)": float,
    "Sum score (cluster 6)": float}
args.img_dir.mkdir(parents=True, exist_ok=True)
args.out_dir.mkdir(parents=True, exist_ok=True)

col_names = {"sii": "SII", "nlr": "NLR", "plr": "PLR", "lmr": "LMR"}
sectors = {"Body fat": 0, "Blood count": 4, "Blood metabol": 0}
col_type_names_plot = {
    "Body fat": "Body fat", "Blood metabol": "Blood metabolic markers",
    "Blood count": "Blood cell count"}
colour_energy = {"Body fat": "#ff96ff", "Blood count": "#96c8ff", "Blood metabol": "#fab46e"}
colour_mood = {"Body fat": "#ffdcff", "Blood count": "#bedcff", "Blood metabol": "#ffdcb4"}
sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

# Phenotype field information
field_cols = {
    "Field ID": str, "Field Description": str, "Short Name": str, "Type": str, "Instance": "Int64"}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    if field_row["Type"] in field_dict.keys():
        field_dict[field_row["Type"]].append(col_id)
    else:
        field_dict[field_row["Type"]] = [col_id]
    col_dtypes[col_id] = float
    col_names[col_id] = field_row["Short Name"]
    if field_row["Type"] in sectors.keys():
        sectors[field_row["Type"]] += 1

# Correlation analysis
col_list_req = field_dict["Dep score"] + ["31-0.0", "21003-2.0", "eid"]
col_type_pheno = ["Body fat", "Brain GMV", "Brain WM", "Blood metabol", "Blood count"]
data_corrs = []
for col_type in col_type_pheno:
    # Association sample
    col_list_curr = field_dict[col_type] + col_list_req
    col_dtype_curr = {key: col_dtypes[key] for key in col_list_curr}
    pheno_name = col_type.replace(" ", "-")
    data_pheno = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"), usecols=col_list_curr,
        dtype=col_dtype_curr, index_col="eid")
    data_pheno = data_pheno.assign(age2=np.power(data_pheno["21003-2.0"], 2))

    # Add derived measures for blood count
    if col_type == "Blood count":
        data_pheno.loc[:, "sii"] = (
                data_pheno["30140-2.0"] * data_pheno["30080-2.0"] / data_pheno["30120-2.0"])
        data_pheno.loc[:, "nlr"] = data_pheno["30140-2.0"] / data_pheno["30120-2.0"]
        data_pheno.loc[:, "plr"] = data_pheno["30080-2.0"] / data_pheno["30120-2.0"]
        data_pheno.loc[:, "lmr"] = data_pheno["30120-2.0"] / data_pheno["30190-2.0"]
        field_dict["Blood count"].extend(["sii", "nlr", "plr", "lmr"])
        fields = pd.concat([fields, pd.DataFrame({
            "Field ID": ["sii", "nlr", "plr", "lmr"], "Field Description": [
                "Systemic immune-inflammation index (SII)", "Neutrophil-to-lymphocyte ratio (NLR)",
                "Platelet-to-lymphocyte ratio (PLR)", "Lymphocyte-to-monocyte ratio (LMR)"]})])

    data_corr_curr = corr_analysis(data_pheno, fields, field_dict, col_type, args.out_dir)
    data_corrs.append(data_corr_curr)
data_corr_all = pd.concat(data_corrs, axis="index", join="inner")
data_corr_all["r"] = data_corr_all["r"].fillna(0)
data_corr_all.to_csv(Path(args.out_dir, "ukb_dep_corr_pheno_all.csv"))

# Correct for multiple comparisons
fdr = multipletests(data_corr_all["p"], method="fdr_bh")
data_corr_fdr = data_corr_all.loc[fdr[0]]
data_corr_fdr.to_csv(Path(args.out_dir, "ukb_dep_corr_pheno_fdr.csv"))

# Plot all correlation results: radar bar plot
for gender in ["female", "male"]:
    data_corr_curr = data_corr_all.loc[data_corr_all["Gender"] == gender]
    circos = Circos(sectors, space=3, start=5, end=355)

    for col_type, sector in zip(sectors.keys(), circos.sectors):
        data_plot_curr = data_corr_curr.loc[data_corr_curr["Type"] == col_type]
        data_energy = data_plot_curr.loc[
            data_plot_curr["Depressive score field"] == "Sum score (cluster 5)"]
        data_mood = data_plot_curr.loc[
            data_plot_curr["Depressive score field"] == "Sum score (cluster 6)"]

        sector.axis(fc="none", lw=0)
        sector.text(col_type_names_plot[sector.name], size=12, r=155)
        track = sector.add_track((20, 100))
        track.axis(fc="none", lw=0)

        xticks_energy = np.arange(0, sector.size)
        xticks_mood = np.arange(0, sector.size) + 0.5
        xticks_centre = np.arange(0, sector.size) + 0.25
        xtick_labels = [col_names[col_id] for col_id in data_energy["Data field"]]
        yticks = [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

        track.bar(
            xticks_energy, data_energy["r"], width=0.5, vmin=yticks[0], vmax=yticks[-1],
            color=colour_energy[col_type])
        track.bar(
            xticks_mood, data_mood["r"], width=0.5, vmin=yticks[0], vmax=yticks[-1],
            color=colour_mood[col_type])
        track.xticks(xticks_centre, xtick_labels, label_orientation="vertical", label_size=10)
        track.grid(y_grid_num=len(yticks), ls="dashed", color="gray")
        if col_type == "Body fat":
            track.yticks(yticks, yticks, vmin=yticks[0], vmax=yticks[-1], side="left")

    f = circos.plotfig()
    plt.savefig(
        Path(args.img_dir, f"ukb_dep_corr_pheno_radar_{gender}.png"), bbox_inches="tight", dpi=500)
    plt.savefig(
        Path(args.img_dir, f"ukb_dep_corr_pheno_radar_{gender}.svg"), bbox_inches="tight",
        format="svg")
    plt.close()
