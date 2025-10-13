from pathlib import Path
import argparse

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(
        data: pd.DataFrame, y_col: str, ytick: list | None, yticklabel: list, x_cols: list,
        out_name: str, plot_type: str):
    dep_desc = ["Depressive mood symptoms", "Depressive energy symptoms"]
    f, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(10, 2.5))
    with sns.plotting_context(context="paper", font_scale=1.5):
        for x_i, x in enumerate(x_cols):
            if plot_type == "violin":
                sns.violinplot(
                    data, y=y_col, x=x, hue=y_col, orient="h", ax=ax[x_i], linecolor="k",
                    dodge=False, order=yticklabel, palette="Greys", hue_order=yticklabel)
            elif plot_type == "reg":
                sns.regplot(
                    data, y=y_col, x=x, ax=ax[x_i], marker="x", scatter_kws={"color": ".2"},
                    line_kws={"color": "red"})
                r, p = pearsonr(data[y_col].astype(float), data[x].astype(float))
                ax[x_i].set_title(f"R = {r:.4f}, p = {p:.4f}", fontdict={"fontsize": 10})
                ax[x_i].set_yticks(ytick, labels=yticklabel)
            if x_i == 0:
                ax[x_i].set_xticks(range(4, 25, 2))
            elif x_i == 1:
                ax[x_i].set_xticks(range(4, 17, 2))
            ax[x_i].set(xlabel=dep_desc[x_i], ylabel="")
    f.savefig(Path(args.img_dir, f"ukb_dep_{out_name}.png"), bbox_inches="tight", dpi=500)
    plt.close(f)


parser = argparse.ArgumentParser(
        description="Plot association between depressive scores and sociodemographic phenotypes",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_csv", type=Path, help="Absolute path to the data csv file to use")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

dep_score = ["Sum score (cluster 5)", "Sum score (cluster 6)"]
col_dtypes = {
    "eid": str, "Sum score (cluster 5)": float, "Sum score (cluster 6)": float}
args.img_dir.mkdir(parents=True, exist_ok=True)

# Phenotype field information
field_cols = {"Field ID": str, "Type": str, "Instance": "Int64"}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.loc[fields["Type"] == "Sociodemo"].iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    col_dtypes[col_id] = float

data_sdem = pd.read_csv(
    args.data_csv, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")

# violinplot: gender
gender_dict = {0: "Female", 1: "Male"}
data_sdem = data_sdem.assign(gender=data_sdem["31-0.0"].apply(lambda x: gender_dict.get(x)))
plot(data_sdem, "gender", None, ["Female", "Male"], dep_score, "gender", "violin")

# violinplot: education
educ_dict = {
    -7: "None of the above", 1: "College or University degree",
    2: "A levels/AS levels or equivalent", 3: "O levels/GCSEs or equivalent",
    4: "CSEs or equivalent", 5: "NVQ or NHD or HNC or equivalent",
    6: "Other professional qualifications"}
data_sdem = data_sdem.assign(educ=data_sdem["6138-2.0"].apply(lambda x: educ_dict.get(x)))
yticks_l = [educ_dict[i] for i in [1, 2, 3, 4, 5, 6, -7]]
plot(data_sdem, "educ", None, yticks_l, dep_score, "educ", "violin")

# violinplot: household income
income_dict = {
    1: "Less than 18,000", 2: "18,000 to 30,999", 3: "31,000 to 51,999", 4: "52,000 to 100,000",
    5: "Greater than 100,000"}
data_sdem = data_sdem.assign(income=data_sdem["738-2.0"].apply(lambda x: income_dict.get(x)))
yticks_l = [income_dict[i] for i in [1, 2, 3, 4, 5]]
plot(data_sdem, "income", None, yticks_l, dep_score, "income", "violin")

# regplot: age
yticks_l = ["50", "60", "70", "80"]
plot(data_sdem, "21003-2.0", [50, 60, 70, 80], yticks_l, dep_score, "age", "reg")
