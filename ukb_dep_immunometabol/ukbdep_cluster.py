from pathlib import Path
import argparse

from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(
        description="Clustering analysis for depressive scores",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

field_dict = {}
col_dtypes = {"eid": str, "MDD diagnosis": float}
args.img_dir.mkdir(parents=True, exist_ok=True)
data_file = Path(args.data_dir, "ukb_data_cluster.csv")

# Phenotype field information
field_cols = {
    "Field ID": str, "Field Description": str, "Type": str, "Instance": "Int64", "Notes": str}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    if field_row["Type"] in field_dict.keys():
        field_dict[field_row["Type"]].append(col_id)
    else:
        field_dict[field_row["Type"]] = [col_id]
    col_dtypes[col_id] = float

# Agglomerate depressive symptom features
cols_curr = ["eid"] + field_dict["Dep sympt"]
col_dtype_curr = {key: item for key, item in col_dtypes.items() if key in cols_curr}
data = pd.read_csv(data_file, usecols=cols_curr, dtype=col_dtype_curr, index_col="eid")
data_std = StandardScaler().fit_transform(data[field_dict["Dep sympt"]])
model = FeatureAgglomeration(n_clusters=2, compute_distances=True)
model.fit(data_std)

# Dendrogram
counts = np.zeros(model.children_.shape[0])
for i, merge in enumerate(model.children_):
    for child_ind in merge:
        if child_ind < len(model.labels_):
            counts[i] = counts[i] + 1 # leaf node
        else:
            counts[i] = counts[i] + counts[child_ind - len(model.labels_)]
linkage_mat = np.column_stack([model.children_, model.distances_, counts]).astype(float)

dep_col_list = [col.split("-")[0] for col in field_dict["Dep sympt"]]
labels_desc = fields["Field Description"].loc[fields["Field ID"].isin(dep_col_list)].tolist()
labels_note = fields["Notes"].loc[fields["Field ID"].isin(dep_col_list)].tolist()
labels = np.array([f"[{note}] {desc}" for desc, note in zip(labels_desc, labels_note)])

colors=["skyblue", "darkseagreen", "pink", "gold", "mediumturquoise", "plum", "orange"]
set_link_color_palette(colors)
_, ax = plt.subplots(figsize=(12, 8))
dendro_res = dendrogram(
    linkage_mat, orientation="left", labels=labels, ax=ax, leaf_font_size=12,
    above_threshold_color="k", color_threshold=0.59*max(linkage_mat[:, 2]))
plt.tight_layout()
plt.savefig(Path(args.img_dir, "ukb_dep_cluster.png", bbox_inches="tight", dpi=500))

# Define clusters
clusters = {}
for leaf, color in zip(reversed(dendro_res["leaves"]), reversed(dendro_res["leaves_color_list"])):
    cluster_name = f"Sum score (cluster {len(colors) - colors.index(color)})"
    if cluster_name in clusters.keys():
        clusters[cluster_name].append(field_dict["Dep sympt"][leaf])
    else:
        clusters[cluster_name] = [field_dict["Dep sympt"][leaf]]
    print(cluster_name, color, labels[leaf])

# Apply clustering to association sample
col_type_pheno = ["Body fat", "Brain GMV", "Brain WM", "Blood metabol", "Blood count"]
data_sdem = []
for col_type in col_type_pheno:
    col_list_curr = (
        field_dict[col_type] + field_dict["Dep sympt"] + field_dict["Sociodemo"]
        + ["eid", "MDD diagnosis"])
    col_dtype_curr = {key: col_dtypes[key] for key in col_list_curr}
    pheno_name = col_type.replace(" ", "-")
    data_pheno = pd.read_csv(
        Path(args.data_dir, f"ukb_data_{pheno_name}.csv"), usecols=col_list_curr,
        dtype=col_dtype_curr, index_col="eid")
    for cluster_name, cluster_col_list in clusters.items():
        data_pheno[cluster_name] = data_pheno[cluster_col_list].sum(axis="columns")
    data_pheno.to_csv(Path(args.data_dir, f"ukb_data_{pheno_name}_clusters.csv"))
    data_sdem.append(data_pheno)
data_sdem = pd.concat(data_sdem, axis="index", join="inner")
data_sdem = data_sdem.drop_duplicates()
data_sdem.to_csv(Path(args.data_dir, "ukb_data_sociodemo_clusters.csv"))
