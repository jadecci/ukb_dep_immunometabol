from pathlib import Path
import argparse

from sklearn.cluster import AgglomerativeClustering, HDBSCAN, Birch, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(
        description="Clustering analysis for depressive scores",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--data_dir", type=Path, help="Absolute path to extracted data")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

field_dict = {}
col_dtypes = {"eid": str}
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

# Data
cols = ["eid"] + field_dict["Dep sympt"]
col_dtype_curr = {key: item for key, item in col_dtypes.items() if key in cols}
data = pd.read_csv(data_file, usecols=cols, dtype=col_dtype_curr, index_col="eid")
data_std = StandardScaler().fit_transform(data[field_dict["Dep sympt"]])

# Clustering methods
models = {
    "hiera_ward_2": AgglomerativeClustering(n_clusters=2, linkage="ward"),
    "hiera_avg_euc_2": AgglomerativeClustering(n_clusters=2, linkage="average", metric="euclidean"),
    "hiera_avg_cosine_2": AgglomerativeClustering(n_clusters=2, linkage="average", metric="cosine"),
    "hiera_avg_l2_2": AgglomerativeClustering(n_clusters=2, linkage="average", metric="l2"),
    "hiera_ward_3": AgglomerativeClustering(n_clusters=3, linkage="ward"),
    "hdbscan_euc": HDBSCAN(min_cluster_size=2, metric="euclidean", copy=True),
    "hdbscan_cos": HDBSCAN(min_cluster_size=2, metric="cosine", copy=True),
    "birch_2": Birch(n_clusters=2),
    "birch_3": Birch(n_clusters=3),
    "spect_nn_2": SpectralClustering(
        n_clusters=2, affinity="nearest_neighbors", assign_labels="cluster_qr"),
    "spect_rbf_2": SpectralClustering(n_clusters=2, affinity="rbf", assign_labels="cluster_qr"),

    "spect_cos_2": SpectralClustering(n_clusters=2, affinity="cosine", assign_labels="cluster_qr"),
    "spect_nn_3": SpectralClustering(
        n_clusters=3, affinity="nearest_neighbors", assign_labels="cluster_qr"),
    "spect_rbf_3": SpectralClustering(n_clusters=3, affinity="rbf", assign_labels="cluster_qr"),

    "spect_cos_3": SpectralClustering(n_clusters=3, affinity="cosine", assign_labels="cluster_qr")}

# Clustering outcomes
clusters = {}
for method, model in models.items():
    model.fit(data_std.T)
    clusters[method] = model.labels_

# Evaluation
eval_names = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
eval_funcs = [silhouette_score, calinski_harabasz_score, davies_bouldin_score]
cluster_eval = {}
cluster_eval_tot = {}
for method, cluster in clusters.items():
    for eval_name, eval_func in zip(eval_names, eval_funcs):
        cluster_eval[f"{method}_{eval_name}"] = {
            "Method": method, "Evaluation": eval_name, "Score": eval_func(data_std.T, cluster)}
    cluster_eval_tot[method] = {
        "Method": method, "Evaluation": "Si x CH / DB", "Score": (
                silhouette_score(data_std.T, cluster)
                * calinski_harabasz_score(data_std.T, cluster)
                / davies_bouldin_score(data_std.T, cluster))}

sns.catplot(
    kind="bar", data=pd.DataFrame(cluster_eval).T, x="Score", y="Method", col="Evaluation",
    orient="h", fill=False, color="gray", linewidth=1.5, sharex=False, height=5, aspect=1)
plt.savefig(Path(args.img_dir, "ukb_dep_cluster_eval.png", bbox_inches="tight", dpi=500))
sns.catplot(
    kind="bar", data=pd.DataFrame(cluster_eval_tot).T, x="Score", y="Method", col="Evaluation",
    orient="h", fill=False, color="gray", linewidth=1.5, sharex=False, height=5, aspect=2)
plt.savefig(Path(args.img_dir, "ukb_dep_cluster_eval_tot.png", bbox_inches="tight", dpi=500))
