from os import remove
from pathlib import Path
import json
import logging

from sklearn.cluster import AgglomerativeClustering, Birch, HDBSCAN, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from pydra.compose import python
import numpy as np
import pandas as pd

from ukb_dep_immunometabol.utils import load_resource, pls_regression


@python.define
class Clustering(python.Task["CanonicalPythonTask.Outputs"]):
    """Clustering analysis

    Args:
        output_dir: Output directory
        data_file: Extracted data file
        data_cluster_file: Extracted data for clustering analysis
    """
    output_dir: Path
    data_file: Path
    data_cluster_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            dep_score_file: Sum score of depressive symptom clusters for all subjects
            evals_file: Clustering evaluation metrics
            link_file: linkage matrix from hierarchical clustering
        """
        dep_score_file: Path
        evals_file: Path
        link_file: Path

    @staticmethod
    def function(output_dir, data_file, data_cluster_file):
        output_cluster_dir = Path(output_dir, "clustering")
        output_cluster_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("cluster_logger")
        logger.setLevel("INFO")
        logfile = Path(output_cluster_dir, "clustering.log")
        if logfile.exists():
            remove(logfile)
        logfile_handler = logging.FileHandler(logfile)
        logfile_handler.setLevel(logging.INFO)
        logger.addHandler(logfile_handler)

        # Input field information
        cols_dep = []
        col_dtypes = {"eid": str}
        labels = []
        field_cols = {
            "Field ID": str, "Type": str, "Instance": "Int64", "Field Description": str,
            "Notes": str}
        field_file = load_resource("ukb_selected_fields.csv")
        field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
        for _, field_row in field_data.loc[field_data["Type"] == "Dep sympt"].iterrows():
            col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
            cols_dep.append(col_id)
            col_dtypes[col_id] = float
            labels.append(f"{field_row['Notes']} {field_row['Field Description']}")

        # Clustering sample data
        data = pd.read_csv(
            data_cluster_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
        data_std = StandardScaler().fit_transform(data[cols_dep])

        # Clustering set-up
        parameters = {
            "Hierarchical": [
                ["ward", "euclidean", 2], ["ward", "euclidean", 3], ["average", "euclidean", 2],
                ["average", "cosine", 2], ["average", "l2", 2]],
            "HDBSCAN": [["euclidean"], ["cosine"]],
            "Birch": [[0.3, 2], [0.5, 2], [0.7, 2], [0.3, 3], [0.5, 3], [0.7, 3]],
            "Spectral": [
                ["nearest_neighbors", 2], ["rbf", 2], ["cosine", 2],
                ["nearest_neighbors", 3], ["rbf", 3], ["cosine", 3]]}
        models = {}

        # Hierarchical clusering
        for param in parameters["Hierarchical"]:
            method = "-".join(map(str, ["Hierarchical"]+param))
            models[method] = AgglomerativeClustering(
                linkage=param[0], metric=param[1], n_clusters=param[2])

        # HDBSCAN
        for param in parameters["HDBSCAN"]:
            method = "-".join(map(str, ["HDBSCAN"]+param))
            models[method] = HDBSCAN(metric=param[0], min_cluster_size=2, copy=True)

        # Birch
        for param in parameters["Birch"]:
            method = "-".join(map(str, ["Birch"]+param))
            models[method] = Birch(threshold=param[0], n_clusters=param[1], branching_factor=10)

        # Spectral clustering
        for param in parameters["Spectral"]:
            method = "-".join(map(str, ["Spectral"]+param))
            models[method] = SpectralClustering(
                affinity=param[0], n_clusters=param[1], assign_labels="cluster_qr")

        # Clustering outcome evaluation
        evals = pd.DataFrame(columns=["Method", "Metric", "Score"])
        for method, model in models.items():
            model.fit(data_std.T)
            logger.info(f"{method}: {model.labels_}")

            score_si = silhouette_score(data_std.T, model.labels_)
            score_ch = calinski_harabasz_score(data_std.T, model.labels_)
            score_db = davies_bouldin_score(data_std.T, model.labels_)
            score_combine = score_si * score_ch / score_db

            eval_curr = [
                [method, "Silhouette", score_si], [method, "Calinski-Harabsz", score_ch],
                [method, "Davies-Bouldin", score_db], [method, ["Si x CH / DB"], score_combine]]
            eval_ind = [f"method_{eval_method}" for eval_method in ["si", "ch", "db", "combi"]]
            eval_curr = pd.DataFrame(eval_curr, columns=evals.columns, index=eval_ind)
            evals = pd.concat([evals, eval_curr], axis="index")
        evals_file = Path(output_cluster_dir, "ukb_cluster_eval.csv")
        evals.to_csv(evals_file)

        # Use hiera-ward-euclidean-3 for final clustering
        model = AgglomerativeClustering(
            linkage="ward", metric="euclidean", n_clusters=3, compute_distances=True)
        model.fit(data_std.T)

        # Dendrogram linkage
        counts = np.zeros(model.children_.shape[0])
        for i, merge in enumerate(model.children_):
            for child_ind in merge:
                if child_ind < len(model.labels_):
                    counts[i] = counts[i] + 1  # leaf node
                else:
                    counts[i] = counts[i] + counts[child_ind - len(model.labels_)]
        hiera_link = np.column_stack([model.children_, model.distances_, counts]).astype(float)
        link_file = Path(output_cluster_dir, "ukb_cluster_hiera_link.npy")
        np.save(link_file, hiera_link)

        # Define clusters
        clusters = {}
        for i, cluster in enumerate(model.labels_):
            score_curr = f"Sum score {cluster}"
            if score_curr in clusters.keys():
                clusters[score_curr].append(cols_dep[i])
            else:
                clusters[score_curr] = [cols_dep[i]]
            logger.info(f"Cluster {cluster}: {labels[i]}")

        # Compute sum scores for all subjects
        data_all = pd.read_csv(
            data_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
        for cluster_name, cluster_col_list in clusters.items():
            score_curr = data_all[cluster_col_list].sum(axis="columns")
            data_all = data_all.assign(**{cluster_name: score_curr})
        dep_score_file = Path(output_cluster_dir, "ukb_cluster_all.csv")
        data_all[clusters.keys()].to_csv(dep_score_file)

        return dep_score_file, evals_file, link_file


@python.define
class PLS(python.Task["CanonicalPythonTask.Outputs"]):
    """PLS Regression analysis

    Args:
        output_dir: Output directory
        data_pheno_files: Extracted data file for each phenotype category
        field_type_file: Data fields grouped by type
        dep_score_file: Sum score of depressive symptom clusters for all subjects
        cat_pheno: List of phenotype categories
        diagn: Diagnosis (MDD, nonMDD, or mixed)
    """
    output_dir: Path
    data_pheno_files: dict
    field_type_file: Path
    dep_score_file: Path
    cat_pheno: list
    diagn: str

    class Outputs(python.Outputs):
        """
        Args:
            pls_vip_file: PLS VIP score results
            pls_corr_file: PLS component-y correlation results
            diagn: Diagnosis (MDD, nonMDD, or mixed)
        """
        pls_vip_file: Path
        pls_corr_file: Path
        diagn: str

    @staticmethod
    def function(output_dir, data_pheno_files, field_type_file, dep_score_file, cat_pheno, diagn):
        output_pls_dir = Path(output_dir, "pls")
        output_pls_dir.mkdir(parents=True, exist_ok=True)

        with open(field_type_file, "r") as f:
            field_types = json.load(f)

        data_pls = []
        corr_pls = []
        col_dtypes = {"eid": str, "31-0.0": float, "21003-2.0": float}
        dep_scores = {"1": "Mood sum score", "2": "Energy sum score"}
        for col_type in cat_pheno:
            # Association sample
            cols = ["31-0.0", "21003-2.0", "eid", "MDD diagnosis"] + field_types[col_type]
            dtypes = col_dtypes | ({col: float for col in field_types[col_type]})
            data_pheno = pd.read_csv(
                data_pheno_files[col_type], usecols=cols, dtype=dtypes, index_col="eid")
            data_pheno = data_pheno.assign(age2=np.power(data_pheno["21003-2.0"], 2))

            # Add depressive sum scores
            # From final clustering outcome, we use cluster 1 (mood) and 2 (energy)
            cols = ["eid", "Sum score 1", "Sum score 2"]
            dtypes = {"eid": str, "Sum score 1": float, "Sum score 2": float}
            data_dep = pd.read_csv(dep_score_file, usecols=cols, dtype=dtypes, index_col="eid")
            data_pheno = pd.concat([data_pheno, data_dep], axis="columns", join="inner")

            # Add derivative features for blood count
            if col_type == "Blood count":
                eps = np.finfo(float).eps
                data_pheno.loc[:, "sii"] = (
                        data_pheno["30140-0.0"] * data_pheno["30080-0.0"]
                        / (data_pheno["30120-0.0"] + eps))
                data_pheno.loc[:, "nlr"] = data_pheno["30140-0.0"] / (data_pheno["30120-0.0"] + eps)
                data_pheno.loc[:, "plr"] = data_pheno["30080-0.0"] / (data_pheno["30120-0.0"] + eps)
                data_pheno.loc[:, "lmr"] = data_pheno["30120-0.0"] / (data_pheno["30190-0.0"] + eps)
                field_types["Blood count"].extend(["sii", "nlr", "plr", "lmr"])

            # Perform PLS in each sex subgroup
            if diagn == "mixed":
                data_diagn = data_pheno
            else:
                data_diagn = data_pheno.loc[data_pheno["MDD diagnosis"] == diagn]
            for i, sex in enumerate(["female", "male"]):
                data_curr = data_diagn.loc[data_diagn["31-0.0"] == i]

                # Perform PLS for each depressive dimension
                for dep_y, dep_covar in zip(["1", "2"], ["2", "1"]):
                    dict_curr = {
                        "Feature type": col_type, "Sex": sex, "Depressive score": dep_scores[dep_y]}
                    x = data_curr[field_types[col_type]]
                    y = data_curr[f"Sum score {dep_y}"]
                    covar = data_curr[["21003-2.0", "age2", f"Sum score {dep_covar}"]]
                    r, r_p, vips, vips_p = pls_regression(x, y, covar)

                    # Component-Y correlation
                    ind_curr = f"{col_type}_{sex}_{dep_y}"
                    corr_pls_curr = dict_curr | {
                        "Correlation": r, "Absolute correlation": np.abs(r), "P-value": r_p}
                    corr_pls_curr = pd.DataFrame(corr_pls_curr, index=[ind_curr])
                    corr_pls.append(corr_pls_curr)

                    # VIP scores
                    for pheno, vip, vip_p in zip(field_types[col_type], vips, vips_p):
                        ind_curr = f"{pheno}_{sex}_{dep_y}"
                        data_pls_curr = dict_curr | {
                            "Data field": pheno, "VIP score": vip, "P-value": vip_p}
                        data_pls_curr = pd.DataFrame(data_pls_curr, index=[ind_curr])
                        data_pls.append(data_pls_curr)
        corr_pls = pd.concat(corr_pls, axis="index")
        data_pls = pd.concat(data_pls, axis="index")

        # Correct for multiple comparisons
        h = multipletests(corr_pls["P-value"], alpha=0.05, method="fdr_bh")
        corr_pls = corr_pls.assign(Significance=h[0])
        h = multipletests(data_pls["P-value"], alpha=0.05, method="fdr_bh")
        data_pls = data_pls.assign(Significance=h[0])

        pls_vip_file = Path(output_pls_dir, f"ukb_{diagn}_pls_vip.csv")
        data_pls.to_csv(pls_vip_file)

        pls_corr_file = Path(output_pls_dir, f"ukb_{diagn}_pls_corr.csv")
        corr_pls.to_csv(pls_corr_file)

        return pls_vip_file, pls_corr_file, diagn
