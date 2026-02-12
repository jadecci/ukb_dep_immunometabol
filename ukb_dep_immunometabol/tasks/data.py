from os import remove
from pathlib import Path
import logging

from pydra.compose import python
import numpy as np
import pandas as pd

from ukb_dep_immunometabol.utils import load_resource


@python.define
class ExtractData(python.Task["CanonicalPythonTask.Outputs"]):
    """Extract data needed for analysis

    Args:
        data_dir: Raw data directory
        output_dir: Output directory
    """
    data_dir: Path
    output_dir: Path

    class Outputs(python.Outputs):
        """
        Args:
            data_file: Extracted data file
            data_pheno_files: Extracted data file for each phenotype category
            data_cluster_file: Extracted data for clustering analysis
            data_sdem_file: Extracted data for sociodemograhpic analysis
            field_types: Data fields grouped by type
        """
        data_file: Path
        data_pheno_files: dict
        data_cluster_file: Path
        data_sdem_file: Path
        field_types: dict

    @staticmethod
    def function(data_dir, output_dir):
        output_data_dir = Path(output_dir, "extracted_data")
        output_data_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("data_logger")
        logger.setLevel("INFO")
        logfile = Path(output_data_dir, "extract_data.log")
        if logfile.exists():
            remove(logfile)
        logfile_handler = logging.FileHandler(logfile)
        logfile_handler.setLevel(logging.INFO)
        logger.addHandler(logfile_handler)

        # Input field information
        field_types = {"Diagn ICD10": []}
        col_dtypes = {"eid": str}
        excludes = {}
        field_cols = {
            "Field ID": str, "Type": str, "Instance": "Int64", "To Exclude 1": float,
            "To Exclude 2": float}
        field_file = load_resource("ukb_selected_fields.csv")
        field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
        for _, field_row in field_data.iterrows():
            col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
            if field_row["Type"] in field_types.keys():
                field_types[field_row["Type"]].append(col_id)
            else:
                field_types[field_row["Type"]] = [col_id]
            col_dtypes[col_id] = float
            if not np.isnan(field_row["To Exclude 1"]):
                excludes[col_id] = [field_row["To Exclude 1"]]
                if not np.isnan(field_row["To Exclude 2"]):
                    excludes[col_id].append(field_row["To Exclude 2"])

        # ICD-10 code
        raw_data_file = Path(data_dir, "ukb_raw.tsv")
        data_head = pd.read_table(raw_data_file, delimiter="\t", encoding="ISO-8859-1", nrows=2)
        for col in data_head.columns:
            if col.split("-")[0] in "41270":
                field_types["Diagn ICD10"].append(col)
                col_dtypes[col] = str

        data_file = Path(output_data_dir, "ukb_data_all.csv")
        if data_file.exists():
            remove(data_file)

        # Subjects with all depressive symptom items
        data_iter = pd.read_table(
            raw_data_file, delimiter="\t", encoding="ISO-8859-1", chunksize=1000,
            usecols=list(col_dtypes), dtype=col_dtypes, index_col="eid")
        for data_pheno in data_iter:
            data_out = data_pheno.dropna(
                axis="index", how="any", subset=field_types["Dep sympt"])
            if not data_out.empty:
                if data_file.exists():
                    data_out.to_csv(data_file, mode="a", header=False)
                else:
                    data_out.to_csv(data_file)
        data_all = pd.read_csv(
            data_file, usecols=list(col_dtypes), dtype=col_dtypes, index_col="eid")
        logger.info(f"Subjects with all depressive symptom items: N = {data_all.shape[0]}")

        # Remove withdrawn subjects
        wd_sub = pd.read_csv(Path(data_dir, "withdrawn_subjects.csv"), header=None).squeeze()
        data_all = data_all.drop(wd_sub, errors="ignore")
        logger.info(f"Subjects with non-retracted consent: N = {data_all.shape[0]}")

        # Remove subjects with "do not know / prefer not to answer" code for depressive symptoms
        for col_id, exclude_list in excludes.items():
            if col_id in field_types["Dep sympt"]:
                for exclude in exclude_list:
                    data_all = data_all.loc[data_all[col_id] != exclude]
        logger.info(f"Subjects with valid code for depressive symptoms: N = {data_all.shape[0]}")
        # Remove subjects with "do not know / prefer not to answer" code for sociodemographic items
        for col_id, exclude_list in excludes.items():
            if col_id in field_types["Sociodemo"]:
                for exclude in exclude_list:
                    data_all = data_all.loc[data_all[col_id] != exclude]
        logger.info(f"Subjects with valid code for sociodemographic items: N = {data_all.shape[0]}")

        # Remove subjects who reported no symptoms
        data_remove = data_all.copy()
        for col_id in field_types["Dep sympt"]:
            data_remove = data_remove.loc[data_remove[col_id] == 1]  # frequency
        data_all = data_all.drop(data_remove.index)
        logger.info(f"Subjects who reported at least one symptom: N = {data_all.shape[0]}")

        # Remove subjects with neurological conditions
        neu_code_pd = pd.read_csv(load_resource("icd10_neurological.csv"))
        neu_code = {}
        neu_diagn = {}
        for col_name, col in neu_code_pd.items():
            neu_code[col_name] = col.dropna().to_list()
        all_neu_code = [code for code_list in neu_code.values() for code in code_list]
        for data_i, data_row in data_all.iterrows():
            code_list = []
            for col in field_types["Diagn ICD10"]:
                if data_row[col] != "" and data_row[col] in all_neu_code:
                    code_list.append(data_row[col])
            neu_diagn[data_i] = 1 if code_list else 0
        data_all = data_all.loc[pd.Series(neu_diagn) == 0]
        logger.info(f"Subjects without neurological conditions: N = {data_all.shape[0]}")

        # Brain GMV data (adjusted by TIV) of 200-parcel Schaefer atlas
        data_gmv = pd.read_csv(
            Path(data_dir, "ukb_cat_rois_Schaefer200_17Networks.csv"), index_col="SubjectID")
        data_gmv.index = data_gmv.index.str.slice(start=4)
        data_gmv = data_gmv.loc[
            (data_gmv.index.isin(data_all.index)) & (data_gmv["Session"] == "ses-2")]
        gmv_cols = data_gmv.columns[10:210].to_list()
        data_gmv = data_gmv[gmv_cols].div(data_gmv["TIV"], axis="index")
        data_all = pd.concat([data_all, data_gmv], axis="columns", join="outer")
        field_types["Brain GMV"] = gmv_cols

        # Brain GMV data (adjusted by TIV) of 34-parcel Melbourne subcortex atlas
        data_gmv = pd.read_csv(
            Path(data_dir, "ukb_cat_rois_Tian_7T_S2.csv"), index_col="SubjectID")
        data_gmv.index = data_gmv.index.str.slice(start=4)
        data_gmv = data_gmv.loc[
            (data_gmv.index.isin(data_all.index)) & (data_gmv["Session"] == "ses-2")]
        gmv_cols = data_gmv.columns[10:44].to_list()
        data_gmv = data_gmv[gmv_cols].div(data_gmv["TIV"], axis="index")
        data_all = pd.concat([data_all, data_gmv], axis="columns", join="outer")
        field_types["Brain GMV"].extend(gmv_cols)

        data_all = data_all.rename_axis("eid")
        data_all.to_csv(data_file)

        # Association sample for each phenotype category
        col_type_pheno = ["Body fat", "Brain GMV", "Brain WM", "Blood metabol", "Blood count"]
        cols_req = field_types["Dep sympt"] + field_types["Sociodemo"]
        data_pheno_files = dict.fromkeys(col_type_pheno)
        data_cluster = data_all.copy()
        data_sdem = []
        for col_type in col_type_pheno:
            pheno_name = col_type.replace(" ", "-")
            data_pheno = data_all[cols_req + field_types[col_type]].copy()
            data_pheno = data_pheno.dropna(axis="index", how="any", subset=field_types[col_type])
            data_pheno_files[col_type] = Path(output_data_dir, f"ukb_data_{pheno_name}.csv")
            data_pheno.to_csv(data_pheno_files[col_type])
            logger.info(f"Subjects with all data for {col_type}: N = {data_pheno.shape[0]}")
            data_cluster = data_cluster.drop(data_pheno.index, errors="ignore")
            data_sdem.append(data_pheno)

        # Clustering sample
        data_cluster_file = Path(output_data_dir, "ukb_data_cluster.csv")
        data_cluster.to_csv(data_cluster_file)
        logger.info(f"Clustering sample: N = {data_cluster.shape[0]}")

        # Sociodemographic analysis sample
        data_sdem = pd.concat(data_sdem, axis="index", join="inner")
        data_sdem = data_sdem.drop_duplicates()
        data_sdem = data_sdem.dropna(axis="index", how="any", subset=field_types["Sociodemo"])
        data_sdem_file = Path(output_data_dir, "ukb_data_sdem.csv")
        data_sdem.to_csv(data_sdem_file)
        logger.info(f"Sociodemographic analysis sample: N = {data_sdem.shape[0]}")

        return data_file, data_pheno_files, data_cluster_file, data_sdem_file, field_types


@python.define
class SociodemoData(python.Task["CanonicalPythonTask.Outputs"]):
    """Sociodemographic data preparation for plotting relations

    Args:
        output_dir: Output directory
        data_sdem_file: Extracted data for sociodemograhpic analysis
        dep_score_file: Sum score of depressive symptom clusters for all subjects
    """
    output_dir: Path
    data_sdem_file: Path
    dep_score_file: Path

    class Outputs(python.Outputs):
        """
        Args:
            sdem_file: Sociodemographic data for plotting
        """
        sdem_file: Path

    @staticmethod
    def function(output_dir, data_sdem_file, dep_score_file):
        output_sdem_dir = Path(output_dir, "sociodemo")
        output_sdem_dir.mkdir(parents=True, exist_ok=True)

        # Sociodemographic sample data
        cols_sdem = ["21003-2.0", "31-0.0", "6138-2.0", "738-2.0"]
        col_dtypes = {col: float for col in cols_sdem} | {"eid": str}
        data = pd.read_csv(
            data_sdem_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")

        # Unravel numerical categorical data
        gender_dict = {0: "Female", 1: "Male"}
        educ_dict = {
            -7: "None of the above", 1: "College or University degree",
            2: "A levels/AS levels or equivalent", 3: "O levels/GCSEs or equivalent",
            4: "CSEs or equivalent", 5: "NVQ or NHD or HNC or equivalent",
            6: "Other professional qualifications"}
        income_dict = {
            1: "Less than 18,000", 2: "18,000 to 30,999", 3: "31,000 to 51,999",
            4: "52,000 to 100,000",
            5: "Greater than 100,000"}

        data_sdem = pd.DataFrame(index=data.index)
        data_sdem = data_sdem.assign(age=data["21003-2.0"])
        data_sdem = data_sdem.assign(sex=data["31-0.0"].apply(lambda x: gender_dict.get(x)))
        data_sdem = data_sdem.assign(educ=data["6138-2.0"].apply(lambda x: educ_dict.get(x)))
        data_sdem = data_sdem.assign(income=data["738-2.0"].apply(lambda x: income_dict.get(x)))

        # Depressive sum scores
        cols_dep = ["Sum score 1", "Sum score 2"]
        col_dtypes = {col: float for col in cols_dep} | {"eid": str}
        dep_score = pd.read_csv(
            dep_score_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
        data = pd.concat([data, dep_score], axis="columns", join="inner")

        data_sdem = data_sdem.assign(mood=data["Sum score 1"])
        data_sdem = data_sdem.assign(energy=data["Sum score 2"])

        sdem_file = Path(output_sdem_dir, "ukb_data_sdem_plot.csv")
        data_sdem.to_csv(sdem_file)

        return sdem_file
