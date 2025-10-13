from os import remove
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
        description="Extract all useful data",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--raw_tsv", type=Path, help="Absolute path to UK Biobank raw data tsv file")
parser.add_argument("--sel_csv", type=Path, help="Absolute path to table of selected fields")
parser.add_argument("--wd_csv", type=Path, help="Absolute path to list of withdrawn subjects")
parser.add_argument("--neu_code", type=Path, help="Absolute path to ICD10 neurological code")
parser.add_argument("--icd_code_csv", type=Path, help="Absolute path to ICD10 code mapping csv")
parser.add_argument("--out_dir", type=Path, help="Absolute path to output directory")
args = parser.parse_args()

encoding = "ISO-8859-1"
chunksize = 1000
field_dict = {"Diagn ICD10": []}
col_dtypes = {"eid": str, "26521-2.0": float}
excludes = {}
args.out_dir.mkdir(parents=True, exist_ok=True)

# Selected field information
field_cols = {
    "Field ID": str, "Type": str, "Instance": "Int64", "To Exclude 1": float, "To Exclude 2": float,
    "Notes": str}
fields = pd.read_csv(args.sel_csv, usecols=list(field_cols.keys()), dtype=field_cols)
for _, field_row in fields.iterrows():
    col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
    if field_row["Type"] in field_dict.keys():
        field_dict[field_row["Type"]].append(col_id)
    else:
        field_dict[field_row["Type"]] = [col_id]
    col_dtypes[col_id] = float
    if not np.isnan(field_row["To Exclude 1"]):
        excludes[col_id] = [field_row["To Exclude 1"]]
        if not np.isnan(field_row["To Exclude 2"]):
            excludes[col_id].append(field_row["To Exclude 2"])

# ICD-10 code
data_head = pd.read_table(args.raw_tsv, delimiter="\t", encoding=encoding, nrows=2)
for col in data_head.columns:
    if col.split("-")[0] in "41270":
        field_dict["Diagn ICD10"].append(col)
        col_dtypes[col] = str

# Extract all data
all_data_file = Path(args.out_dir, "ukb_data_all.csv")
if all_data_file.exists():
    remove(all_data_file)
data_iter = pd.read_table(
    args.raw_tsv, delimiter="\t", encoding=encoding, chunksize=chunksize,
    usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
for data_curr in data_iter:
    data_out = data_curr.dropna(axis="index", how="any", subset=field_dict["Dep sympt"])
    if not data_out.empty:
        if all_data_file.exists():
            data_out.to_csv(all_data_file, mode="a", header=False)
        else:
            data_out.to_csv(all_data_file)

# Remove withdrawn subjects
all_data = pd.read_csv(
    all_data_file, usecols=list(col_dtypes.keys()), dtype=col_dtypes, index_col="eid")
print(f"Subjects with all depressive symptom items: N = {all_data.shape[0]}")
wd_sub = pd.read_csv(args.wd_csv, header=None).squeeze()
all_data = all_data.drop(wd_sub, errors="ignore")
print(f"Subjects with non-retracted consent: N = {all_data.shape[0]}")

# Remove subjects with "do not know / prefer not to answer" code (for depressive symptoms)
for col_id, exclude_list in excludes.items():
    if col_id in field_dict["Dep sympt"]:
        for exclude in exclude_list:
            all_data = all_data.loc[all_data[col_id] != exclude]
print(f"Subjects with valid code for depressive symptoms: N = {all_data.shape[0]}")
# Remove subjects with "do not know / prefer not to answer" code (for sociodemographic items)
for col_id, exclude_list in excludes.items():
    if col_id in field_dict["Sociodemo"]:
        for exclude in exclude_list:
            all_data = all_data.loc[all_data[col_id] != exclude]
print(f"Subjects with valid code for sociodemographic items: N = {all_data.shape[0]}")

# Remove subjects who reported no symptoms
data_remove = all_data.copy()
for _, field_df_row in fields.iterrows():
    if field_df_row["Type"] == "Dep sympt":
        col_id = f"{field_df_row['Field ID']}-{field_df_row['Instance']}.0"
        if field_df_row["Notes"] == "N-12":
            data_remove = data_remove.loc[data_remove[col_id] == 0]  # binary
        else:
            data_remove = data_remove.loc[data_remove[col_id] == 1]  # frequency
all_data = all_data.drop(data_remove.index)
print(f"Subjects who reported at least one symptom: N = {all_data.shape[0]}")

# Remove subjects with neurological conditions
neu_code_pd = pd.read_csv(args.neu_code)
neu_code = {}
neu_diagn = {}
for col_name, col in neu_code_pd.items():
    neu_code[col_name] = col.dropna().to_list()
all_neu_code = [code for code_list in neu_code.values() for code in code_list]
for data_i, data_row in all_data.iterrows():
    code_list = []
    for col in field_dict["Diagn ICD10"]:
        if data_row[col] != "" and data_row[col] in all_neu_code:
            code_list.append(data_row[col])
    neu_diagn[data_i] = 1 if code_list else 0
all_data = all_data.loc[pd.Series(neu_diagn) == 0]
print(f"Subjects without neurological conditions: N = {all_data.shape[0]}")

# Normalise GMV by subject-wise TIV
all_data[field_dict["Brain GMV"]] = all_data[field_dict["Brain GMV"]].div(
    all_data["26521-2.0"], axis="index")

# Add MDD diagnosis column
code_map = pd.read_csv(args.icd_code_csv)
code_list = []
dep_diagn = {}
for _, block_row in code_map.loc[code_map["Level"] == 2].iterrows():
    if block_row["Coding"] in ["F32", "F33", "F34", "F38", "F39"]:
        if block_row["Selectable"] == "Yes":
            code_list.append(block_row["Coding"])
        else:
            code_next = code_map.loc[code_map["Parent"] == str(block_row["Node"])]
            for _, code_row in code_next.iterrows():
                code_list.append(code_row["Coding"])
for data_i, data_row in all_data.iterrows():
    code_list_curr = []
    for col in field_dict["Diagn ICD10"]:
        if data_row[col] != "" and data_row[col] in code_list:
            code_list_curr.append(data_row[col])
    dep_diagn[data_i] = 1 if code_list_curr else 0
all_data["MDD diagnosis"] = pd.Series(dep_diagn)
print(f"Subjects with MDD diagnosis: N = {all_data.loc[all_data['MDD diagnosis'] == 1].shape[0]}")
all_data.to_csv(all_data_file)

# Define sample for each phenotype category
field_req = field_dict["Dep sympt"] + field_dict["Sociodemo"] + ["MDD diagnosis"]
col_type_pheno = ["Body fat", "Brain GMV", "Brain WM", "Blood metabol", "Blood count"]
data_cluster = all_data.copy()
for col_type in col_type_pheno:
    pheno_name = col_type.replace(" ", "-")
    field_req_curr = field_dict[col_type] + field_req
    data_curr = all_data[field_req_curr].copy()
    data_pheno = data_curr.dropna(axis="index", how="any", subset=field_req_curr)
    data_pheno.to_csv(Path(args.out_dir, f"ukb_data_{pheno_name}.csv"))
    print(f"Subjects with all phenotypes for {col_type}: N = {data_pheno.shape[0]}")
    data_cluster = data_cluster.drop(data_pheno.index, errors="ignore")
data_cluster.to_csv(Path(args.out_dir, "ukb_data_cluster.csv"))
print(f"Training sample: N = {data_cluster.shape[0]}")
