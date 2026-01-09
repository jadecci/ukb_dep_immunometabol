from importlib.resources import files
from pathlib import Path

from pydra.compose import python
import pandas as pd

import ukb_dep_immunometabol


def load_resource(file_name: str) -> Path:
    resource_file = files(ukb_dep_immunometabol) / "data" / file_name
    return Path(str(resource_file))


@python.define
def init_data(data_dir: Path) -> (dict, dict):
    # Field information
    field_info = {
        "field_dict": {"Diagn ICD10": []},
        "col_dtypes": {"eid": str, "26521-2.0": float, "MDD diagnosis": float},
        "excludes": {}}
    field_cols = {
        "Field ID": str, "Type": str, "Instance": "Int64", "To Exclude 1": float,
        "To Exclude 2": float, "Short Name": str, "Field Description": str, "Notes": str}
    field_file = load_resource("ukb_selected_fileds.csv")
    field_data = pd.read_csv(field_file, usecols=list(field_cols.keys()), dtype=field_cols)
    for _, field_row in field_data.iterrows():
        col_id = f"{field_row['Field ID']}-{field_row['Instance']}.0"
        if field_row["Type"] in field_info["field_dict"].keys():
            field_info["field_dict"][field_row["Type"]].append(col_id)
        else:
            field_info["field_dict"][field_row["Type"]] = [col_id]
        field_info["col_dtypes"][col_id] = float

    data_files = {
        "": []
    }
    return field_info, data_files


@python.define
def prepare_data():
    encoding = "ISO-8859-1"
    chunksize = 1000
    field_dict = {"Diagn ICD10": []}
    col_dtypes = {"eid": str, "26521-2.0": float}
    excludes = {}
    return
