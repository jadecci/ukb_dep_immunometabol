from importlib.resources import files
from pathlib import Path

import ukb_dep_immunometabol


def load_resource(file_name: str) -> Path:
    resource_file = files(ukb_dep_immunometabol) / "data" / file_name
    return Path(str(resource_file))
