from pathlib import Path
import argparse

from pydra.compose import workflow
from pydra.utils import show_workflow


@workflow.define
def ukbdep_wf():
    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterisation of depressive symptom dimensions in UK Biobank",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory containing restricted UKB data")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    config = vars(parser.parse_args())

    show_workflow(ukbdep_wf)


if __name__ == "__main__":
    main()
