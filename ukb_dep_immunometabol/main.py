from pathlib import Path
import argparse

from pydra.compose import workflow
from pydra.engine.submitter import Submitter
from pydra.utils import print_help, plot_workflow

from ukb_dep_immunometabol.tasks.analysis import Clustering, PLS
from ukb_dep_immunometabol.tasks.data import ExtractData, SociodemoData
from ukb_dep_immunometabol.tasks.plotting import (
    PlotClusterEval, PlotHieraCluster, PlotSociodemo, PlotPLSRadar)


@workflow.define
class UKBDepWf(workflow.Task["CanonicalWorkflowTask.Outputs"]):

    data_dir: Path
    output_dir: Path

    @staticmethod
    def constructor(data_dir, output_dir):
        extract_data = workflow.add(ExtractData(data_dir=data_dir, output_dir=output_dir))

        cluster= workflow.add(Clustering(
            output_dir=output_dir, data_file=extract_data.data_file,
            data_cluster_file=extract_data.data_cluster_file))
        plot_evals = workflow.add(PlotClusterEval(
            output_dir=output_dir, evals_file=cluster.evals_file))
        plot_clusters = workflow.add(PlotHieraCluster(
            output_dir=output_dir, link_file=cluster.link_file))

        sdem = workflow.add(SociodemoData(
            output_dir=output_dir, data_sdem_file=extract_data.data_sdem_file,
            dep_score_file=cluster.dep_score_file))
        plot_sdem = workflow.add(PlotSociodemo(output_dir=output_dir, sdem_file=sdem.sdem_file))

        pls = workflow.add(PLS(
            output_dir=output_dir, data_pheno_files=extract_data.data_pheno_files,
            field_types=extract_data.field_types, dep_score_file=cluster.dep_score_file))
        plot_pls = workflow.add(PlotPLSRadar(output_dir=output_dir, pls_file=pls.pls_file))

        return (
            plot_evals.eval_plot, plot_clusters.cluster_plot, plot_sdem.sdem_plot,
            plot_pls.pls_radar_plots)

    class Outputs(workflow.Outputs):
        eval_plot: Path
        cluster_plot: Path
        sdem_plot: Path
        pls_radar_plots: list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterisation of depressive symptom dimensions in UK Biobank",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory containing restricted UKB data")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--work_dir", type=Path, default=None, help="Cache directory for Pydra")
    config = vars(parser.parse_args())

    config["output_dir"].mkdir(parents=True, exist_ok=True)

    print_help(UKBDepWf)
    plot_workflow(UKBDepWf, out_dir=config["output_dir"], export="png")#, plot_type="detailed")

    wf = UKBDepWf(data_dir=config["data_dir"], output_dir=config["output_dir"])
    with Submitter(worker="cf", cache_root=config["work_dir"]) as submitter:
        output = submitter(wf)
    print(output)


if __name__ == "__main__":
    main()
