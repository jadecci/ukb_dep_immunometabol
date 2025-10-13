from pathlib import Path
import argparse

import nibabel as nib
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_glass_brain, plot_surf_stat_map
from nilearn.surface import load_surf_data
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(
        description="Plot correlations between depressive sum scores and brain biomarkers",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--results_dir", type=Path, help="Absolute path to results directory")
parser.add_argument("--data_dir", type=Path, help="Absolute path to data directory")
parser.add_argument("--img_dir", type=Path, help="Absolute path to output plots directory")
args = parser.parse_args()

args.img_dir.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(Path(args.results_dir, "ukb_dep_corr_pheno_fdr.csv"))
dep_scores = ["Sum score (cluster 5)", "Sum score (cluster 6)"]

# JHU atlas
jhu_img = nib.load(Path(args.data_dir, "JHU-ICBM-labels-1mm.nii.gz"))
jhu_vol = jhu_img.get_fdata()
jhu_lut = pd.read_csv(Path(args.data_dir, "JHU_labels.csv"))

# fsaverage surface and Destrieux 2009 atlas
fs_meshes = load_fsaverage(mesh="fsaverage")
fs_bg = load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="sulcal")
l_annot = load_surf_data(Path(args.data_dir, "lh.aparc.a2009s.annot"))
r_annot = load_surf_data(Path(args.data_dir, "rh.aparc.a2009s.annot"))
fs_lut = pd.read_csv(Path(args.data_dir, "FS_a2009s_labels.csv"))

# FreeSurfer aseg segmentation in MNI space
aseg_img = nib.load(Path(args.data_dir, "mni_aseg.mgz"))
aseg_vol = aseg_img.get_fdata()
aseg_lut = pd.read_csv(Path(args.data_dir, "FS_aseg_labels.csv"))

for dep_i, dep in enumerate(dep_scores):
    for gender in ["female", "male"]:
        data_curr = data.loc[(data["Gender"] == gender) & (data["Depressive score field"] == dep)]

        # White matter biomarkers
        for wm in ["FA", "MD", "ICVF", "ISOVF", "OD"]:
            data_wm = data_curr.loc[data_curr["Field description"].str.contains(f"Mean {wm}")]
            vol_curr = np.zeros(jhu_vol.shape)
            for _, data_row in data_wm.iterrows():
                wm_region = data_row["Field description"].split("in ")[1]
                wm_region_val = jhu_lut["Atlas value"].loc[jhu_lut["Brain region"] == wm_region]
                vol_curr[np.where(jhu_vol == wm_region_val.values[0])] = data_row["r"]

            wm_img = nib.Nifti1Image(vol_curr, jhu_img.affine)
            plt = plot_glass_brain(
                wm_img, colorbar=True, cmap="coolwarm", vmax=0.1, symmetric_cbar=True,
                plot_abs=False, threshold=0.01)
            plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_{wm}_{gender}.png"))
            plt.close()

        # Gray matter volume (cortical) biomarkers
        data_gmv = data_curr.loc[
            (data_curr["Type"] == "Brain GMV") & (data_curr["Data field"].str[:2] == "27")]
        fs_surf_l = np.zeros(l_annot.shape)
        fs_surf_r = np.zeros(r_annot.shape)
        for _, data_row in data_gmv.iterrows():
            fs_region = data_row["Field description"].split("Volume of ")[1]
            if "left" in fs_region:
                fs_region_val = fs_lut["Atlas value"].loc[fs_lut["Left brain region"] == fs_region]
                fs_surf_l[np.where(l_annot == fs_region_val.values[0])] = data_row["r"]
            if "right" in fs_region:
                fs_region_val = fs_lut["Atlas value"].loc[fs_lut["Right brain region"] == fs_region]
                fs_surf_r[np.where(r_annot == fs_region_val.values[0])] = data_row["r"]

        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_l, bg_map=fs_bg, hemi="left",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01,
            view="lateral")
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_gmv_{gender}_ll.png"))
        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_l, bg_map=fs_bg, hemi="left",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01,
            view="medial")
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_gmv_{gender}_lm.png"))
        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_r, bg_map=fs_bg, hemi="right",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01,
            view="lateral")
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_gmv_{gender}_rl.png"))
        plt = plot_surf_stat_map(
            surf_mesh=fs_meshes["inflated"], stat_map=fs_surf_r, bg_map=fs_bg, hemi="right",
            cmap="coolwarm", colorbar=True, vmax=0.1, symmetric_cbar=True, threshold=0.01,
            view="medial")
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_gmv_{gender}_rm.png"))

        # Gray matter volume (subcortical) biomarkers
        data_gmv = data_curr.loc[
            (data_curr["Type"] == "Brain GMV") & (data_curr["Data field"].str[:2] == "26")]
        vol_curr = np.zeros(aseg_vol.shape)
        for _, data_row in data_gmv.iterrows():
            aseg_region = data_row["Field description"].split("Volume of ")[1]
            aseg_region_val = aseg_lut["Atlas value"].loc[aseg_lut["Brain region"] == aseg_region]
            vol_curr[np.where(aseg_vol == aseg_region_val.values[0])] = data_row["r"]

        gmv_img = nib.Nifti1Image(vol_curr, aseg_img.affine)
        plt = plot_glass_brain(
            gmv_img, colorbar=True, cmap="coolwarm", vmax=0.1, symmetric_cbar=True,
            plot_abs=False, threshold=0.01)
        plt.savefig(Path(args.img_dir, f"corr_dep{dep_i}_gmv_subcort_{gender}.png"))
        plt.close()
