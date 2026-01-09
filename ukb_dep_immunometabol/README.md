## Usage & Replication
### 1. Preparation

Download/copy the aggregated raw data tsv files to `ukb_raw.tsv`. 
Then set up the project virtual environment:

```bash
python3 -m venv ~/.venvs/ukb_dep && source ~/.venvs/ukb_dep/bin/activate
datalad clone git@github.com:jadecci/ukb_dep_immunometabol.git ukb_dep_immunometabol
python3 -m pip install ukb_dep_immunometabol
```

### 2. Data Extraction

Extract training and test data:

```bash
python3 ukb_dep_immunometabol/ukb_dep_immunometabol/ukbdep_extract_data.py \
  --raw_tsv ukb_raw_tsv \
  --sel_csv ukb_dep_immunometabol/data/UKB_selected_fields_pheno.csv \
  --wd_csv ukb_dep_immunometabolreplication_data/w41655_20250818.csv \
  --neu_code ukb_dep_immunometabol/data/icd10_neurological.csv \
  --icd_code_csv ukb_dep_immunometabol/data/icd_10_level_map.csv \
  --out_dir results/extracted_data \
```

Cluster depressive scores and compute sum scores by cluster:

```bash
python3 ukb_dep_immunometabol/ukb_dep_immunometabol/ukbdep_cluster.py \
  --data_dir results/extracted_data \
  --sel_csv ukb_deppred/data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots
```

### 3. Association analyses

Association between sociodemographic factors and depressive sum scores:

```bash
python3 uukb_dep_immunometabol/ukb_dep_immunometabol/ukbdep_sociodemo.py \
  --data_csv results/extracted_data/ukb_data_sociodemo_clusters.csv \
  --sel_csv ukb_deppred/data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots
```

Correlation between body/brain biomarkers and depressive sum scores:

```bash
python3 ukb_dep_immunometabol/ukb_dep_immunometabol/ukbdep_corr_pheno.py \
  --data_dir results/extracted_data \
  --sel_csv ukb_deppred/data/UKB_selected_fields_pheno.csv \
  --img_dir results/plots \
  --out_dir results/corr_dep
```

Also plot the brain correlations:

```bash
python3 ukb_deppred/ukbdep_corr_brain.py --results_dir results/corr_dep \
  --data_dir ukb_deppred/data \
  --img_dir results/raw_plots
```
