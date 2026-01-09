# Characterisation of depressive symptom dimensions in UK Biobank

## Reference

t.b.a.

## Usage

The repository contains all data and code necessary for replication except the raw data from 
UK Biobank. To install the repository:

```bash
python3 -m pip install  git+ssh://git@github:/jadecci/ukb_dep_immunometabol.git
```

The main function that implements the analysis workflow can be called on command line. 
To see the usage:

```bash
ukb_dep -h
```

### Replication

The workflow requires the raw data from UK Biobank to be organised as follows:

```console
rep_data
├── ukb_gmv_CAT12.8_Schaefer400.csv
├── ukb_raw.tsv
└── withdrawn_subjects.csv
```

These correspond to:
1. Gray matter volume data computed using CAT 12.8, parcellated in the Schaefer 400-parcel atlas
2. Raw phenotype data from UK Biobank in tsv format
3. List of withdrawn subject IDs to exclude

Then, run the workflow with:

```bash
ukb_dep --data_dir rep_data
```



