# Required Data Files

Place the following raw CSV files in this directory (`dataset/`). These are not committed to Git.

- microbiology_cultures_cohort.csv
- microbiology_cultures_demographics.csv
- microbiology_cultures_ward_info.csv
- microbiology_cultures_labs.csv
- microbiology_cultures_vitals.csv
- microbiology_cultures_comorbidity.csv
- microbiology_cultures_antibiotic_class_exposure.csv
- microbiology_cultures_antibiotic_subtype_exposure.csv
- microbiology_culture_prior_infecting_organism.csv
- microbiology_cultures_priorprocedures.csv
- microbiology_cultures_nursing_home_visits.csv
- microbiology_cultures_adi_scores.csv
- microbiology_cultures_microbial_resistance.csv

After running `preprocessing/build_dl_features.py`, this folder will also contain `amr_analysis_bundle.joblib` (also Git-ignored).
