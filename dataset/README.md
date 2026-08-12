# Required Data Files

Download the required data files from [Dryad Dataset: doi:10.5061/dryad.jq2bvq8kp](https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp) and move them to this folder (`dataset/`). 
Please ensure you use the files dated **Oct 22 2025**.

These are the required files : 

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
