import pandas as pd
import numpy as np
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    print("=== Building DL feature bundle ===")

    # ---------------------------------------------------------
    # 1. Primary Cohort Loading
    # ---------------------------------------------------------
    print("Loading microbiology_cultures_cohort.csv...")
    cohort = pd.read_csv("../dataset/microbiology_cultures_cohort.csv")
    MERGE_KEY = 'order_proc_id_coded'

    df = cohort.copy()
    df = df[df['susceptibility'].isin(['Susceptible', 'Resistant'])].copy()
    df['label'] = (df['susceptibility'] == 'Resistant').astype(int)

    print(f"Initial cohort filtered to Susceptible/Resistant. Shape: {df.shape}")

    del cohort
    gc.collect()

    df['culture_type_enc'] = df['culture_description'].map({'URINE': 0, 'BLOOD': 1, 'RESPIRATORY': 2}).fillna(-1).astype(int)
    df['ordering_mode_enc'] = df['ordering_mode'].map({'Inpatient': 0, 'Outpatient': 1, 'Null': 2}).fillna(2).astype(int)

    org_le = LabelEncoder()
    df['organism_enc'] = org_le.fit_transform(df['organism'].fillna('Unknown'))

    ab_le = LabelEncoder()
    df['antibiotic_enc'] = ab_le.fit_transform(df['antibiotic'].fillna('Unknown'))

    df['order_time'] = pd.to_datetime(df['order_time_jittered_utc'], utc=True, errors='coerce')
    df['order_year']  = df['order_time'].dt.year.fillna(2020).astype(int)
    df['order_month'] = df['order_time'].dt.month.fillna(1).astype(int)

    # --- Demographics ---
    print("Merging demographics...")
    demographics = pd.read_csv("../dataset/microbiology_cultures_demographics.csv")
    age_map = {'18-24 years': 1, '25-34 years': 2, '35-44 years': 3,
               '45-54 years': 4, '55-64 years': 5, '65-74 years': 6,
               '75-84 years': 7, '85-89 years': 8, '90+ years': 9}
    demo_feat = demographics[[MERGE_KEY, 'age', 'gender']].copy()
    demo_feat = demo_feat.drop_duplicates(MERGE_KEY)
    demo_feat['age_enc']    = demo_feat['age'].map(age_map).fillna(5).astype(int)
    demo_feat['gender_enc'] = pd.to_numeric(demo_feat['gender'], errors='coerce').fillna(0).astype(int)
    demo_feat = demo_feat[[MERGE_KEY, 'age_enc', 'gender_enc']]

    df = df.merge(demo_feat, on=MERGE_KEY, how='left')
    df['age_enc']    = df['age_enc'].fillna(5).astype(int)
    df['gender_enc'] = df['gender_enc'].fillna(0).astype(int)

    del demographics, demo_feat
    gc.collect()

    # --- Ward Info ---
    print("Merging ward info...")
    ward_info = pd.read_csv("../dataset/microbiology_cultures_ward_info.csv")
    ward_feat = ward_info[[MERGE_KEY, 'hosp_ward_IP', 'hosp_ward_OP', 'hosp_ward_ER', 'hosp_ward_ICU']].copy()
    ward_feat = ward_feat.drop_duplicates(MERGE_KEY)

    df = df.merge(ward_feat, on=MERGE_KEY, how='left')
    for col in ['hosp_ward_IP', 'hosp_ward_OP', 'hosp_ward_ER', 'hosp_ward_ICU']:
        df[col] = df[col].fillna(0).astype(int)

    del ward_info, ward_feat
    gc.collect()

    # --- Labs ---
    print("Merging labs...")
    labs = pd.read_csv("../dataset/microbiology_cultures_labs.csv")
    lab_cols = ['median_wbc', 'median_neutrophils', 'median_lymphocytes',
                'median_hgb', 'median_plt', 'median_na', 'median_hco3',
                'median_bun', 'median_cr', 'median_lactate', 'median_procalcitonin',
                'first_wbc', 'last_wbc', 'first_cr', 'last_cr',
                'first_lactate', 'last_lactate', 'first_procalcitonin', 'last_procalcitonin']

    available_lab_cols = [c for c in lab_cols if c in labs.columns]
    lab_feat = labs[[MERGE_KEY] + available_lab_cols].copy()
    lab_feat = lab_feat.drop_duplicates(MERGE_KEY)

    for col in available_lab_cols:
        lab_feat[col] = pd.to_numeric(lab_feat[col].replace('Null', np.nan), errors='coerce')

    df = df.merge(lab_feat, on=MERGE_KEY, how='left')
    for col in available_lab_cols:
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != 'object' else 0)

    del labs, lab_feat
    gc.collect()

    # --- Vitals ---
    print("Merging vitals...")
    vitals = pd.read_csv("../dataset/microbiology_cultures_vitals.csv")
    vital_cols = ['median_heartrate', 'median_resprate', 'median_temp',
                  'median_sysbp', 'median_diasbp',
                  'first_heartrate', 'last_heartrate',
                  'first_sysbp', 'last_sysbp', 'first_temp', 'last_temp']

    available_vital_cols = [c for c in vital_cols if c in vitals.columns]
    vital_feat = vitals[[MERGE_KEY] + available_vital_cols].copy()
    vital_feat = vital_feat.drop_duplicates(MERGE_KEY)

    for col in available_vital_cols:
        vital_feat[col] = pd.to_numeric(vital_feat[col].replace('Null', np.nan), errors='coerce')

    df = df.merge(vital_feat, on=MERGE_KEY, how='left')
    for col in available_vital_cols:
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != 'object' else 0)

    del vitals, vital_feat
    gc.collect()

    # ---------------------------------------------------------
    # 2. INLINE Comorbidity Processing (No Chunking, No File Saving)
    # ---------------------------------------------------------
    print("Processing and merging comorbidities inline...")
    comorb_raw = pd.read_csv('../dataset/microbiology_cultures_comorbidity.csv', low_memory=False)

    comorb_active = comorb_raw[comorb_raw['comorbidity_component_start_days_culture'] >= 0].copy()
    del comorb_raw
    gc.collect()

    comorb_active['comorb_col'] = (
        comorb_active['comorbidity_component']
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.strip('_')
        .apply(lambda x: 'comorb_' + x)
    )

    comorb_count = (
        comorb_active.groupby(MERGE_KEY)['comorbidity_component']
        .nunique()
        .reset_index()
        .rename(columns={'comorbidity_component': 'comorb_total_count'})
    )

    comorb_pivot = pd.get_dummies(
        comorb_active[[MERGE_KEY, 'comorb_col']].drop_duplicates(),
        columns=['comorb_col'],
        prefix='',
        prefix_sep=''
    )
    del comorb_active
    gc.collect()

    comorbidity_df = comorb_pivot.groupby(MERGE_KEY).max().reset_index()
    del comorb_pivot

    COMORB_COLS = [c for c in comorbidity_df.columns if c != MERGE_KEY]
    comorbidity_df = comorbidity_df.merge(comorb_count, on=MERGE_KEY, how='left')

    for col in COMORB_COLS:
        comorbidity_df[col] = comorbidity_df[col].fillna(0).astype(np.int8)
    comorbidity_df['comorb_total_count'] = comorbidity_df['comorb_total_count'].fillna(0).astype(int)

    df = df.merge(comorbidity_df, on=MERGE_KEY, how='left')

    print(f"Comorb processing complete. Added {len(COMORB_COLS)} binary metrics.")
    del comorbidity_df
    gc.collect()

    # --- Antibiotic Exposures & Others ---
    print("Merging antibiotic exposures, prior orgs, procedures, etc...")
    ab_class = pd.read_csv("../dataset/microbiology_cultures_antibiotic_class_exposure.csv")
    if 'antibiotic_class' in ab_class.columns:
        ab_class_pivot = pd.get_dummies(ab_class[[MERGE_KEY, 'antibiotic_class']].drop_duplicates(), columns=['antibiotic_class'], prefix='abclass')
        ab_class_agg = ab_class_pivot.groupby(MERGE_KEY).max().reset_index()
        df = df.merge(ab_class_agg, on=MERGE_KEY, how='left')
        ab_class_cols = [c for c in ab_class_agg.columns if c != MERGE_KEY]
        for col in ab_class_cols: df[col] = df[col].fillna(0).astype(int)
    del ab_class; gc.collect()

    ab_subtype = pd.read_csv("../dataset/microbiology_cultures_antibiotic_subtype_exposure.csv")
    if 'antibiotic_subtype_category' in ab_subtype.columns:
        ab_sub_pivot = pd.get_dummies(ab_subtype[[MERGE_KEY, 'antibiotic_subtype_category']].drop_duplicates(), columns=['antibiotic_subtype_category'], prefix='absub')
        ab_sub_agg = ab_sub_pivot.groupby(MERGE_KEY).max().reset_index()
        df = df.merge(ab_sub_agg, on=MERGE_KEY, how='left')
        ab_sub_cols = [c for c in ab_sub_agg.columns if c != MERGE_KEY]
        for col in ab_sub_cols: df[col] = df[col].fillna(0).astype(int)
    del ab_subtype; gc.collect()

    prior_org = pd.read_csv("../dataset/microbiology_culture_prior_infecting_organism.csv")
    if 'prior_organism' in prior_org.columns:
        prior_org_pivot = pd.get_dummies(prior_org[[MERGE_KEY, 'prior_organism']].drop_duplicates(), columns=['prior_organism'], prefix='priororg')
        prior_org_agg = prior_org_pivot.groupby(MERGE_KEY).max().reset_index()
        df = df.merge(prior_org_agg, on=MERGE_KEY, how='left')
        prior_org_cols = [c for c in prior_org_agg.columns if c != MERGE_KEY]
        for col in prior_org_cols: df[col] = df[col].fillna(0).astype(int)
    del prior_org; gc.collect()

    procedures = pd.read_csv("../dataset/microbiology_cultures_priorprocedures.csv")
    if 'procedure_description' in procedures.columns:
        proc_pivot = pd.get_dummies(procedures[[MERGE_KEY, 'procedure_description']].drop_duplicates(), columns=['procedure_description'], prefix='proc')
        proc_agg = proc_pivot.groupby(MERGE_KEY).max().reset_index()
        df = df.merge(proc_agg, on=MERGE_KEY, how='left')
        proc_cols = [c for c in proc_agg.columns if c != MERGE_KEY]
        for col in proc_cols: df[col] = df[col].fillna(0).astype(int)
    del procedures; gc.collect()

    nursing_home = pd.read_csv("../dataset/microbiology_cultures_nursing_home_visits.csv")
    if 'nursing_home_visit_culture' in nursing_home.columns:
        nh_feat = nursing_home[[MERGE_KEY, 'nursing_home_visit_culture']].copy()
        nh_feat = nh_feat.groupby(MERGE_KEY)['nursing_home_visit_culture'].max().reset_index()
        nh_feat.rename(columns={'nursing_home_visit_culture': 'nursing_home_visits'}, inplace=True)
        df = df.merge(nh_feat, on=MERGE_KEY, how='left')
        df['nursing_home_visits'] = df['nursing_home_visits'].fillna(0).astype(int)
    del nursing_home; gc.collect()

    adi = pd.read_csv("../dataset/microbiology_cultures_adi_scores.csv")
    if 'adi_score' in adi.columns:
        adi_feat = adi[[MERGE_KEY, 'adi_score', 'adi_state_rank']].copy()
        adi_feat = adi_feat.drop_duplicates(MERGE_KEY)
        adi_feat['adi_score']      = pd.to_numeric(adi_feat['adi_score'].replace('Null', np.nan), errors='coerce').fillna(50)
        adi_feat['adi_state_rank'] = pd.to_numeric(adi_feat['adi_state_rank'].replace('Null', np.nan), errors='coerce').fillna(5)
        df = df.merge(adi_feat, on=MERGE_KEY, how='left')
        df['adi_score']      = df['adi_score'].fillna(50)
        df['adi_state_rank'] = df['adi_state_rank'].fillna(5)
    del adi; gc.collect()

    microbial_res = pd.read_csv("../dataset/microbiology_cultures_microbial_resistance.csv")
    if 'resistant_time_to_culturetime' in microbial_res.columns:
        mres_feat = microbial_res[[MERGE_KEY, 'resistant_time_to_culturetime']].copy()
        mres_agg = mres_feat.groupby(MERGE_KEY)['resistant_time_to_culturetime'].agg(['count', 'min']).reset_index()
        mres_agg.columns = [MERGE_KEY, 'prior_resistance_count', 'min_resistance_days']
        df = df.merge(mres_agg, on=MERGE_KEY, how='left')
        df['prior_resistance_count'] = df['prior_resistance_count'].fillna(0)
        df['min_resistance_days']    = df['min_resistance_days'].fillna(9999)
    del microbial_res; gc.collect()

    print(f"Final Dataframe built. Shape before train-test split: {df.shape}")

    # ---------------------------------------------------------
    # 3. Feature Preparation
    # ---------------------------------------------------------
    CAT_FEATURES = ['organism_enc', 'antibiotic_enc', 'culture_type_enc',
                    'ordering_mode_enc', 'age_enc', 'gender_enc',
                    'order_year', 'order_month']

    lab_num_cols   = [c for c in available_lab_cols  if c in df.columns]
    vital_num_cols = [c for c in available_vital_cols if c in df.columns]

    CONT_FEATURES = (lab_num_cols + vital_num_cols +
                     ['adi_score', 'adi_state_rank',
                      'nursing_home_visits',
                      'prior_resistance_count', 'min_resistance_days',
                      'comorb_total_count'])
    CONT_FEATURES = [c for c in CONT_FEATURES if c in df.columns]

    BINARY_FEATURES = (['hosp_ward_IP', 'hosp_ward_OP', 'hosp_ward_ER', 'hosp_ward_ICU'] +
                       COMORB_COLS +
                       [c for c in df.columns if c.startswith('abclass_')] +
                       [c for c in df.columns if c.startswith('absub_')] +
                       [c for c in df.columns if c.startswith('priororg_')] +
                       [c for c in df.columns if c.startswith('proc_')])
    BINARY_FEATURES = [c for c in BINARY_FEATURES if c in df.columns]

    ALL_FEATURES = list(dict.fromkeys(CAT_FEATURES + CONT_FEATURES + BINARY_FEATURES))

    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Carry the merge key alongside the feature matrix so it survives the
    # exact same row filtering/ordering as X/y, with zero extra joins later.
    df_model = df[[MERGE_KEY] + ALL_FEATURES + ['label']].copy()

    for col in CONT_FEATURES:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        df_model[col] = df_model[col].fillna(df_model[col].median())

    for col in CAT_FEATURES:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0).astype(int)

    for col in BINARY_FEATURES:
        if col in COMORB_COLS:
            df_model[col] = df_model[col].fillna(0).astype(np.int8)
        else:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0).astype(int)

    cat_cardinalities = {col: int(df_model[col].max()) + 1 for col in CAT_FEATURES}
    cat_embed_dims = {col: min(50, (card + 1) // 2) for col, card in cat_cardinalities.items()}

    scaler = StandardScaler()
    df_model[CONT_FEATURES] = scaler.fit_transform(df_model[CONT_FEATURES])

    # merge_keys is extracted with the exact same row order as X/y below, so
    # index i in any of these three arrays always refers to the same row.
    merge_keys = df_model[MERGE_KEY].values
    X = df_model[ALL_FEATURES].values.astype(np.float32)
    y = df_model['label'].values.astype(np.float32)

    del df, df_model
    gc.collect()

    # NOTE on reproducibility: random_state=42 + stratify=y only guarantees
    # an identical split if X/y/merge_keys arrive in identical row order on
    # every run. Re-reading and re-merging 10+ raw CSVs is NOT guaranteed to
    # preserve row order across runs (file content changes, pandas version
    # changes, etc.), which is exactly what caused the metric drift earlier.
    # Saving the literal arrays below (see section 6b) removes this risk
    # entirely for all future analysis — the analysis script never needs to
    # call train_test_split again.
    (X_train, X_test,
     y_train, y_test,
     keys_train, keys_test) = train_test_split(
        X, y, merge_keys, test_size=0.15, stratify=y, random_state=42
    )

    del X
    gc.collect()

    (X_train, X_val,
     y_train, y_val,
     keys_train, keys_val) = train_test_split(
        X_train, y_train, keys_train, test_size=0.15, stratify=y_train, random_state=42
    )

    print(f"Split sizes -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    cat_idx  = list(range(len(CAT_FEATURES)))
    cont_idx = list(range(len(CAT_FEATURES), len(CAT_FEATURES) + len(CONT_FEATURES)))
    bin_idx  = list(range(len(CAT_FEATURES) + len(CONT_FEATURES), len(ALL_FEATURES)))

    # ---------------------------------------------------------
    # 4. Save preprocessing bundle for downstream models
    # ---------------------------------------------------------
    # #migrate: save bundle to shared dataset/ folder so LR, XGB and DL use identical splits
    analysis_bundle = {
        'CAT_FEATURES':      CAT_FEATURES,
        'CONT_FEATURES':     CONT_FEATURES,
        'BINARY_FEATURES':   BINARY_FEATURES,
        'ALL_FEATURES':      ALL_FEATURES,
        'cat_idx':           cat_idx,
        'cont_idx':          cont_idx,
        'bin_idx':           bin_idx,
        'cat_cardinalities': cat_cardinalities,
        'cat_embed_dims':    cat_embed_dims,
        'MERGE_KEY':         MERGE_KEY,
        'scaler':            scaler,
        'org_le':            org_le,
        'ab_le':             ab_le,
        'X_train':           X_train,
        'X_val':             X_val,
        'X_test':            X_test,
        'y_train':           y_train,
        'y_val':             y_val,
        'y_test':            y_test,
        'keys_train':        keys_train,
        'keys_val':          keys_val,
        'keys_test':         keys_test,
        'n_features_total':  len(ALL_FEATURES),
    }
    joblib.dump(analysis_bundle, '../dataset/amr_analysis_bundle.joblib', compress=3)
    print("Saved preprocessing bundle -> ../dataset/amr_analysis_bundle.joblib")