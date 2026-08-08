"""
00_common.py
-------------
RUN THIS FILE FIRST, ONCE.

This is your ORIGINAL Phase 1-9 feature engineering, UNCHANGED, followed by a
single fit_transform of the preprocessor and a single train/test split.

It caches:
  ./cache/X_train.npz     (sparse, preprocessed)
  ./cache/X_test.npz      (sparse, preprocessed)
  ./cache/y_train.npy
  ./cache/y_test.npy
  ./cache/preprocessor.joblib
  ./cache/meta.json

Every other script (01-07) imports `load_cached_split()` from common.py
(a separate, importable module — Python module names can't start with a
digit, which is why the loader function itself lives in common.py and this
00_ file just drives the one-time build) and loads these cached arrays
instead of re-reading your CSVs and re-running one-hot encoding from
scratch. That's the single biggest speed win across this whole exercise —
feature engineering happens once, not 7 times.

If the cache already exists, this script just confirms it and exits fast.
"""

import pandas as pd
import numpy as np
import re
import gc
import os
import sys
import json
import joblib
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from common import (
    CACHE_DIR, X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH,
    PREPROCESSOR_PATH, META_PATH, load_cached_split,
)

os.makedirs(CACHE_DIR, exist_ok=True)

# #migrate: load split/seed settings from the single config file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_XGB_IMB_CFG = _CFG.get('xgboost_imbalance', {})
SEED = _CFG.get('seed', 42)
TEST_SIZE = _XGB_IMB_CFG.get('test_size', 0.2)
RANDOM_STATE = _XGB_IMB_CFG.get('random_state', SEED)
DATASET_DIR = str(resolve_path(_XGB_IMB_CFG.get('dataset_dir', 'dataset')))

_cache_exists = all(
    os.path.exists(p)
    for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, PREPROCESSOR_PATH, META_PATH]
)

if _cache_exists:
    print("[LOG] Cached preprocessed split already exists — skipping feature engineering.")
    print(f"      Run with FORCE_REBUILD=True at the top of this file to regenerate.")
else:
    print("[LOG] No cache found. Running full feature engineering pipeline (Phases 1-9)...")

    # ==========================================================================
    # PHASE 1: LOAD AND PREPARE COHORT
    # ==========================================================================
    print("[LOG] Loading and Preparing Cohort Data...")
    # #migrate: dataset path from config (user copies files into data folder)
    df_cohort = pd.read_csv(os.path.join(DATASET_DIR, "microbiology_cultures_cohort.csv"))

    valid_outcomes = ['Susceptible', 'Resistant', 'Intermediate']
    df_cohort = df_cohort[df_cohort['susceptibility'].isin(valid_outcomes)]

    df_cohort['is_resistant'] = df_cohort['susceptibility'].apply(
        lambda x: 0 if x == 'Susceptible' else 1
    )

    df_cohort = df_cohort.drop(columns=[
        'susceptibility',
        'was_positive',
        'pat_enc_csn_id_coded',
        'anon_id',
        'order_time_jittered_utc',
    ])

    print(f"[LOG] Initial Cohort size: {len(df_cohort)}")

    # ==========================================================================
    # PHASE 2: GENERATE ANTIBIOTIC FEATURES
    # ==========================================================================
    print("[LOG] Generating Antibiotic Features...")
    class_mapping = {
        'Beta Lactam': 'Beta_Lactam',
        'Monobactam': 'Beta_Lactam',
        'Combination Antibiotic': 'Beta_Lactam',
        'Fluoroquinolone': 'Fluoroquinolone',
        'Glycopeptide': 'MRSA_VRE_Active',
        'Oxazolidinone': 'MRSA_VRE_Active',
        'Aminoglycoside': 'Severe_GramNeg',
        'Polymyxin, Lipopeptide': 'Severe_GramNeg',
        'Macrolide Lincosamide': 'Community_Resp',
        'Tetracycline': 'Community_Resp',
        'Ansamycin': 'Community_Resp',
        'Antitubercular': 'Community_Resp',
        'Sulfonamide': 'UTI_Sulfa',
        'Folate Synthesis Inhibitor': 'UTI_Sulfa',
        'Nitrofuran': 'UTI_Sulfa',
        'Fosfomycin': 'UTI_Sulfa',
        'Urinary Antiseptic': 'UTI_Sulfa',
        'Nitroimidazole': 'Anaerobes'
    }

    def generate_abx_features_optimized(df):
        df = df.copy()
        df['super_class'] = df['antibiotic_class'].map(class_mapping)
        df = df.dropna(subset=['super_class'])
        df['recency_weight'] = np.exp(-df['time_to_culturetime'] / 30)
        df['recent_flag'] = (df['time_to_culturetime'] <= 30).astype(int)

        grouped = df.groupby(['order_proc_id_coded', 'super_class'])
        features = grouped.agg(
            weighted_exposure=('recency_weight', 'sum'),
            time_since_last=('time_to_culturetime', 'min'),
        ).reset_index()

        features_wide = features.pivot(index='order_proc_id_coded', columns='super_class')
        features_wide.columns = [f"{cls}_{feat}" for feat, cls in features_wide.columns]
        features_wide = features_wide.reset_index()

        global_features = df.groupby('order_proc_id_coded').agg(
            total_abx_count=('super_class', 'count'),
            abx_diversity=('super_class', 'nunique'),
            recent_any_abx=('recent_flag', 'max')
        ).reset_index()

        broad_classes = ['Beta_Lactam', 'Fluoroquinolone', 'Severe_GramNeg']
        df['broad_flag'] = df['super_class'].isin(broad_classes).astype(int)

        broad_feature = df.groupby('order_proc_id_coded').agg(
            broad_spectrum_count=('broad_flag', 'sum')
        ).reset_index()

        final = (
            features_wide
            .merge(global_features, on='order_proc_id_coded', how='left')
            .merge(broad_feature, on='order_proc_id_coded', how='left')
        )
        return final.fillna(0)

    # #migrate: dataset path from config (user copies files into data folder)
    abx_file = os.path.join(DATASET_DIR, 'microbiology_cultures_antibiotic_class_exposure.csv')
    df_abx = pd.read_csv(abx_file, usecols=['order_proc_id_coded', 'antibiotic_class', 'time_to_culturetime'])
    abx_features = generate_abx_features_optimized(df_abx)

    print(f"[LOG] Antibiotic features extracted: {len(abx_features.columns)} columns")
    del df_abx; gc.collect()

    # ==========================================================================
    # PHASE 3: WARD & DEMOGRAPHICS FEATURES
    # ==========================================================================
    print("[LOG] Extracting Ward & Demographic Features...")
    # #migrate: dataset path from config (user copies files into data folder)
    df_ward = pd.read_csv(os.path.join(DATASET_DIR, "microbiology_cultures_ward_info.csv"))
    ward_features = df_ward[['order_proc_id_coded', 'hosp_ward_IP', 'hosp_ward_OP', 'hosp_ward_ER', 'hosp_ward_ICU']]

    df_demo = pd.read_csv(os.path.join(DATASET_DIR, "microbiology_cultures_demographics.csv"))
    df_demo['gender_binary'] = pd.to_numeric(df_demo['gender'], errors='coerce')
    age_midpoint_map = {
        '18-24 years': 21.0, '25-34 years': 29.5, '35-44 years': 39.5, '45-54 years': 49.5,
        '55-64 years': 59.5, '65-74 years': 69.5, '75-84 years': 79.5, '85-89 years': 87.0, 'above 90': 92.0
    }
    df_demo['age_numeric'] = df_demo['age'].map(age_midpoint_map)
    demo_features = df_demo[['order_proc_id_coded', 'age_numeric', 'gender_binary']]

    print(f"[LOG] Ward columns: {len(ward_features.columns)} | Demo columns: {len(demo_features.columns)}")
    del df_ward, df_demo; gc.collect()

    # ==========================================================================
    # PHASE 4: PRIOR PROCEDURES FEATURES
    # ==========================================================================
    print("[LOG] Processing Prior Procedures (Chunking)...")
    # #migrate: dataset path from config (user copies files into data folder)
    proc_file = os.path.join(DATASET_DIR, 'microbiology_cultures_priorprocedures.csv')
    proc_aggs = []

    def optimize_dtypes(df):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == "object":
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype("category")
            elif str(col_type).startswith("int"):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif str(col_type).startswith("float"):
                df[col] = pd.to_numeric(df[col], downcast="float")
            elif col_type == "bool":
                df[col] = df[col].astype("int8")
        return df

    try:
        for chunk in pd.read_csv(proc_file, chunksize=50000,
                                  usecols=['order_proc_id_coded', 'procedure_description',
                                           'procedure_time_to_culturetime']):
            desc = chunk['procedure_description'].str.lower()
            chunk['flag_cvc'] = (desc == 'cvc').astype('int8')
            chunk['flag_mechvent'] = (desc == 'mechvent').astype('int8')
            chunk['flag_surgery'] = (desc == 'surgical_procedure').astype('int8')
            chunk['flag_invasive_device'] = desc.str.contains(
                r'catheter|dialysis|parenteral', regex=True).astype('int8')

            chunk_agg = chunk.groupby('order_proc_id_coded').agg({
                'flag_cvc': 'max', 'flag_mechvent': 'max', 'flag_surgery': 'max', 'flag_invasive_device': 'max'
            }).reset_index()
            proc_aggs.append(chunk_agg)

        df_proc_final = pd.concat(proc_aggs).groupby('order_proc_id_coded').max().reset_index()
        df_proc_final = optimize_dtypes(df_proc_final)
        del proc_aggs, chunk, chunk_agg; gc.collect()
        print(f"[LOG] Procedure features generated successfully. Columns: {len(df_proc_final.columns)}")
    except Exception as e:
        print(f"[ERROR] Procedure extraction failed: {e}")

    # ==========================================================================
    # PHASE 5: MICROBIAL RESISTANCE FEATURES
    # ==========================================================================
    print("[LOG] Processing Microbial Resistance Features...")
    # #migrate: dataset path from config (user copies files into data folder)
    resist_file = os.path.join(DATASET_DIR, 'microbiology_cultures_microbial_resistance.csv')
    df_resist = pd.read_csv(resist_file, usecols=['order_proc_id_coded', 'antibiotic', 'resistant_time_to_culturetime'])
    df_resist['antibiotic'] = df_resist['antibiotic'].str.strip()

    resistance_map = {
        'Carbapenem': ['Ertapenem', 'Meropenem'],
        'LastResort': ['Colistin', 'Linezolid', 'Minocycline', 'Amikacin', 'Rifampin'],
        'Quinolone': ['Ciprofloxacin', 'Levofloxacin', 'Moxifloxacin'],
        'Ceph': ['Cefepime', 'Ceftazidime', 'Ceftriaxone', 'Cefazolin', 'Cefoxitin', 'Cefpodoxime'],
        'Vanco': ['Vancomycin'],
        'UTI': ['Nitrofurantoin']
    }
    drug_to_group = {drug: group for group, drugs in resistance_map.items() for drug in drugs}
    df_resist['R_group'] = df_resist['antibiotic'].map(drug_to_group)
    df_resist = df_resist.dropna(subset=['R_group'])

    df_resist['recency_weight'] = np.exp(-df_resist['resistant_time_to_culturetime'] / 60)

    grouped = df_resist.groupby(['order_proc_id_coded', 'R_group'])
    features = grouped.agg(
        R_count=('R_group', 'count'), R_weighted=('recency_weight', 'sum'),
        R_time_since_last=('resistant_time_to_culturetime', 'min')
    ).reset_index()

    features_wide = features.pivot(index='order_proc_id_coded', columns='R_group')
    features_wide.columns = [f"{grp}_{feat}" for feat, grp in features_wide.columns]
    features_wide = features_wide.reset_index()
    del grouped, features

    df_resist['recent_flag'] = (df_resist['resistant_time_to_culturetime'] <= 90).astype(int)
    global_feats = df_resist.groupby('order_proc_id_coded').agg(
        total_R_events=('R_group', 'count'), R_diversity=('R_group', 'nunique'), recent_any_R=('recent_flag', 'max')
    ).reset_index()

    severe_groups = ['Carbapenem', 'LastResort']
    df_resist['severe_flag'] = df_resist['R_group'].isin(severe_groups).astype(int)
    severe_feat = df_resist.groupby('order_proc_id_coded').agg(severe_R_events=('severe_flag', 'sum')).reset_index()
    recency_any = df_resist.groupby('order_proc_id_coded').agg(
        days_since_last_R=('resistant_time_to_culturetime', 'min')).reset_index()

    df_resist_agg = (
        features_wide
        .merge(global_feats, on='order_proc_id_coded', how='left')
        .merge(severe_feat, on='order_proc_id_coded', how='left')
        .merge(recency_any, on='order_proc_id_coded', how='left')
    ).fillna(0)

    del df_resist, features_wide, global_feats, severe_feat, recency_any; gc.collect()
    print(f"[LOG] Total resistance features: {len(df_resist_agg.columns)}")

    # ==========================================================================
    # PHASE 6: PRIOR MEDICATION FEATURES
    # ==========================================================================
    print("[LOG] Processing Prior Medication Features...")
    med_map = {
        'CIP': 'Fluoroquinolone', 'CIP1': 'Fluoroquinolone', 'CIP2': 'Fluoroquinolone', 'CIP3': 'Fluoroquinolone',
        'CIP4': 'Fluoroquinolone',
        'LEV': 'Fluoroquinolone', 'LEV1': 'Fluoroquinolone', 'LEV2': 'Fluoroquinolone',
        'MOX': 'Fluoroquinolone', 'MOX1': 'Fluoroquinolone', 'OFL': 'Fluoroquinolone', 'GAT': 'Fluoroquinolone',
        'CEF': 'Beta_Lactam', 'CEF1': 'Beta_Lactam', 'CEF2': 'Beta_Lactam', 'CEF3': 'Beta_Lactam', 'CEF4': 'Beta_Lactam',
        'CEF5': 'Beta_Lactam', 'CEF6': 'Beta_Lactam', 'CEF7': 'Beta_Lactam', 'CEF8': 'Beta_Lactam', 'CEF9': 'Beta_Lactam',
        'CEF10': 'Beta_Lactam', 'CEF11': 'Beta_Lactam', 'CEP': 'Beta_Lactam', 'KEF': 'Beta_Lactam', 'PEN': 'Beta_Lactam',
        'AMP': 'Beta_Lactam', 'AMP1': 'Beta_Lactam', 'AMO': 'Beta_Lactam', 'AMO1': 'Beta_Lactam', 'AUG': 'Beta_Lactam',
        'PIP': 'Beta_Lactam', 'PIP1': 'Beta_Lactam', 'DIC': 'Beta_Lactam',
        'ERT': 'Carbapenem', 'MER': 'Carbapenem',
        'VAN': 'MRSA_VRE_Active', 'VAN1': 'MRSA_VRE_Active', 'VAN2': 'MRSA_VRE_Active', 'VAN3': 'MRSA_VRE_Active',
        'LIN': 'MRSA_VRE_Active', 'LIN1': 'MRSA_VRE_Active', 'ZYV': 'MRSA_VRE_Active', 'DAP': 'MRSA_VRE_Active',
        'TED': 'MRSA_VRE_Active',
        'SUL': 'UTI_Sulfa', 'NIT': 'UTI_Sulfa', 'NIT1': 'UTI_Sulfa', 'FOS': 'UTI_Sulfa', 'HIP': 'UTI_Sulfa',
        'TRI': 'UTI_Sulfa',
        'AZI': 'Community_Resp', 'ZIT': 'Community_Resp', 'CLA': 'Community_Resp', 'ERY': 'Community_Resp',
        'ERY1': 'Community_Resp',
        'DOX': 'Community_Resp', 'DOX1': 'Community_Resp', 'MIN': 'Community_Resp', 'MAC': 'Community_Resp',
        'MAC1': 'Community_Resp', 'FID': 'Community_Resp',
        'GEN': 'Severe_GramNeg', 'GEN1': 'Severe_GramNeg', 'GEN2': 'Severe_GramNeg', 'TOB': 'Severe_GramNeg',
        'TOB1': 'Severe_GramNeg',
        'AMI': 'Severe_GramNeg', 'COL': 'Severe_GramNeg', 'AZT': 'Severe_GramNeg', 'AZT1': 'Severe_GramNeg',
        'MET': 'Anaerobes', 'MET1': 'Anaerobes', 'MET2': 'Anaerobes', 'MET3': 'Anaerobes',
        'CLI': 'Anaerobes', 'CLI1': 'Anaerobes', 'CLI2': 'Anaerobes',
        'RIF': 'Other_Antibiotic', 'RIF1': 'Other_Antibiotic', 'RIF2': 'Other_Antibiotic',
        'XIF': 'Other_Antibiotic', 'BAC': 'Other_Antibiotic', 'BAC1': 'Other_Antibiotic',
        'ISO': 'Other_Antibiotic', 'ETH': 'Other_Antibiotic', 'SIL': 'Other_Antibiotic'
    }

    # #migrate: dataset path from config (user copies files into data folder)
    df_meds = pd.read_csv(os.path.join(DATASET_DIR, 'microbiology_cultures_prior_med.csv'))
    df_meds['order_proc_id_coded'] = df_meds['order_proc_id_coded'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_meds['abx_group'] = df_meds['medication_category'].map(med_map)
    df_meds = df_meds.dropna(subset=['abx_group'])
    df_meds = df_meds[df_meds['medication_time_to_culturetime'] > 0]

    def build_features(df, window_days, suffix):
        mask = df['medication_time_to_culturetime'] <= window_days
        subset = df[mask]
        if subset.empty:
            return pd.DataFrame(columns=['order_proc_id_coded'])
        feats = pd.crosstab(subset['order_proc_id_coded'], subset['abx_group']).add_prefix(f'abx_{suffix}_')
        feats[f'abx_{suffix}_total_count'] = subset.groupby('order_proc_id_coded').size()
        return feats.reset_index()

    print("[LOG] Building medication 30-day and 90-day histories...")
    df_30 = build_features(df_meds, 30, '30d')
    df_90 = build_features(df_meds, 90, '90d')
    window_feats = df_90.merge(df_30, on='order_proc_id_coded', how='outer')

    features = df_meds.groupby(['order_proc_id_coded', 'abx_group']).agg(
        time_since_last=('medication_time_to_culturetime', 'min')).reset_index()
    features_wide = features.pivot(index='order_proc_id_coded', columns='abx_group')
    features_wide.columns = [f"med_{grp}_time_since_last" for feat, grp in features_wide.columns]
    features_wide = features_wide.reset_index()
    del features

    global_feats = df_meds.groupby('order_proc_id_coded').agg(
        med_total_abx_count=('abx_group', 'count'), med_days_since_last_any_abx=('medication_time_to_culturetime', 'min')
    ).reset_index()

    df_meds_final = pd.DataFrame({'order_proc_id_coded': df_meds['order_proc_id_coded'].unique()})
    df_meds_final = (
        df_meds_final
        .merge(window_feats, on='order_proc_id_coded', how='left')
        .merge(features_wide, on='order_proc_id_coded', how='left')
        .merge(global_feats, on='order_proc_id_coded', how='left')
    )

    count_cols = [c for c in df_meds_final.columns if 'count' in c or 'abx_30d_' in c or 'abx_90d_' in c]
    df_meds_final[count_cols] = df_meds_final[count_cols].fillna(0)
    time_cols = [c for c in df_meds_final.columns if 'time_since_last' in c or 'days_since_last' in c]
    df_meds_final[time_cols] = df_meds_final[time_cols].fillna(9999)

    df_meds_final['order_proc_id_coded'] = df_meds_final['order_proc_id_coded'].astype('int64')

    del df_meds, df_30, df_90, window_feats, features_wide, global_feats; gc.collect()
    print(f"[LOG] Total medication features generated: {len(df_meds_final.columns) - 1}")

    # ==========================================================================
    # PHASE 7: COMORBIDITIES
    # ==========================================================================
    print("[LOG] Processing Comorbidities Features...")

    # #migrate: dataset path from config (user copies files into data folder)
    df_comorb = pd.read_csv(os.path.join(DATASET_DIR, 'microbiology_cultures_comorbidity.csv'),
                             usecols=['order_proc_id_coded', 'comorbidity_component',
                                      'comorbidity_component_start_days_culture',
                                      'comorbidity_component_end_days_culture'])

    CATEGORY_KEYWORDS = {
        'cardiovascular': r'heart|cardiac|hypertension|vascular|coronary|arrhythmia|myocardial|atherosclerosis|angina|aortic|circulatory|pericarditis|endocarditis|valv|congestive|fibrillation|dysrhythmia|hypotension|aneurysm|phlebitis|thromboemboli|varicose|venous insufficiency|postthrombotic|cerebrovascular|infarction|dissection|embolism|arterial|shock|conduction|vein|lymphatic|cardiomyopathy|myocard|chest pain',
        'respiratory': r'lung|pulmonary|respiratory|asthma|copd|pneumonia|bronch|pleural|pleurisy|pneumothorax|influenza|sinusitis|tonsil|aspiration pneumon|mediastinal|tuberculosis',
        'neurological': r'brain|neuro|stroke|seizure|epilepsy|parkinson|paralysis|cerebral|spinal|cns|encephalitis|meningitis|dementia|migraine|headache|myelopathy|cognitive|concussion|tbi|polyneuropath|\bmyopathies\b|\bmyopathy\b|nerve root|cerebral palsy|neurodevelopmental|cord injury|poliomyelitis|neurocognitive|neuroendocrine|neurogenic|neuropathic|nervous',
        'liver_gi': r'liver|hepatitis|cirrhosis|gastro|intestinal|ulcer|pancrea|hepatic|biliary|bile|esophag|colitis|diverticul|appendic|hemorrhoid|hernia|peritonitis|duodenit|bowel|stomach|digestive|nausea|vomiting|dysphagia|abdominal|anal and rectal|mouth|teeth|gingiva|jaw|palate|intoxication',
        'musculoskeletal': r'joint|bone|muscle|arthritis|osteo|gout|rheumat|spondyl|back|spine|scoliosis|tendon|synovial|fracture|sprain|strain|dislocation|limb|musculoskeletal|dorsopath|connective tissue|deformit|arthropathy|biomechanical|osteoporosis|osteomalacia|aseptic necrosis|osteonecrosis|osteomyelitis|orthopedic|immune.mediated.*arthrop|reactive.*arthrop',
        'hematologic': r'anemia|coagulation|hemophilia|blood.*cell|platelet|coagulopathy|hemolytic|thrombocytopenia|polycythemia|hematologic|hemorrhagic|sickle cell|leukocyt|blood loss|injury to blood vessel',
        'mental_health': r'depression|anxiety|psych|substance|opioid|alcohol|schizo|bipolar|mood disorder|mental|behavioral|conduct disorder|personality disorder|trauma.*stress|obsessive|compulsive|eating disorder|suicid|self.harm|hallucinogen|stimulant.related|cannabis|inhalant|sedative.related|tobacco.related|sleep.*disorder|somatic disorder|impulse.control|depressive disorder|psychoses|maltreatment',
        'eye_ear': r'eye|ocular|vision|blindness|cataract|glaucoma|retinal|cornea|uveitis|ear|hearing|deafness|mastoid|strabismus|refractive|oculofacial|neuro.ophthalmol|otitis',
        'skin': r'skin|dermatitis|cellulitis|psoriasis|subcutaneous|burn|corrosion|pressure ulcer|non.pressure ulcer|contact dermatitis',
        'diabetes': r'diabetes|diabetic|glucose tolerance',
        'renal_urological': r'kidney|renal|urinary|bladder|calculus|nephritis|nephrosis|renal sclerosis|proteinuria|hematuria|vesicoureteral|urethr|ureter|renal failure|chronic kidney',
        'metabolic_endocrine': r'obesity|lipid|metabolic|electrolyte|nutrition|thyroid|endocrine|adrenal|pituitary|hypothyroid|malnutrition|weight loss|nutritional deficiency|fluid.*disorder|crystal arthropathy|acidemia|hypoxia|vitamin|mineral deficien',
        'oncology_immune': r'cancer|leukemia|lymphoma|hiv|transplant|sarcoma|myeloma|aplastic anemia|malignant|malignancy|carcinoma|metastas|neoplastic|immunodeficien|immunosuppress|autoimmun|lupus|solid tumor|secondary malignan|myelodysplastic|cystic fibrosis|immunity disorder|neoplasm|tumor|hodgkin|antineoplastic|mesothelioma|autoinflammatory|allerg',
        'infection_systemic': r'septicemia|sepsis|\bviral infection\b|\bbacterial infections?\b|\bfungal infections?\b|\bparasitic\b|intestinal infection|foodborne|sexually transmitted|antimicrobial.*resistance|resistance to antimicrobial|fever|gangrene|skin.*infection|abscess|sequela.*infect|infect.*sequela',
        'pregnancy_maternity': r'pregnan|gestat|matern|fetal|labor|deliver|childbirth|puerperium|liveborn|amniotic|c-section|perinatal infection|abort|antenatal|trimester|placenta|ectopic|molar pregnancy|puerper|OB.related',
        'injury_trauma': r'\binjur|fracture|wound|burn|trauma|concussion|foreign body|crushing|amputation|open wound|sprain|strain|dislocation|superficial injury|contusion|internal organ injury|spinal cord injury|traumatic brain|injury to nerve|injury to blood',
        'neonatal_perinatal': r'neonatal|newborn|liveborn|birth trauma|perinatal|short gestation|low birth weight|fetal growth|neonatal abstinence|neonatal acidemia|neonatal hypoxia|neonatal cerebral|neonatal digestive|neonatal.*feeding|hemorrhagic.*newborn|hemolytic jaundice|perinatal jaundice|fetal alcohol|newborn affected by maternal|respiratory.*perinatal|chromosomal abnormali|cleft',
        'reproductive_gynecology': r'menstrual|ovarian|uterine|cervic|endometri|vulv|vagin|fallopian|female.*genital|male.*genital|penis|testis|prostat|erectile|infertil|contraceptive|procreative|pelvic organ|menopaus|prolapse.*genital|inflammatory.*pelvic|inflammatory.*male genital|female reproductive|male reproductive',
        'drug_adverse_effects': r'adverse effect.*drug|poisoning by drug|toxic effect|underdosing|drug induced|medicament|pharmacological|overdose',
        'administrative_encounters': r'encounter for|screening|prophylactic|administrative|follow.up|observation and examination|history of.*disease|personal.*history|family.*history|medical examination|medical evaluation|abnormal findings|carrier status|immunization|external cause|counseling|organ transplant status|other aftercare|encounter.*mental health|encounter.*antineoplastic|encounter.*abuse|sensation|perception|invalid|lifestyle|life management|malaise|fatigue|general signs|general symptom|congenital anomal|other specified status|postprocedural.*spleen|postoperative.*spleen|complication.*sequela|\bimplant,?\s+device\b|\bgraft related encounter\b|device or graft related',
    }

    def clean_column_name(text):
        if pd.isna(text):
            return 'unknown'
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    df_comorb['clean_comorb'] = df_comorb['comorbidity_component'].apply(clean_column_name)

    df_comorb['is_active'] = (
        (df_comorb['comorbidity_component_start_days_culture'] >= 0) &
        (
            df_comorb['comorbidity_component_end_days_culture'].isna() |
            (df_comorb['comorbidity_component_end_days_culture'] <= 0)
        )
    ).astype('int8')

    for cat, pattern in CATEGORY_KEYWORDS.items():
        df_comorb[f'flag_{cat}'] = df_comorb['clean_comorb'].str.contains(pattern, case=False, na=False).astype('int8')

    agg_funcs = {'is_active': 'sum'}
    for cat in CATEGORY_KEYWORDS.keys():
        agg_funcs[f'flag_{cat}'] = 'max'

    final_features = df_comorb.groupby('order_proc_id_coded').agg(agg_funcs).reset_index()

    final_features.rename(columns={'is_active': 'comorb_active_count'}, inplace=True)
    final_features['comorb_active_count'] = final_features['comorb_active_count'].astype('int16')

    flag_cols = []
    for cat in CATEGORY_KEYWORDS.keys():
        new_col_name = f'comorb_flag_{cat}'
        final_features.rename(columns={f'flag_{cat}': new_col_name}, inplace=True)
        final_features[new_col_name] = final_features[new_col_name].astype('int8')
        flag_cols.append(new_col_name)

    final_features['comorb_category_count'] = final_features[flag_cols].sum(axis=1).astype('int8')

    burden = df_comorb[['order_proc_id_coded', 'clean_comorb']].drop_duplicates()
    burden = burden.groupby('order_proc_id_coded').size().reset_index(name='comorb_total_count')
    burden['comorb_total_count'] = burden['comorb_total_count'].astype('int16')

    df_comorb = final_features.merge(burden, on='order_proc_id_coded', how='left')

    del final_features, burden; gc.collect()

    # ==========================================================================
    # PHASE 8: MASTER MERGE
    # ==========================================================================
    print("[LOG] Executing Master Merge...")

    data = df_cohort.merge(abx_features, on='order_proc_id_coded', how='left')
    data = data.merge(ward_features, on='order_proc_id_coded', how='left')
    data = data.merge(demo_features, on='order_proc_id_coded', how='left')
    data = data.merge(df_proc_final, on='order_proc_id_coded', how='left')
    data = data.merge(df_resist_agg, on='order_proc_id_coded', how='left')
    data = data.merge(df_meds_final, on='order_proc_id_coded', how='left')
    data = data.merge(df_comorb, on='order_proc_id_coded', how='left')

    print("[LOG] Master Merge Complete. Reclaiming memory from partial dataframes...")
    del df_cohort, abx_features, ward_features, demo_features, df_proc_final, df_resist_agg, df_meds_final, df_comorb
    gc.collect()

    fill_zero_cols = [
        col for col in data.columns
        if col.startswith('comorb_') or col.startswith('med_') or ('weighted_' in col) or ('last_' in col) or ('R_' in col)
    ]
    fill_zero_cols = [c for c in fill_zero_cols if 'days_since' not in c]

    data[fill_zero_cols] = data[fill_zero_cols].fillna(0)
    data['hosp_ward_IP'] = data['hosp_ward_ICU'].fillna(0)
    data['hosp_ward_OP'] = data['hosp_ward_ER'].fillna(0)
    data['hosp_ward_ER'] = data['hosp_ward_ER'].fillna(0)
    data['hosp_ward_ICU'] = data['hosp_ward_ICU'].fillna(0)

    print("[LOG] Optimizing Master Data Types...")
    data = optimize_dtypes(data)
    gc.collect()

    print(f"[LOG] Final Master Data Shape: {data.shape}")

    # ==========================================================================
    # PHASE 9: SETUP & PREPROCESSING FOR ML
    # ==========================================================================
    cat_cols = ['organism', 'ordering_mode', 'antibiotic']
    exclude_cols = set(cat_cols + ['is_resistant', 'order_proc_id_coded'])
    num_cols = [c for c in data.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])]

    print(f"[LOG] Auto-detected {len(num_cols)} numeric features.")
    valid_data = data.dropna(subset=['is_resistant'])

    X = valid_data[cat_cols + num_cols]
    y = valid_data['is_resistant'].astype(int)

    # ==========================================================================
    # FIT PREPROCESSOR + SPLIT (ONCE) AND CACHE
    # ==========================================================================
    preprocessor = ColumnTransformer(transformers=[
        ('cat', Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    print("[LOG] Running Preprocessing (One-Hot Encoding)...")
    X_processed = preprocessor.fit_transform(X)
    y_array = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_array,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_array
    )

    sp.save_npz(X_TRAIN_PATH, sp.csr_matrix(X_train))
    sp.save_npz(X_TEST_PATH, sp.csr_matrix(X_test))
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    meta = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'n_features': X_train.shape[1],
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[LOG] Cached preprocessed train/test split + preprocessor + meta to {CACHE_DIR}/")
    print(f"[LOG] X_train shape: {X_train.shape}  |  X_test shape: {X_test.shape}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[CHECK] Loaded X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"[CHECK] Train class balance -> 0:{(y_train==0).sum()}  1:{(y_train==1).sum()}")
