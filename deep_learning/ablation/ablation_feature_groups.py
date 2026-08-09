"""
Feature group definitions and removal logic for the TabTransformer ablation
study.

#migrate: extracted from tabtransformer_ablation.py
"""

from typing import Dict, List, Sequence


def build_feature_groups(
    cat_features: Sequence[str],
    cont_features: Sequence[str],
    binary_features: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Build clinically interpretable groups from the exact saved feature names.

    Prefix-driven groups automatically include all one-hot columns. Features not
    captured by a named group remain in the full model and are documented in the
    feature-group output table.
    """
    all_features = list(cat_features) + list(cont_features) + list(binary_features)

    def existing(names: Sequence[str]) -> List[str]:
        return [name for name in names if name in all_features]

    def starts_with(*prefixes: str) -> List[str]:
        return [name for name in all_features if name.startswith(prefixes)]

    lab_names = [
        name for name in cont_features
        if any(
            token in name
            for token in [
                "wbc", "neutroph", "lymph", "hgb", "plt", "_na",
                "hco3", "bun", "_cr", "lactate", "procalcitonin",
            ]
        )
    ]
    vital_names = [
        name for name in cont_features
        if any(
            token in name
            for token in ["heartrate", "resprate", "temp", "sysbp", "diasbp"]
        )
    ]

    groups = {
        "MICROBIOLOGY_IDENTITY": existing(
            ["organism_enc", "antibiotic_enc", "culture_type_enc"]
        ),
        "CARE_CONTEXT": existing(
            ["ordering_mode_enc", "hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
        ),
        "DEMOGRAPHICS": existing(["age_enc", "gender_enc"]),
        "TEMPORAL": existing(["order_year", "order_month"]),
        "LABS": lab_names,
        "VITALS": vital_names,
        "COMORBIDITIES": existing(["comorb_total_count"]) + starts_with("comorb_"),
        "ANTIBIOTIC_EXPOSURE": starts_with("abclass_", "absub_"),
        "PRIOR_ORGANISMS": starts_with("priororg_"),
        "PRIOR_PROCEDURES": starts_with("proc_"),
        "SOCIOECONOMIC": existing(["adi_score", "adi_state_rank"]),
        "NURSING_HOME": existing(["nursing_home_visits"]),
        "PRIOR_RESISTANCE": existing(
            ["prior_resistance_count", "min_resistance_days"]
        ),
        "ALL_CATEGORICAL": list(cat_features),
        "ALL_CONTINUOUS": list(cont_features),
        "ALL_BINARY": list(binary_features),
    }

    # Remove duplicates within each group while preserving order.
    groups = {
        group: list(dict.fromkeys(features))
        for group, features in groups.items()
        if len(features) > 0
    }
    return groups


def create_experiments(feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Create one FULL_CONTROL experiment and one NO_<group> experiment per group."""
    experiments: Dict[str, List[str]] = {"FULL_CONTROL": []}
    for group_name, feature_names in feature_groups.items():
        experiments[f"NO_{group_name}"] = feature_names
    return experiments
