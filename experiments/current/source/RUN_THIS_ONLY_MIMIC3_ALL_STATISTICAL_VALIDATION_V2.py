# -*- coding: utf-8 -*-
r"""
One-click MIMIC-III statistical external-validation pipeline for the CRP/WBC paper.

IMPORTANT
---------
This script uses statistical methods only:
- correlation analysis;
- univariable logistic regression;
- external prediction using coefficients fitted in MIMIC-IV;
- ROC-AUC, Accuracy, Recall, Precision, Specificity, F1 and Brier score;
- bootstrap confidence intervals;
- Cox proportional hazards regression;
- restricted cubic spline (RCS) Cox analysis.

It does NOT use:
- neural networks;
- PyTorch;
- SMOTE;
- SVM or random forest for the main validation analysis.

The script is intentionally self-contained. Put it in:

    D:\python_projects\python_project IPM\crp_wbc_hu

and run:

    D:\anaconda\python.exe ^
    "D:\python_projects\python_project IPM\crp_wbc_hu\RUN_THIS_ONLY_MIMIC3_ALL_STATISTICAL_VALIDATION.py"

Expected MIMIC-III feature files
--------------------------------
The script automatically checks historical filenames and, if needed, recursively scans CSV headers for:

    E:\clinic_work\data_mimic3\wbc_crp_data\
        crp_with_death2_metrics_new_metrics.csv
        or crp_with_death2_metrics.csv

    E:\clinic_work\data_mimic3\wbc_crp_data\
        wbc_full2_metrics.csv

    E:\clinic_work\data_mimic3\sepsis_wbc_crp_data\
        sepsis_subjects_mimic3.csv

If the CRP file does not already contain crp_high_grade_occurrence, it is
calculated from:

    crp_grade_4_ratio / (crp_mean_time_diff_hours + EPS)

Optional MIMIC-III endpoint/survival inputs
-------------------------------------------
The script first searches for a patient-level file named like:

    mimic3_outcomes_survival.csv
    mimic3_death_indicators.csv
    mimic3_survival_core.csv

A supplied outcome file may contain any of:

    subject_id
    death_if
    death_icu
    death_in_hosp
    death_28d
    death_90d
    death_1y
    t_primary_days
    t_icu_days
    t_hosp_days
    t_28d_days
    t_90d_days
    t_1y_days
    age
    sex

If no such file is found, the script tries to derive endpoint and survival
variables from raw MIMIC-III tables:

    PATIENTS.csv
    ADMISSIONS.csv
    ICUSTAYS.csv

For raw-table derivation, the first ICU stay per patient is used as the
time origin. The resulting assumptions are recorded in the output audit.

Main output directory
---------------------
    final_real\mimic3_statistical_external_validation

The output folder includes:
    00_input_audit.txt
    01_mimic4_patient_level_indices.csv
    02_mimic3_patient_level_indices.csv
    03_cohort_summary.csv
    04_mimic3_correlations.csv
    05_derivation_logistic_models.csv
    06_external_validation_metrics.csv
    07_mimic3_refit_logistic_associations.csv
    08_mimic3_cox_results.csv
    09_mimic3_rcs_nonlinearity.csv
    10_article_ready_results.txt
    figures\...
    rcs_curves\...

Notes
-----
1. MIMIC-IV is used to fit each logistic model.
2. The fitted MIMIC-IV model is applied unchanged to MIMIC-III.
3. MIMIC-III is not used for model selection or threshold selection.
4. The probability threshold is fixed at 0.5, matching the manuscript.
5. Cox and RCS results are produced only when valid event and duration
   variables are available.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau, pearsonr, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import statsmodels.api as sm
except Exception as exc:
    raise RuntimeError(
        "statsmodels could not be imported. Install/repair it with:\n"
        'D:\\anaconda\\python.exe -m pip install -U "statsmodels>=0.14"\n'
        f"Original error: {exc}"
    ) from exc

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    LIFELINES_AVAILABLE = True
    LIFELINES_IMPORT_ERROR = ""
except Exception as exc:
    CoxPHFitter = None
    proportional_hazard_test = None
    LIFELINES_AVAILABLE = False
    LIFELINES_IMPORT_ERROR = str(exc)

try:
    from patsy import build_design_matrices, dmatrix
    PATSY_AVAILABLE = True
    PATSY_IMPORT_ERROR = ""
except Exception as exc:
    build_design_matrices = None
    dmatrix = None
    PATSY_AVAILABLE = False
    PATSY_IMPORT_ERROR = str(exc)


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\python_projects\python_project IPM\crp_wbc_hu"
)

MIMIC3_ROOT = Path(
    r"E:\clinic_work\data_mimic3"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "final_real"
    / "mimic3_statistical_external_validation"
)

EPS = 1e-6
PROBABILITY_THRESHOLD = 0.5
RANDOM_STATE = 20260719
N_BOOTSTRAP = 2000
MIN_BOOTSTRAP_VALID = 300
BOOTSTRAP_ALPHA = 0.05

# Run all three manuscript indices.
INDEX_COLUMNS = [
    "CIDI-T",
    "CIDI-LVH",
    "CIDI-LVH-S",
]

# CIDI-LVH-S is the primary external-validation exposure.
PRIMARY_INDEX = "CIDI-LVH-S"

OUTCOME_COLUMNS = [
    "death_if",
    "death_icu",
    "death_in_hosp",
    "death_28d",
    "death_90d",
    "death_1y",
]

OUTCOME_LABELS = {
    "death_if": "Primary mortality",
    "death_icu": "ICU mortality",
    "death_in_hosp": "In-hospital mortality",
    "death_28d": "28-day mortality",
    "death_90d": "90-day mortality",
    "death_1y": "1-year mortality",
}

DURATION_MAP = {
    "death_if": "t_primary_days",
    "death_icu": "t_icu_days",
    "death_in_hosp": "t_hosp_days",
    "death_28d": "t_28d_days",
    "death_90d": "t_90d_days",
    "death_1y": "t_1y_days",
}

# Cox models to report.
COX_MODEL_SPECS = {
    "unadjusted": [],
    "age_adjusted": ["age"],
    "sex_adjusted": ["sex"],
    "age_sex_adjusted": ["age", "sex"],
}

# RCS is run for CIDI-LVH-S. Empty means unadjusted RCS.
RCS_ADJUST_VARS: list[str] = []
RCS_N_KNOTS = 5
RCS_GRID_POINTS = 240
RCS_MIN_EVENTS = 15

# Raw MIMIC-III derivation:
# The first ICU stay per patient is used as the anchor.
USE_FIRST_ICU_STAY = True

# MIMIC-III death_if aggregation follows the historical code.
MIMIC3_DEATH_IF_AGGREGATION = "first"

# MIMIC-IV death_if aggregation follows the historical code.
MIMIC4_DEATH_IF_AGGREGATION = "min"


# =============================================================================
# 2. CANDIDATE PATHS
# =============================================================================

MIMIC4_CRP_CANDIDATES = [
    PROJECT_ROOT / "final_real" / "crp_with_metrics_final_new_metrics_speed.csv",
    PROJECT_ROOT / "final_real" / "crp_with_metrics_final_new_metrics.csv",
    PROJECT_ROOT / "final_real" / "crp_with_metrics_final.csv",
]

MIMIC4_WBC_CANDIDATES = [
    PROJECT_ROOT / "final_real" / "wbc_oringin2_with_metrics_final.csv",
]

MIMIC4_SUBJECT_CANDIDATES = [
    PROJECT_ROOT / "Sepsis" / "sepsis_subjects_mimic4.csv",
]

MIMIC4_OUTCOME_CANDIDATES = [
    PROJECT_ROOT / "final_real" / "mimic4_death_indicators.csv",
]

MIMIC4_CORE_CANDIDATES = [
    PROJECT_ROOT / "final_real" / "sepsis_core_model_fast.csv",
]

MIMIC3_CRP_CANDIDATES = [
    # Main historical names
    MIMIC3_ROOT / "wbc_crp_data" / "crp_with_death2_metrics_new_metrics.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "crp_with_death2_metrics.csv",

    # Names referenced by the original project scripts
    MIMIC3_ROOT / "wbc_crp_data" / "crp_with_death2_metrics_filtered_first.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "crp_with_death2_metrics_filtered.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "patients_crp_with_death2_metrics.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "patients_crp_with_death2.csv",

    # Sepsis-filtered outputs
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "crp_sepsis_subjects_only.csv",
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "crp_sepsis_subjects_only_first.csv",

    # Optional copies inside the current project
    PROJECT_ROOT / "mimic3" / "crp_with_death2_metrics_new_metrics.csv",
    PROJECT_ROOT / "mimic3" / "crp_with_death2_metrics.csv",
    PROJECT_ROOT / "mimic3" / "crp_with_death2_metrics_filtered_first.csv",
    PROJECT_ROOT / "mimic3" / "crp_sepsis_subjects_only.csv",
]

MIMIC3_WBC_CANDIDATES = [
    # Main historical names
    MIMIC3_ROOT / "wbc_crp_data" / "wbc_full2_metrics.csv",

    # Names referenced by the original project scripts
    MIMIC3_ROOT / "wbc_crp_data" / "wbc_full2_metrics_filtered_first.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "wbc_full2_metrics_filtered.csv",
    MIMIC3_ROOT / "wbc_crp_data" / "wbc_full2.csv",

    # Sepsis-filtered outputs
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "wbc_sepsis_subjects_only.csv",
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "wbc_sepsis_subjects_only_first.csv",

    # Optional copies inside the current project
    PROJECT_ROOT / "mimic3" / "wbc_full2_metrics.csv",
    PROJECT_ROOT / "mimic3" / "wbc_full2_metrics_filtered_first.csv",
    PROJECT_ROOT / "mimic3" / "wbc_sepsis_subjects_only.csv",
]

MIMIC3_SUBJECT_CANDIDATES = [
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "sepsis_subjects_mimic3.csv",
    PROJECT_ROOT / "Sepsis" / "sepsis_subjects_mimic3.csv",
    PROJECT_ROOT / "mimic3" / "sepsis_subjects_mimic3.csv",
]

MIMIC3_OUTCOME_CANDIDATES = [
    MIMIC3_ROOT / "mimic3_outcomes_survival.csv",
    MIMIC3_ROOT / "mimic3_death_indicators.csv",
    MIMIC3_ROOT / "mimic3_survival_core.csv",
    MIMIC3_ROOT / "sepsis_wbc_crp_data" / "mimic3_outcomes_survival.csv",
    PROJECT_ROOT / "mimic3" / "mimic3_outcomes_survival.csv",
    PROJECT_ROOT / "final_real" / "mimic3_outcomes_survival.csv",
]

MIMIC3_PATIENTS_CANDIDATES = [
    MIMIC3_ROOT / "PATIENTS.csv",
    MIMIC3_ROOT / "PATIENTS.csv.gz",
    MIMIC3_ROOT / "patients.csv",
    MIMIC3_ROOT / "patients.csv.gz",
]

MIMIC3_ADMISSIONS_CANDIDATES = [
    MIMIC3_ROOT / "ADMISSIONS.csv",
    MIMIC3_ROOT / "ADMISSIONS.csv.gz",
    MIMIC3_ROOT / "admissions.csv",
    MIMIC3_ROOT / "admissions.csv.gz",
]

MIMIC3_ICUSTAYS_CANDIDATES = [
    MIMIC3_ROOT / "ICUSTAYS.csv",
    MIMIC3_ROOT / "ICUSTAYS.csv.gz",
    MIMIC3_ROOT / "icustays.csv",
    MIMIC3_ROOT / "icustays.csv.gz",
]


# =============================================================================
# 3. GLOBAL STYLE
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)

COLOR_MAP = {
    "CIDI-T": "#7F9DBD",
    "CIDI-LVH": "#A8B9A5",
    "CIDI-LVH-S": "#D2A086",
}

GRID_COLOR = "#D8D3CD"
TEXT_COLOR = "#2E2B29"


# =============================================================================
# 4. DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ResolvedInputs:
    mimic4_crp: Path
    mimic4_wbc: Path
    mimic4_subjects: Path
    mimic4_outcomes: Optional[Path]
    mimic4_core: Optional[Path]
    mimic3_crp: Path
    mimic3_wbc: Path
    mimic3_subjects: Path
    mimic3_outcomes: Optional[Path]
    mimic3_patients: Optional[Path]
    mimic3_admissions: Optional[Path]
    mimic3_icustays: Optional[Path]


# =============================================================================
# 5. GENERAL UTILITIES
# =============================================================================

def print_banner(title: str) -> None:
    line = "=" * 110
    print(f"\n{line}\n{title}\n{line}")


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def require_existing(paths: Sequence[Path], label: str) -> Path:
    resolved = first_existing(paths)
    if resolved is None:
        candidates = "\n".join(f"  - {path}" for path in paths)
        raise FileNotFoundError(
            f"Could not find {label}. Checked:\n{candidates}"
        )
    return resolved


def _read_header_safely(path: Path) -> set[str]:
    try:
        header = pd.read_csv(
            path,
            nrows=0,
            low_memory=False,
        )
        return {
            str(column).strip().lower()
            for column in header.columns
        }
    except Exception:
        return set()


def auto_discover_feature_file(
    explicit_candidates: Sequence[Path],
    search_roots: Sequence[Path],
    feature_type: str,
) -> Path:
    """
    Resolve a MIMIC-III CRP/WBC feature file.

    Resolution order:
    1. Explicit historical filenames.
    2. Recursive header-based search under the supplied roots.

    Only CSV files with the required feature columns are accepted.
    """
    resolved = first_existing(explicit_candidates)
    if resolved is not None:
        return resolved

    if feature_type == "crp":
        accepted_column_sets = [
            {"subject_id", "crp_median", "crp_max", "crp_p10"},
            {"subject_id", "median_crp", "max_crp", "p10_crp"},
        ]
        preferred_tokens = [
            "crp",
            "metrics",
            "filtered",
            "sepsis",
        ]
    elif feature_type == "wbc":
        accepted_column_sets = [
            {"subject_id", "wbc_std", "wbc_max"},
            {"subject_id", "std_wbc", "max_wbc"},
        ]
        preferred_tokens = [
            "wbc",
            "metrics",
            "filtered",
            "sepsis",
        ]
    else:
        raise ValueError(
            f"Unsupported feature_type: {feature_type}"
        )

    matches: list[tuple[int, Path]] = []
    inspected_paths: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue

        for pattern in ("*.csv", "*.CSV"):
            for path in root.rglob(pattern):
                if path in inspected_paths:
                    continue

                inspected_paths.add(path)

                header = _read_header_safely(path)
                if not header:
                    continue

                if not any(
                    required.issubset(header)
                    for required in accepted_column_sets
                ):
                    continue

                lower_name = str(path).lower()
                score = sum(
                    token in lower_name
                    for token in preferred_tokens
                )

                # Prefer files that already contain a mortality label.
                if "death_if" in header:
                    score += 3

                # Prefer files with patient-level metric columns.
                if feature_type == "crp":
                    score += int(
                        "crp_mean_time_diff_hours" in header
                        or "mean_time_diff_hours" in header
                    )
                    score += int(
                        "crp_grade_4_ratio" in header
                        or "grade_4_ratio" in header
                    )
                else:
                    score += int(
                        "wbc_measurement_count" in header
                    )

                matches.append(
                    (score, path)
                )

    if not matches:
        explicit_text = "\n".join(
            f"  - {path}"
            for path in explicit_candidates
        )
        roots_text = "\n".join(
            f"  - {root}"
            for root in search_roots
        )

        raise FileNotFoundError(
            f"Could not find a usable MIMIC-III {feature_type.upper()} "
            f"feature file.\n"
            f"Explicit filenames checked:\n{explicit_text}\n"
            f"Recursive search roots:\n{roots_text}\n"
            "A usable file must contain subject_id and the required "
            "CRP/WBC patient-level metric columns."
        )

    matches.sort(
        key=lambda item: (
            -item[0],
            len(str(item[1])),
            str(item[1]).lower(),
        )
    )

    chosen_score, chosen_path = matches[0]

    print(
        f"[AUTO-DISCOVERY] Selected MIMIC-III "
        f"{feature_type.upper()} file:\n"
        f"  {chosen_path}\n"
        f"  discovery score={chosen_score}"
    )

    if len(matches) > 1:
        print(
            "[AUTO-DISCOVERY] Other usable candidates:"
        )
        for score, path in matches[1:10]:
            print(
                f"  score={score}: {path}"
            )

    return chosen_path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(column).strip().lower()
        for column in out.columns
    ]

    aliases = {
        "subjectid": "subject_id",
        "subject": "subject_id",
        "hadmid": "hadm_id",
        "icustay_id": "stay_id",
        "icustayid": "stay_id",
        "gender": "sex",
        "hospital_expire_flag": "death_in_hosp",
        "expire_flag": "death_if",
    }

    rename = {
        column: aliases[column]
        for column in out.columns
        if column in aliases
    }

    if rename:
        out = out.rename(columns=rename)

    return out


def read_csv_normalized(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        **kwargs,
    )
    return normalize_columns(df)


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def binary_clean(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    result.loc[numeric == 0] = 0.0
    result.loc[numeric == 1] = 1.0

    text = series.astype(str).str.strip().str.lower()
    result.loc[text.isin(["false", "no", "n", "alive", "survived"])] = 0.0
    result.loc[text.isin(["true", "yes", "y", "dead", "death", "died"])] = 1.0

    return result


def aggregate_binary(
    series: pd.Series,
    method: str,
) -> float:
    values = binary_clean(series).dropna()

    if values.empty:
        return float("nan")

    if method == "first":
        return float(values.iloc[0])
    if method == "last":
        return float(values.iloc[-1])
    if method == "min":
        return float(values.min())
    if method == "max":
        return float(values.max())
    if method == "majority":
        mode = values.mode()
        return float(mode.iloc[0]) if not mode.empty else float("nan")

    raise ValueError(
        f"Unsupported binary aggregation method: {method}"
    )


def safe_log_product(
    product: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        product,
        errors="coerce",
    )

    output = pd.Series(
        np.nan,
        index=numeric.index,
        dtype=float,
    )

    valid = (
        numeric.notna()
        & np.isfinite(numeric)
        & (numeric > 0)
    )

    output.loc[valid] = np.log(
        numeric.loc[valid]
    )

    return output


def safe_json_value(value):
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_text(path: Path, text: str) -> None:
    path.write_text(
        text,
        encoding="utf-8",
    )


def format_p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


# =============================================================================
# 6. INPUT RESOLUTION AND AUDIT
# =============================================================================

def resolve_inputs() -> ResolvedInputs:
    return ResolvedInputs(
        mimic4_crp=require_existing(
            MIMIC4_CRP_CANDIDATES,
            "MIMIC-IV CRP feature file",
        ),
        mimic4_wbc=require_existing(
            MIMIC4_WBC_CANDIDATES,
            "MIMIC-IV WBC feature file",
        ),
        mimic4_subjects=require_existing(
            MIMIC4_SUBJECT_CANDIDATES,
            "MIMIC-IV sepsis subject list",
        ),
        mimic4_outcomes=first_existing(
            MIMIC4_OUTCOME_CANDIDATES
        ),
        mimic4_core=first_existing(
            MIMIC4_CORE_CANDIDATES
        ),
        mimic3_crp=auto_discover_feature_file(
            explicit_candidates=MIMIC3_CRP_CANDIDATES,
            search_roots=[
                MIMIC3_ROOT,
                PROJECT_ROOT / "mimic3",
                PROJECT_ROOT,
            ],
            feature_type="crp",
        ),
        mimic3_wbc=auto_discover_feature_file(
            explicit_candidates=MIMIC3_WBC_CANDIDATES,
            search_roots=[
                MIMIC3_ROOT,
                PROJECT_ROOT / "mimic3",
                PROJECT_ROOT,
            ],
            feature_type="wbc",
        ),
        mimic3_subjects=require_existing(
            MIMIC3_SUBJECT_CANDIDATES,
            "MIMIC-III sepsis subject list",
        ),
        mimic3_outcomes=first_existing(
            MIMIC3_OUTCOME_CANDIDATES
        ),
        mimic3_patients=first_existing(
            MIMIC3_PATIENTS_CANDIDATES
        ),
        mimic3_admissions=first_existing(
            MIMIC3_ADMISSIONS_CANDIDATES
        ),
        mimic3_icustays=first_existing(
            MIMIC3_ICUSTAYS_CANDIDATES
        ),
    )


def build_input_audit(
    inputs: ResolvedInputs,
) -> str:
    lines = [
        "MIMIC-III statistical external-validation input audit",
        "=" * 72,
        "",
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"lifelines available: {LIFELINES_AVAILABLE}",
        f"patsy available: {PATSY_AVAILABLE}",
        "",
        "Resolved files:",
    ]

    for field_name, value in inputs.__dict__.items():
        lines.append(
            f"{field_name}: {value if value is not None else 'NOT FOUND'}"
        )

    lines.extend(
        [
            "",
            "Protocol:",
            "- Statistical methods only; no neural network and no SMOTE.",
            "- Logistic models are fitted in MIMIC-IV.",
            "- MIMIC-IV coefficients are applied unchanged to MIMIC-III.",
            f"- Fixed probability threshold: {PROBABILITY_THRESHOLD}.",
            f"- Bootstrap replicates: {N_BOOTSTRAP}.",
            f"- MIMIC-IV death_if aggregation: {MIMIC4_DEATH_IF_AGGREGATION}.",
            f"- MIMIC-III death_if aggregation: {MIMIC3_DEATH_IF_AGGREGATION}.",
            "- Candidate indices: CIDI-T, CIDI-LVH, CIDI-LVH-S.",
            "- CIDI-LVH-S is treated as the primary index.",
            "",
        ]
    )

    if not LIFELINES_AVAILABLE:
        lines.extend(
            [
                "Cox/RCS warning:",
                "lifelines is unavailable, so Cox and RCS will be skipped.",
                f"Import error: {LIFELINES_IMPORT_ERROR}",
                'Install with: D:\\anaconda\\python.exe -m pip install "lifelines>=0.30"',
                "",
            ]
        )

    if not PATSY_AVAILABLE:
        lines.extend(
            [
                "RCS warning:",
                "patsy is unavailable, so RCS will be skipped.",
                f"Import error: {PATSY_IMPORT_ERROR}",
                'Install with: D:\\anaconda\\python.exe -m pip install "patsy>=0.5"',
                "",
            ]
        )

    if inputs.mimic3_outcomes is None:
        raw_complete = all(
            path is not None
            for path in [
                inputs.mimic3_patients,
                inputs.mimic3_admissions,
                inputs.mimic3_icustays,
            ]
        )

        if raw_complete:
            lines.extend(
                [
                    "MIMIC-III outcome source:",
                    "- No patient-level outcome file was found.",
                    "- Outcomes will be derived from PATIENTS, ADMISSIONS and ICUSTAYS.",
                    "- The first ICU stay per patient will be used as the time origin.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "MIMIC-III outcome warning:",
                    "- No patient-level outcome/survival file was found.",
                    "- Raw PATIENTS/ADMISSIONS/ICUSTAYS are incomplete.",
                    "- Primary death_if may still be available from the CRP feature file.",
                    "- Multi-endpoint Cox/RCS may be skipped.",
                    "",
                ]
            )

    return "\n".join(lines)


# =============================================================================
# 7. INDEX CONSTRUCTION
# =============================================================================

REQUIRED_INDEX_SOURCE_COLUMNS = [
    "subject_id",
    "crp_median",
    "crp_max",
    "crp_min",
    "crp_p10",
    "crp_mean_time_diff_hours",
    "crp_grade_4_ratio",
    "wbc_std",
    "wbc_max",
]


def add_crp_new_metrics_if_needed(
    crp: pd.DataFrame,
) -> pd.DataFrame:
    out = crp.copy()

    # The oldest MIMIC-III CRP metric script used names such as
    # median_crp and grade_4_ratio, whereas later scripts used
    # crp_median and crp_grade_4_ratio. Normalize both formats.
    legacy_aliases = {
        "measurement_count": "crp_measurement_count",
        "mean_crp": "crp_mean",
        "std_crp": "crp_std",
        "max_crp": "crp_max",
        "min_crp": "crp_min",
        "median_crp": "crp_median",
        "q1_crp": "crp_q1",
        "q3_crp": "crp_q3",
        "p10_crp": "crp_p10",
        "p90_crp": "crp_p90",
        "avg_crp_roc": "crp_avg_roc",
        "max_crp_roc": "crp_max_roc",
        "mean_time_diff_hours": "crp_mean_time_diff_hours",
        "mean_valuenum_diff": "crp_mean_valuenum_diff",
        "fluctuation_index_abs": "crp_fluctuation_index_abs",
        "fluctuation_index": "crp_fluctuation_index",
        "trend_code": "crp_trend_code",
        "avg_crp_grade_code": "crp_avg_grade_code",
        "grade_0_ratio": "crp_grade_0_ratio",
        "grade_1_ratio": "crp_grade_1_ratio",
        "grade_2_ratio": "crp_grade_2_ratio",
        "grade_3_ratio": "crp_grade_3_ratio",
        "grade_4_ratio": "crp_grade_4_ratio",
    }

    out = out.rename(
        columns={
            source: target
            for source, target in legacy_aliases.items()
            if source in out.columns
            and target not in out.columns
        }
    )

    numeric_candidates = [
        "crp_mean",
        "crp_std",
        "crp_max",
        "crp_min",
        "crp_median",
        "crp_q1",
        "crp_q3",
        "crp_p10",
        "crp_p90",
        "crp_avg_roc",
        "crp_max_roc",
        "crp_mean_time_diff_hours",
        "crp_fluctuation_index_abs",
        "crp_measurement_count",
        "crp_avg_grade_code",
        "crp_grade_4_ratio",
        "crp_trend_code",
    ]

    for column in numeric_candidates:
        if column in out.columns:
            out[column] = to_numeric_series(
                out[column]
            )

    if "crp_high_grade_occurrence" not in out.columns:
        required = [
            "crp_grade_4_ratio",
            "crp_mean_time_diff_hours",
        ]
        missing = [
            column
            for column in required
            if column not in out.columns
        ]

        if missing:
            raise ValueError(
                "Cannot calculate crp_high_grade_occurrence; "
                f"missing columns: {missing}"
            )

        out["crp_high_grade_occurrence"] = (
            out["crp_grade_4_ratio"]
            / (
                out["crp_mean_time_diff_hours"]
                + EPS
            )
        )
    else:
        out["crp_high_grade_occurrence"] = to_numeric_series(
            out["crp_high_grade_occurrence"]
        )

    return out


def build_patient_level_indices(
    crp_path: Path,
    wbc_path: Path,
    subject_path: Path,
    death_if_method: str,
    cohort_name: str,
) -> pd.DataFrame:
    print(
        f"[{cohort_name}] Reading CRP: {crp_path}"
    )
    crp = add_crp_new_metrics_if_needed(
        read_csv_normalized(crp_path)
    )

    print(
        f"[{cohort_name}] Reading WBC: {wbc_path}"
    )
    wbc = read_csv_normalized(
        wbc_path
    )

    wbc_legacy_aliases = {
        "measurement_count": "wbc_measurement_count",
        "mean_wbc": "wbc_mean",
        "std_wbc": "wbc_std",
        "max_wbc": "wbc_max",
        "min_wbc": "wbc_min",
        "median_wbc": "wbc_median",
        "q1_wbc": "wbc_q1",
        "q3_wbc": "wbc_q3",
        "p10_wbc": "wbc_p10",
        "p90_wbc": "wbc_p90",
        "avg_wbc_roc": "wbc_avg_roc",
        "max_wbc_roc": "wbc_max_roc",
        "mean_time_diff_hours": "wbc_mean_time_diff_hours",
        "mean_valuenum_diff": "wbc_mean_valuenum_diff",
        "fluctuation_index_abs": "wbc_fluctuation_index_abs",
        "fluctuation_index": "wbc_fluctuation_index",
        "trend_code": "wbc_trend_code",
        "avg_wbc_grade_code": "wbc_avg_grade_code",
        "grade_0_ratio": "wbc_grade_0_ratio",
        "grade_1_ratio": "wbc_grade_1_ratio",
        "grade_2_ratio": "wbc_grade_2_ratio",
        "grade_3_ratio": "wbc_grade_3_ratio",
        "grade_4_ratio": "wbc_grade_4_ratio",
        "grade_5_ratio": "wbc_grade_5_ratio",
        "grade_6_ratio": "wbc_grade_6_ratio",
    }

    wbc = wbc.rename(
        columns={
            source: target
            for source, target in wbc_legacy_aliases.items()
            if source in wbc.columns
            and target not in wbc.columns
        }
    )

    subjects = read_csv_normalized(
        subject_path
    )

    if "subject_id" not in subjects.columns:
        raise ValueError(
            f"{subject_path} does not contain subject_id."
        )

    valid_ids = set(
        pd.to_numeric(
            subjects["subject_id"],
            errors="coerce",
        )
        .dropna()
        .astype(np.int64)
        .tolist()
    )

    for dataframe in [crp, wbc]:
        if "subject_id" not in dataframe.columns:
            raise ValueError(
                "CRP/WBC input does not contain subject_id."
            )

        dataframe["subject_id"] = pd.to_numeric(
            dataframe["subject_id"],
            errors="coerce",
        )

    crp = crp[
        crp["subject_id"].isin(valid_ids)
    ].copy()

    wbc = wbc[
        wbc["subject_id"].isin(valid_ids)
    ].copy()

    if "death_if" in crp.columns:
        death_agg = (
            crp.groupby(
                "subject_id",
                sort=False,
            )["death_if"]
            .apply(
                lambda values: aggregate_binary(
                    values,
                    death_if_method,
                )
            )
            .rename("death_if")
            .reset_index()
        )
    else:
        death_agg = pd.DataFrame(
            columns=[
                "subject_id",
                "death_if",
            ]
        )

    crp_patient = (
        crp.drop(
            columns=["death_if"],
            errors="ignore",
        )
        .drop_duplicates(
            subset="subject_id",
            keep="first",
        )
    )

    wbc_patient = (
        wbc.drop_duplicates(
            subset="subject_id",
            keep="first",
        )
    )

    merged = pd.merge(
        crp_patient,
        wbc_patient,
        on="subject_id",
        how="inner",
        suffixes=("", "_wbcdup"),
    )

    if not death_agg.empty:
        merged = merged.merge(
            death_agg,
            on="subject_id",
            how="left",
        )

    for column in REQUIRED_INDEX_SOURCE_COLUMNS:
        if column not in merged.columns:
            raise ValueError(
                f"[{cohort_name}] Required source column is missing: {column}"
            )
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

    if "crp_high_grade_occurrence" not in merged.columns:
        merged["crp_high_grade_occurrence"] = (
            merged["crp_grade_4_ratio"]
            / (
                merged["crp_mean_time_diff_hours"]
                + EPS
            )
        )
    else:
        merged["crp_high_grade_occurrence"] = pd.to_numeric(
            merged["crp_high_grade_occurrence"],
            errors="coerce",
        )

    lambda_10 = (
        merged["crp_min"]
        / (
            merged["crp_p10"]
            + EPS
        )
    )

    crp_burden = (
        merged["crp_median"]
        + merged["crp_max"]
    )

    dt = (
        merged["crp_mean_time_diff_hours"]
        + EPS
    )

    cidi_t_product = (
        lambda_10
        * crp_burden
        * merged["wbc_max"]
        * merged["wbc_std"]
        / dt
    )

    cidi_lvh_product = (
        lambda_10
        * crp_burden
        * merged["wbc_max"]
        * merged["wbc_std"]
        * merged["crp_high_grade_occurrence"]
    )

    cidi_lvh_s_product = (
        crp_burden
        * merged["wbc_std"]
        * merged["crp_high_grade_occurrence"]
    )

    merged["CIDI-T"] = safe_log_product(
        cidi_t_product
    )
    merged["CIDI-LVH"] = safe_log_product(
        cidi_lvh_product
    )
    merged["CIDI-LVH-S"] = safe_log_product(
        cidi_lvh_s_product
    )

    keep_columns = [
        "subject_id",
        "death_if",
        "CIDI-T",
        "CIDI-LVH",
        "CIDI-LVH-S",
        "crp_median",
        "crp_max",
        "crp_mean_time_diff_hours",
        "crp_grade_4_ratio",
        "crp_high_grade_occurrence",
        "wbc_std",
        "wbc_max",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in merged.columns
    ]

    result = (
        merged[keep_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .drop_duplicates(
            subset="subject_id",
            keep="first",
        )
        .copy()
    )

    result["cohort"] = cohort_name

    print(
        f"[{cohort_name}] Patient-level merged records: {len(result)}"
    )

    return result


# =============================================================================
# 8. OUTCOME TABLES
# =============================================================================

def merge_optional_columns(
    base: pd.DataFrame,
    supplement: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    available = [
        column
        for column in columns
        if column in supplement.columns
    ]

    if not available:
        return base

    supplement_use = (
        supplement[
            ["subject_id"] + available
        ]
        .drop_duplicates(
            subset="subject_id",
            keep="first",
        )
    )

    output = base.merge(
        supplement_use,
        on="subject_id",
        how="left",
        suffixes=("", "_supplement"),
    )

    for column in available:
        supplement_column = f"{column}_supplement"

        if supplement_column not in output.columns:
            continue

        if column not in output.columns:
            output[column] = output[supplement_column]
        else:
            output[column] = output[column].where(
                output[column].notna(),
                output[supplement_column],
            )

        output = output.drop(
            columns=[supplement_column]
        )

    return output


def load_patient_level_outcome_file(
    path: Path,
) -> pd.DataFrame:
    outcomes = read_csv_normalized(
        path
    )

    if "subject_id" not in outcomes.columns:
        raise ValueError(
            f"Outcome file lacks subject_id: {path}"
        )

    outcomes["subject_id"] = pd.to_numeric(
        outcomes["subject_id"],
        errors="coerce",
    )

    for outcome in OUTCOME_COLUMNS:
        if outcome in outcomes.columns:
            outcomes[outcome] = binary_clean(
                outcomes[outcome]
            )

    for duration in DURATION_MAP.values():
        if duration in outcomes.columns:
            outcomes[duration] = pd.to_numeric(
                outcomes[duration],
                errors="coerce",
            )

    if "age" in outcomes.columns:
        outcomes["age"] = pd.to_numeric(
            outcomes["age"],
            errors="coerce",
        )

    if "sex" in outcomes.columns:
        outcomes["sex"] = standardize_sex(
            outcomes["sex"]
        )

    return outcomes


def standardize_sex(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    output.loc[numeric == 0] = 0.0
    output.loc[numeric == 1] = 1.0

    text = series.astype(str).str.strip().str.upper()
    output.loc[text.isin(["F", "FEMALE"])] = 0.0
    output.loc[text.isin(["M", "MALE"])] = 1.0

    return output


def choose_first_icu_stay(
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    sort_columns = [
        column
        for column in [
            "subject_id",
            "intime",
            "outtime",
        ]
        if column in icustays.columns
    ]

    if sort_columns:
        icustays = icustays.sort_values(
            sort_columns
        )

    return icustays.drop_duplicates(
        subset="subject_id",
        keep="first",
    )


def days_between(
    end: pd.Series,
    start: pd.Series,
) -> pd.Series:
    return (
        end - start
    ).dt.total_seconds() / 86400.0


def derive_mimic3_outcomes_from_raw(
    patients_path: Path,
    admissions_path: Path,
    icustays_path: Path,
    subject_ids: Iterable[int],
) -> pd.DataFrame:
    print_banner(
        "Deriving MIMIC-III endpoints from raw tables"
    )

    patients = read_csv_normalized(
        patients_path
    )
    admissions = read_csv_normalized(
        admissions_path
    )
    icustays = read_csv_normalized(
        icustays_path
    )

    subject_set = set(
        int(value)
        for value in subject_ids
        if pd.notna(value)
    )

    for dataframe in [
        patients,
        admissions,
        icustays,
    ]:
        if "subject_id" not in dataframe.columns:
            raise ValueError(
                "Raw MIMIC-III table lacks subject_id."
            )

        dataframe["subject_id"] = pd.to_numeric(
            dataframe["subject_id"],
            errors="coerce",
        )

        dataframe.dropna(
            subset=["subject_id"],
            inplace=True,
        )

        dataframe["subject_id"] = dataframe["subject_id"].astype(
            np.int64
        )

        dataframe.drop(
            dataframe[
                ~dataframe["subject_id"].isin(
                    subject_set
                )
            ].index,
            inplace=True,
        )

    datetime_columns = {
        "patients": [
            "dob",
            "dod",
            "dod_hosp",
            "dod_ssn",
        ],
        "admissions": [
            "admittime",
            "dischtime",
            "deathtime",
        ],
        "icustays": [
            "intime",
            "outtime",
        ],
    }

    for column in datetime_columns["patients"]:
        if column in patients.columns:
            patients[column] = pd.to_datetime(
                patients[column],
                errors="coerce",
            )

    for column in datetime_columns["admissions"]:
        if column in admissions.columns:
            admissions[column] = pd.to_datetime(
                admissions[column],
                errors="coerce",
            )

    for column in datetime_columns["icustays"]:
        if column in icustays.columns:
            icustays[column] = pd.to_datetime(
                icustays[column],
                errors="coerce",
            )

    required_icu = [
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
    ]
    missing_icu = [
        column
        for column in required_icu
        if column not in icustays.columns
    ]

    if missing_icu:
        raise ValueError(
            "ICUSTAYS is missing required columns: "
            f"{missing_icu}"
        )

    anchor = (
        choose_first_icu_stay(icustays)
        if USE_FIRST_ICU_STAY
        else icustays.drop_duplicates(
            subset="subject_id",
            keep="first",
        )
    )

    admission_columns = [
        column
        for column in [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "death_in_hosp",
        ]
        if column in admissions.columns
    ]

    admission_use = admissions[
        admission_columns
    ].drop_duplicates(
        subset=[
            column
            for column in [
                "subject_id",
                "hadm_id",
            ]
            if column in admission_columns
        ],
        keep="first",
    )

    anchor = anchor.merge(
        admission_use,
        on=[
            column
            for column in [
                "subject_id",
                "hadm_id",
            ]
            if column in anchor.columns
            and column in admission_use.columns
        ],
        how="left",
    )

    patient_columns = [
        column
        for column in [
            "subject_id",
            "sex",
            "dob",
            "dod",
            "dod_hosp",
            "dod_ssn",
            "death_if",
        ]
        if column in patients.columns
    ]

    anchor = anchor.merge(
        patients[
            patient_columns
        ].drop_duplicates(
            subset="subject_id",
            keep="first",
        ),
        on="subject_id",
        how="left",
        suffixes=("", "_patient"),
    )

    if "sex" in anchor.columns:
        anchor["sex"] = standardize_sex(
            anchor["sex"]
        )

    if "dob" in anchor.columns:
        anchor["age"] = days_between(
            anchor["intime"],
            anchor["dob"],
        ) / 365.2425

        # MIMIC-III dates for patients older than 89 may be shifted.
        anchor.loc[
            anchor["age"] > 89,
            "age",
        ] = 90.0

        anchor.loc[
            anchor["age"] < 0,
            "age",
        ] = np.nan

    dod_candidates = [
        column
        for column in [
            "dod",
            "dod_hosp",
            "dod_ssn",
            "deathtime",
        ]
        if column in anchor.columns
    ]

    if dod_candidates:
        death_dates = anchor[
            dod_candidates
        ].copy()

        anchor["death_date_best"] = death_dates.min(
            axis=1
        )
    else:
        anchor["death_date_best"] = pd.NaT

    if "deathtime" in anchor.columns:
        deathtime = anchor["deathtime"]
    else:
        deathtime = anchor["death_date_best"]

    death_after_icu_start = (
        deathtime.notna()
        & anchor["intime"].notna()
        & (deathtime >= anchor["intime"])
    )

    anchor["death_icu"] = (
        death_after_icu_start
        & anchor["outtime"].notna()
        & (deathtime <= anchor["outtime"])
    ).astype(float)

    if "death_in_hosp" in anchor.columns:
        anchor["death_in_hosp"] = binary_clean(
            anchor["death_in_hosp"]
        )
    else:
        anchor["death_in_hosp"] = (
            death_after_icu_start
            & anchor["dischtime"].notna()
            & (deathtime <= anchor["dischtime"])
        ).astype(float)

    death_days = days_between(
        anchor["death_date_best"],
        anchor["intime"],
    )

    for horizon, outcome_name, duration_name in [
        (28.0, "death_28d", "t_28d_days"),
        (90.0, "death_90d", "t_90d_days"),
        (365.0, "death_1y", "t_1y_days"),
    ]:
        event = (
            death_days.notna()
            & (death_days >= 0)
            & (death_days <= horizon)
        )

        anchor[outcome_name] = event.astype(
            float
        )

        duration = pd.Series(
            horizon,
            index=anchor.index,
            dtype=float,
        )

        duration.loc[event] = death_days.loc[
            event
        ]

        anchor[duration_name] = duration.clip(
            lower=1e-4,
            upper=horizon,
        )

    icu_event_date = deathtime.where(
        anchor["death_icu"] == 1,
        anchor["outtime"],
    )

    anchor["t_icu_days"] = days_between(
        icu_event_date,
        anchor["intime"],
    ).clip(
        lower=1e-4,
    )

    hospital_event_date = deathtime.where(
        anchor["death_in_hosp"] == 1,
        anchor["dischtime"],
    )

    anchor["t_hosp_days"] = days_between(
        hospital_event_date,
        anchor["intime"],
    ).clip(
        lower=1e-4,
    )

    if "death_if" in anchor.columns:
        anchor["death_if"] = binary_clean(
            anchor["death_if"]
        )

    keep = [
        "subject_id",
        "age",
        "sex",
        "death_if",
        "death_icu",
        "death_in_hosp",
        "death_28d",
        "death_90d",
        "death_1y",
        "t_icu_days",
        "t_hosp_days",
        "t_28d_days",
        "t_90d_days",
        "t_1y_days",
    ]

    keep = [
        column
        for column in keep
        if column in anchor.columns
    ]

    result = anchor[
        keep
    ].drop_duplicates(
        subset="subject_id",
        keep="first",
    )

    return result


def enrich_mimic4_outcomes(
    indices: pd.DataFrame,
    outcome_path: Optional[Path],
    core_path: Optional[Path],
) -> pd.DataFrame:
    result = indices.copy()

    if outcome_path is not None:
        outcomes = load_patient_level_outcome_file(
            outcome_path
        )
        result = merge_optional_columns(
            result,
            outcomes,
            list(OUTCOME_COLUMNS)
            + list(DURATION_MAP.values())
            + ["age", "sex"],
        )

    if core_path is not None:
        core = read_csv_normalized(
            core_path
        )

        core_aliases = {
            "icu_mortality": "death_icu",
            "hosp_mortality": "death_in_hosp",
            "d28_mortality": "death_28d",
        }

        core = core.rename(
            columns={
                key: value
                for key, value in core_aliases.items()
                if key in core.columns
                and value not in core.columns
            }
        )

        if "sex" in core.columns:
            core["sex"] = standardize_sex(
                core["sex"]
            )

        result = merge_optional_columns(
            result,
            core,
            list(OUTCOME_COLUMNS)
            + list(DURATION_MAP.values())
            + ["age", "sex"],
        )

    return clean_outcome_columns(
        result
    )


def enrich_mimic3_outcomes(
    indices: pd.DataFrame,
    inputs: ResolvedInputs,
) -> tuple[pd.DataFrame, str]:
    result = indices.copy()
    source_description = (
        "death_if from CRP feature file only"
    )

    if inputs.mimic3_outcomes is not None:
        outcomes = load_patient_level_outcome_file(
            inputs.mimic3_outcomes
        )

        result = merge_optional_columns(
            result,
            outcomes,
            list(OUTCOME_COLUMNS)
            + list(DURATION_MAP.values())
            + ["age", "sex"],
        )

        source_description = (
            f"patient-level outcome file: {inputs.mimic3_outcomes}"
        )

    else:
        raw_complete = all(
            path is not None
            for path in [
                inputs.mimic3_patients,
                inputs.mimic3_admissions,
                inputs.mimic3_icustays,
            ]
        )

        if raw_complete:
            raw_outcomes = derive_mimic3_outcomes_from_raw(
                patients_path=inputs.mimic3_patients,
                admissions_path=inputs.mimic3_admissions,
                icustays_path=inputs.mimic3_icustays,
                subject_ids=result["subject_id"].dropna().tolist(),
            )

            result = merge_optional_columns(
                result,
                raw_outcomes,
                list(OUTCOME_COLUMNS)
                + list(DURATION_MAP.values())
                + ["age", "sex"],
            )

            source_description = (
                "derived from raw PATIENTS/ADMISSIONS/ICUSTAYS; "
                "first ICU stay used as time origin"
            )

    return (
        clean_outcome_columns(result),
        source_description,
    )


def clean_outcome_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    for outcome in OUTCOME_COLUMNS:
        if outcome in result.columns:
            result[outcome] = binary_clean(
                result[outcome]
            )

    for duration in DURATION_MAP.values():
        if duration in result.columns:
            result[duration] = pd.to_numeric(
                result[duration],
                errors="coerce",
            )

            result.loc[
                result[duration] <= 0,
                duration,
            ] = np.nan

    if "age" in result.columns:
        result["age"] = pd.to_numeric(
            result["age"],
            errors="coerce",
        )

    if "sex" in result.columns:
        result["sex"] = standardize_sex(
            result["sex"]
        )

    return result


# =============================================================================
# 9. COHORT SUMMARIES
# =============================================================================

def cohort_summary_rows(
    dataframe: pd.DataFrame,
    cohort_name: str,
) -> list[dict]:
    rows = []

    base = {
        "cohort": cohort_name,
        "total_patient_records": int(
            dataframe["subject_id"].nunique()
        ),
    }

    for outcome in OUTCOME_COLUMNS:
        if outcome not in dataframe.columns:
            continue

        valid = dataframe[outcome].dropna()

        if valid.empty:
            continue

        row = base.copy()
        row.update(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "n_valid": int(len(valid)),
                "events": int(valid.sum()),
                "event_rate": float(valid.mean()),
            }
        )

        rows.append(row)

    return rows


# =============================================================================
# 10. CORRELATION ANALYSIS
# =============================================================================

def calculate_correlations(
    dataframe: pd.DataFrame,
    cohort_name: str,
) -> pd.DataFrame:
    rows = []

    for outcome in OUTCOME_COLUMNS:
        if outcome not in dataframe.columns:
            continue

        for index_name in INDEX_COLUMNS:
            if index_name not in dataframe.columns:
                continue

            use = dataframe[
                [
                    outcome,
                    index_name,
                ]
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if len(use) < 3:
                continue

            if use[outcome].nunique() < 2:
                continue

            spearman_coef, spearman_p = spearmanr(
                use[outcome],
                use[index_name],
            )

            kendall_coef, kendall_p = kendalltau(
                use[outcome],
                use[index_name],
            )

            if use[index_name].std(ddof=0) == 0:
                pearson_coef = np.nan
                pearson_p = np.nan
            else:
                pearson_coef, pearson_p = pearsonr(
                    use[outcome],
                    use[index_name],
                )

            rows.append(
                {
                    "cohort": cohort_name,
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "index": index_name,
                    "n": int(len(use)),
                    "events": int(use[outcome].sum()),
                    "spearman_coef": float(spearman_coef),
                    "spearman_p": float(spearman_p),
                    "kendall_coef": float(kendall_coef),
                    "kendall_p": float(kendall_p),
                    "pearson_coef": float(pearson_coef),
                    "pearson_p": float(pearson_p),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 11. LOGISTIC REGRESSION
# =============================================================================

def fit_univariable_logistic(
    dataframe: pd.DataFrame,
    outcome: str,
    index_name: str,
):
    use = dataframe[
        [
            outcome,
            index_name,
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(use) < 10:
        raise ValueError(
            "Fewer than 10 valid records."
        )

    if use[outcome].nunique() < 2:
        raise ValueError(
            "Outcome has only one class."
        )

    x = sm.add_constant(
        use[[index_name]],
        has_constant="add",
    )

    y = use[outcome].astype(
        float
    )

    model = sm.GLM(
        y,
        x,
        family=sm.families.Binomial(),
    )

    result = model.fit(
        maxiter=500,
        disp=0,
    )

    coefficient = float(
        result.params[index_name]
    )

    standard_error = float(
        result.bse[index_name]
    )

    p_value = float(
        result.pvalues[index_name]
    )

    ci = result.conf_int().loc[
        index_name
    ]

    return {
        "result": result,
        "n": int(len(use)),
        "events": int(y.sum()),
        "coefficient": coefficient,
        "standard_error": standard_error,
        "odds_ratio": float(np.exp(coefficient)),
        "or_ci_low": float(np.exp(ci.iloc[0])),
        "or_ci_high": float(np.exp(ci.iloc[1])),
        "p_value": p_value,
        "training_data": use,
    }


def predict_from_statsmodels(
    fitted_result,
    dataframe: pd.DataFrame,
    index_name: str,
) -> pd.Series:
    x = sm.add_constant(
        dataframe[[index_name]],
        has_constant="add",
    )

    prediction = fitted_result.predict(
        x
    )

    return pd.Series(
        prediction,
        index=dataframe.index,
        dtype=float,
    )


def classification_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_probability,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    rng = np.random.default_rng(
        random_state
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_probability = np.asarray(
        y_probability,
        dtype=float,
    )

    n = len(y_true)

    metric_names = [
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "brier_score",
    ]

    samples = {
        name: []
        for name in metric_names
    }

    for _ in range(n_bootstrap):
        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        y_sample = y_true[
            sample_indices
        ]

        probability_sample = y_probability[
            sample_indices
        ]

        if np.unique(y_sample).size < 2:
            continue

        try:
            metrics = classification_metrics(
                y_sample,
                probability_sample,
                threshold,
            )
        except Exception:
            continue

        for name in metric_names:
            value = metrics[name]
            if np.isfinite(value):
                samples[name].append(
                    float(value)
                )

    output = {
        "bootstrap_valid_replicates": min(
            len(values)
            for values in samples.values()
        )
        if samples
        else 0
    }

    lower_q = BOOTSTRAP_ALPHA / 2
    upper_q = 1 - BOOTSTRAP_ALPHA / 2

    for name, values in samples.items():
        if len(values) < MIN_BOOTSTRAP_VALID:
            output[f"{name}_ci_low"] = np.nan
            output[f"{name}_ci_high"] = np.nan
            continue

        output[f"{name}_ci_low"] = float(
            np.quantile(
                values,
                lower_q,
            )
        )

        output[f"{name}_ci_high"] = float(
            np.quantile(
                values,
                upper_q,
            )
        )

    return output


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> tuple[float, float]:
    probability = np.clip(
        np.asarray(
            y_probability,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    logit_probability = np.log(
        probability / (1 - probability)
    )

    design = sm.add_constant(
        pd.DataFrame(
            {
                "logit_probability": logit_probability
            }
        ),
        has_constant="add",
    )

    try:
        model = sm.GLM(
            y_true,
            design,
            family=sm.families.Binomial(),
        ).fit(
            maxiter=500,
            disp=0,
        )

        intercept = float(
            model.params["const"]
        )

        slope = float(
            model.params["logit_probability"]
        )

        return intercept, slope
    except Exception:
        return np.nan, np.nan


def run_logistic_external_validation(
    mimic4: pd.DataFrame,
    mimic3: pd.DataFrame,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    derivation_rows = []
    external_rows = []
    mimic3_refit_rows = []

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    common_outcomes = [
        outcome
        for outcome in OUTCOME_COLUMNS
        if outcome in mimic4.columns
        and outcome in mimic3.columns
    ]

    for outcome in common_outcomes:
        external_valid_outcome = mimic3[
            outcome
        ].dropna()

        if (
            len(external_valid_outcome) < 10
            or external_valid_outcome.nunique() < 2
        ):
            continue

        prediction_records = []

        for index_position, index_name in enumerate(
            INDEX_COLUMNS
        ):
            try:
                fitted = fit_univariable_logistic(
                    mimic4,
                    outcome,
                    index_name,
                )
            except Exception as exc:
                derivation_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "index": index_name,
                        "status": f"FAILED: {exc}",
                    }
                )
                continue

            derivation_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "index": index_name,
                    "status": "OK",
                    "n": fitted["n"],
                    "events": fitted["events"],
                    "coefficient": fitted["coefficient"],
                    "standard_error": fitted["standard_error"],
                    "odds_ratio": fitted["odds_ratio"],
                    "or_ci_low": fitted["or_ci_low"],
                    "or_ci_high": fitted["or_ci_high"],
                    "p_value": fitted["p_value"],
                }
            )

            external_use = mimic3[
                [
                    "subject_id",
                    outcome,
                    index_name,
                ]
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            if (
                len(external_use) < 10
                or external_use[outcome].nunique() < 2
            ):
                continue

            probabilities = predict_from_statsmodels(
                fitted["result"],
                external_use,
                index_name,
            )

            y_true = external_use[
                outcome
            ].astype(int).to_numpy()

            y_probability = probabilities.to_numpy()

            metrics = classification_metrics(
                y_true,
                y_probability,
                PROBABILITY_THRESHOLD,
            )

            intervals = bootstrap_metric_intervals(
                y_true,
                y_probability,
                PROBABILITY_THRESHOLD,
                N_BOOTSTRAP,
                RANDOM_STATE
                + index_position
                + 1000
                * (
                    OUTCOME_COLUMNS.index(
                        outcome
                    )
                    + 1
                ),
            )

            calibration_intercept, calibration_slope = (
                calibration_intercept_slope(
                    y_true,
                    y_probability,
                )
            )

            external_row = {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS[outcome],
                "index": index_name,
                "status": "OK",
                **metrics,
                **intervals,
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
            }

            external_rows.append(
                external_row
            )

            prediction_frame = external_use.copy()
            prediction_frame[
                "predicted_probability"
            ] = y_probability
            prediction_frame[
                "predicted_class"
            ] = (
                y_probability
                >= PROBABILITY_THRESHOLD
            ).astype(int)

            prediction_frame.to_csv(
                figure_dir.parent
                / (
                    f"predictions_{outcome}_"
                    f"{index_name.replace('-', '_')}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

            prediction_records.append(
                {
                    "index": index_name,
                    "y_true": y_true,
                    "probability": y_probability,
                }
            )

            try:
                mimic3_fit = fit_univariable_logistic(
                    mimic3,
                    outcome,
                    index_name,
                )

                mimic3_refit_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "index": index_name,
                        "status": "OK",
                        "n": mimic3_fit["n"],
                        "events": mimic3_fit["events"],
                        "coefficient": mimic3_fit["coefficient"],
                        "standard_error": mimic3_fit["standard_error"],
                        "odds_ratio": mimic3_fit["odds_ratio"],
                        "or_ci_low": mimic3_fit["or_ci_low"],
                        "or_ci_high": mimic3_fit["or_ci_high"],
                        "p_value": mimic3_fit["p_value"],
                    }
                )
            except Exception as exc:
                mimic3_refit_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "index": index_name,
                        "status": f"FAILED: {exc}",
                    }
                )

        if prediction_records:
            plot_external_roc(
                outcome=outcome,
                records=prediction_records,
                output_path=figure_dir
                / f"ROC_external_{outcome}.pdf",
            )

            plot_external_calibration(
                outcome=outcome,
                records=prediction_records,
                output_path=figure_dir
                / f"Calibration_external_{outcome}.pdf",
            )

    return (
        pd.DataFrame(derivation_rows),
        pd.DataFrame(external_rows),
        pd.DataFrame(mimic3_refit_rows),
    )


def plot_external_roc(
    outcome: str,
    records: list[dict],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(6.2, 5.4)
    )

    axis = figure.add_axes(
        [0.14, 0.14, 0.82, 0.79]
    )

    for record in records:
        fpr, tpr, _ = roc_curve(
            record["y_true"],
            record["probability"],
        )

        auc_value = roc_auc_score(
            record["y_true"],
            record["probability"],
        )

        index_name = record["index"]

        axis.plot(
            fpr,
            tpr,
            linewidth=1.8,
            color=COLOR_MAP.get(
                index_name
            ),
            label=f"{index_name} (AUC={auc_value:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="#777777",
    )

    axis.set_xlabel(
        "False-positive rate"
    )
    axis.set_ylabel(
        "True-positive rate"
    )
    axis.set_xlim(
        0,
        1,
    )
    axis.set_ylim(
        0,
        1.01,
    )
    axis.set_title(
        f"MIMIC-III external validation: {OUTCOME_LABELS[outcome]}"
    )
    axis.grid(
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color=GRID_COLOR,
        alpha=0.7,
    )
    axis.legend(
        loc="lower right",
        frameon=False,
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    figure.savefig(
        output_path.with_suffix(
            ".png"
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def plot_external_calibration(
    outcome: str,
    records: list[dict],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(6.2, 5.4)
    )

    axis = figure.add_axes(
        [0.14, 0.14, 0.82, 0.79]
    )

    for record in records:
        fraction_positive, mean_predicted = calibration_curve(
            record["y_true"],
            record["probability"],
            n_bins=8,
            strategy="quantile",
        )

        axis.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            linewidth=1.5,
            markersize=4,
            color=COLOR_MAP.get(
                record["index"]
            ),
            label=record["index"],
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="#777777",
    )

    axis.set_xlabel(
        "Predicted probability"
    )
    axis.set_ylabel(
        "Observed event proportion"
    )
    axis.set_xlim(
        0,
        1,
    )
    axis.set_ylim(
        0,
        1,
    )
    axis.set_title(
        f"MIMIC-III calibration: {OUTCOME_LABELS[outcome]}"
    )
    axis.grid(
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color=GRID_COLOR,
        alpha=0.7,
    )
    axis.legend(
        loc="upper left",
        frameon=False,
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    figure.savefig(
        output_path.with_suffix(
            ".png"
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# 12. COX REGRESSION
# =============================================================================

def fit_cox_model(
    dataframe: pd.DataFrame,
    outcome: str,
    duration: str,
    index_name: str,
    adjustment_vars: Sequence[str],
) -> dict:
    if not LIFELINES_AVAILABLE:
        raise RuntimeError(
            "lifelines is unavailable."
        )

    columns = [
        duration,
        outcome,
        index_name,
        *adjustment_vars,
    ]

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    use = dataframe[
        columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna().copy()

    use[duration] = pd.to_numeric(
        use[duration],
        errors="coerce",
    )

    use[outcome] = binary_clean(
        use[outcome]
    )

    use = use[
        use[duration] > 0
    ].dropna()

    if len(use) < 20:
        raise ValueError(
            "Fewer than 20 valid observations."
        )

    events = int(
        use[outcome].sum()
    )

    if events < 5:
        raise ValueError(
            "Fewer than 5 events."
        )

    for variable in adjustment_vars:
        if use[variable].nunique() < 2:
            raise ValueError(
                f"Adjustment variable {variable} has no variation."
            )

    cph = CoxPHFitter()
    cph.fit(
        use,
        duration_col=duration,
        event_col=outcome,
        show_progress=False,
    )

    summary = cph.summary.loc[
        index_name
    ]

    ph_p = np.nan

    try:
        ph_test = proportional_hazard_test(
            cph,
            use,
            time_transform="rank",
        )

        if index_name in ph_test.summary.index:
            ph_p = float(
                ph_test.summary.loc[
                    index_name,
                    "p",
                ]
            )
    except Exception:
        pass

    sd_value = float(
        use[index_name].std(
            ddof=1
        )
    )

    coefficient = float(
        summary["coef"]
    )

    return {
        "n": int(len(use)),
        "events": events,
        "coefficient": coefficient,
        "hazard_ratio_per_unit": float(
            summary["exp(coef)"]
        ),
        "hr_ci_low_per_unit": float(
            summary["exp(coef) lower 95%"]
        ),
        "hr_ci_high_per_unit": float(
            summary["exp(coef) upper 95%"]
        ),
        "p_value": float(
            summary["p"]
        ),
        "index_sd": sd_value,
        "hazard_ratio_per_sd": float(
            np.exp(
                coefficient
                * sd_value
            )
        ),
        "ph_test_p": ph_p,
        "concordance_index": float(
            cph.concordance_index_
        ),
    }


def run_cox_models(
    mimic3: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not LIFELINES_AVAILABLE:
        return pd.DataFrame(
            [
                {
                    "status": (
                        "SKIPPED: lifelines unavailable: "
                        f"{LIFELINES_IMPORT_ERROR}"
                    )
                }
            ]
        )

    for outcome in OUTCOME_COLUMNS:
        duration = DURATION_MAP[outcome]

        if (
            outcome not in mimic3.columns
            or duration not in mimic3.columns
        ):
            continue

        for index_name in [
            "CIDI-LVH",
            "CIDI-LVH-S",
        ]:
            for model_name, adjustment_vars in COX_MODEL_SPECS.items():
                try:
                    fitted = fit_cox_model(
                        dataframe=mimic3,
                        outcome=outcome,
                        duration=duration,
                        index_name=index_name,
                        adjustment_vars=adjustment_vars,
                    )

                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": OUTCOME_LABELS[outcome],
                            "duration_column": duration,
                            "index": index_name,
                            "model": model_name,
                            "adjustment_variables": ",".join(
                                adjustment_vars
                            ),
                            "status": "OK",
                            **fitted,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "outcome": outcome,
                            "outcome_label": OUTCOME_LABELS[outcome],
                            "duration_column": duration,
                            "index": index_name,
                            "model": model_name,
                            "adjustment_variables": ",".join(
                                adjustment_vars
                            ),
                            "status": f"SKIPPED/FAILED: {exc}",
                        }
                    )

    return pd.DataFrame(rows)


# =============================================================================
# 13. RESTRICTED CUBIC SPLINE COX
# =============================================================================

def choose_rcs_knots(
    values: pd.Series,
    requested_knots: int,
) -> list[float]:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna().to_numpy()

    if clean.size < 10:
        raise ValueError(
            "Insufficient data for RCS knots."
        )

    quantile_map = {
        5: [
            0.05,
            0.275,
            0.50,
            0.725,
            0.95,
        ],
        4: [
            0.05,
            0.35,
            0.65,
            0.95,
        ],
        3: [
            0.10,
            0.50,
            0.90,
        ],
    }

    for knot_count in range(
        requested_knots,
        2,
        -1,
    ):
        quantiles = quantile_map.get(
            knot_count,
            np.linspace(
                0.05,
                0.95,
                knot_count,
            ).tolist(),
        )

        knots = np.quantile(
            clean,
            quantiles,
        )

        if np.unique(
            knots
        ).size == len(knots):
            return [
                float(value)
                for value in knots
            ]

    raise ValueError(
        "Could not obtain unique RCS knots."
    )


def build_rcs_design(
    dataframe: pd.DataFrame,
    variable: str,
    knots: Sequence[float],
    adjustment_vars: Sequence[str],
):
    spline = dmatrix(
        "cr(x, knots=knots, constraints='center') - 1",
        {
            "x": dataframe[variable].to_numpy(),
            "knots": list(knots),
        },
        return_type="dataframe",
    )

    spline_columns = list(
        spline.columns
    )

    design_info = spline.design_info
    design = spline.copy()

    for adjustment in adjustment_vars:
        design[adjustment] = dataframe[
            adjustment
        ].to_numpy()

    return (
        design,
        spline_columns,
        design_info,
    )


def run_single_rcs(
    dataframe: pd.DataFrame,
    outcome: str,
    duration: str,
    output_curve_csv: Path,
    output_figure_pdf: Path,
) -> dict:
    if not LIFELINES_AVAILABLE:
        raise RuntimeError(
            "lifelines is unavailable."
        )

    if not PATSY_AVAILABLE:
        raise RuntimeError(
            "patsy is unavailable."
        )

    columns = [
        outcome,
        duration,
        PRIMARY_INDEX,
        *RCS_ADJUST_VARS,
    ]

    use = dataframe[
        columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna().copy()

    use[outcome] = binary_clean(
        use[outcome]
    )

    use[duration] = pd.to_numeric(
        use[duration],
        errors="coerce",
    )

    use = use[
        use[duration] > 0
    ].dropna()

    n = len(use)
    events = int(
        use[outcome].sum()
    )

    if n < 30:
        raise ValueError(
            "Fewer than 30 valid records."
        )

    if events < RCS_MIN_EVENTS:
        raise ValueError(
            f"Fewer than {RCS_MIN_EVENTS} events."
        )

    knots = choose_rcs_knots(
        use[PRIMARY_INDEX],
        RCS_N_KNOTS,
    )

    design, spline_columns, design_info = build_rcs_design(
        use,
        PRIMARY_INDEX,
        knots,
        RCS_ADJUST_VARS,
    )

    rcs_frame = pd.concat(
        [
            use[
                [
                    duration,
                    outcome,
                ]
            ].reset_index(
                drop=True
            ),
            design.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    cox_rcs = CoxPHFitter()
    cox_rcs.fit(
        rcs_frame,
        duration_col=duration,
        event_col=outcome,
        show_progress=False,
    )

    linear_columns = [
        duration,
        outcome,
        PRIMARY_INDEX,
        *RCS_ADJUST_VARS,
    ]

    linear_frame = use[
        linear_columns
    ].copy()

    cox_linear = CoxPHFitter()
    cox_linear.fit(
        linear_frame,
        duration_col=duration,
        event_col=outcome,
        show_progress=False,
    )

    likelihood_ratio = 2 * (
        cox_rcs.log_likelihood_
        - cox_linear.log_likelihood_
    )

    df_difference = max(
        len(spline_columns) - 1,
        1,
    )

    nonlinearity_p = float(
        chi2.sf(
            likelihood_ratio,
            df_difference,
        )
    )

    x_min = float(
        use[PRIMARY_INDEX].quantile(
            0.01
        )
    )

    x_max = float(
        use[PRIMARY_INDEX].quantile(
            0.99
        )
    )

    x_grid = np.linspace(
        x_min,
        x_max,
        RCS_GRID_POINTS,
    )

    reference = float(
        use[PRIMARY_INDEX].median()
    )

    grid_spline = build_design_matrices(
        [design_info],
        {
            "x": x_grid,
            "knots": knots,
        },
    )[0]

    ref_spline = build_design_matrices(
        [design_info],
        {
            "x": np.asarray([reference]),
            "knots": knots,
        },
    )[0]

    grid_design = pd.DataFrame(
        np.asarray(grid_spline),
        columns=spline_columns,
    )

    ref_design = pd.DataFrame(
        np.asarray(ref_spline),
        columns=spline_columns,
    )

    for adjustment in RCS_ADJUST_VARS:
        adjustment_value = float(
            use[adjustment].median()
        )

        grid_design[adjustment] = adjustment_value
        ref_design[adjustment] = adjustment_value

    coefficient_names = list(
        cox_rcs.params_.index
    )

    beta = cox_rcs.params_.loc[
        coefficient_names
    ].to_numpy()

    covariance = cox_rcs.variance_matrix_.loc[
        coefficient_names,
        coefficient_names,
    ].to_numpy()

    grid_matrix = grid_design[
        coefficient_names
    ].to_numpy()

    ref_matrix = ref_design[
        coefficient_names
    ].to_numpy()[0]

    difference_matrix = (
        grid_matrix
        - ref_matrix
    )

    log_hr = difference_matrix @ beta

    variance = np.einsum(
        "ij,jk,ik->i",
        difference_matrix,
        covariance,
        difference_matrix,
    )

    standard_error = np.sqrt(
        np.maximum(
            variance,
            0,
        )
    )

    hr = np.exp(
        log_hr
    )

    hr_lower = np.exp(
        log_hr
        - 1.96
        * standard_error
    )

    hr_upper = np.exp(
        log_hr
        + 1.96
        * standard_error
    )

    curve = pd.DataFrame(
        {
            PRIMARY_INDEX: x_grid,
            "hazard_ratio": hr,
            "hr_ci_low": hr_lower,
            "hr_ci_high": hr_upper,
            "reference_value": reference,
        }
    )

    curve.to_csv(
        output_curve_csv,
        index=False,
        encoding="utf-8-sig",
    )

    figure = plt.figure(
        figsize=(6.8, 5.2)
    )

    axis = figure.add_axes(
        [0.14, 0.15, 0.82, 0.79]
    )

    axis.plot(
        x_grid,
        hr,
        color="#7F9DBD",
        linewidth=1.8,
    )

    axis.fill_between(
        x_grid,
        hr_lower,
        hr_upper,
        color="#7F9DBD",
        alpha=0.20,
        linewidth=0,
    )

    axis.axhline(
        1.0,
        color="#666666",
        linestyle="--",
        linewidth=1,
    )

    axis.axvline(
        reference,
        color="#999999",
        linestyle=":",
        linewidth=1,
    )

    axis.set_xlabel(
        PRIMARY_INDEX
    )
    axis.set_ylabel(
        "Hazard ratio"
    )
    axis.set_title(
        f"MIMIC-III RCS: {OUTCOME_LABELS[outcome]}"
    )

    axis.grid(
        linestyle=(0, (2, 3)),
        linewidth=0.7,
        color=GRID_COLOR,
        alpha=0.7,
    )

    axis.text(
        0.02,
        0.98,
        f"Nonlinearity P = {format_p(nonlinearity_p)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    figure.savefig(
        output_figure_pdf,
        bbox_inches="tight",
    )

    figure.savefig(
        output_figure_pdf.with_suffix(
            ".png"
        ),
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return {
        "n": int(n),
        "events": int(events),
        "knots": json.dumps(
            knots,
            ensure_ascii=False,
        ),
        "reference_value": reference,
        "linear_log_likelihood": float(
            cox_linear.log_likelihood_
        ),
        "rcs_log_likelihood": float(
            cox_rcs.log_likelihood_
        ),
        "likelihood_ratio": float(
            likelihood_ratio
        ),
        "df_difference": int(
            df_difference
        ),
        "nonlinearity_p": nonlinearity_p,
    }


def run_rcs_models(
    mimic3: pd.DataFrame,
    curve_dir: Path,
    figure_dir: Path,
) -> pd.DataFrame:
    curve_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not LIFELINES_AVAILABLE:
        return pd.DataFrame(
            [
                {
                    "status": (
                        "SKIPPED: lifelines unavailable: "
                        f"{LIFELINES_IMPORT_ERROR}"
                    )
                }
            ]
        )

    if not PATSY_AVAILABLE:
        return pd.DataFrame(
            [
                {
                    "status": (
                        "SKIPPED: patsy unavailable: "
                        f"{PATSY_IMPORT_ERROR}"
                    )
                }
            ]
        )

    rows = []

    for outcome in OUTCOME_COLUMNS:
        duration = DURATION_MAP[outcome]

        if (
            outcome not in mimic3.columns
            or duration not in mimic3.columns
        ):
            continue

        try:
            result = run_single_rcs(
                dataframe=mimic3,
                outcome=outcome,
                duration=duration,
                output_curve_csv=curve_dir
                / f"RCS_curve_{outcome}.csv",
                output_figure_pdf=figure_dir
                / f"RCS_{outcome}.pdf",
            )

            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "duration_column": duration,
                    "index": PRIMARY_INDEX,
                    "adjustment_variables": ",".join(
                        RCS_ADJUST_VARS
                    ),
                    "status": "OK",
                    **result,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS[outcome],
                    "duration_column": duration,
                    "index": PRIMARY_INDEX,
                    "adjustment_variables": ",".join(
                        RCS_ADJUST_VARS
                    ),
                    "status": f"SKIPPED/FAILED: {exc}",
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# 14. ARTICLE-READY SUMMARY
# =============================================================================

def generate_article_ready_text(
    outcome_source_description: str,
    cohort_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    external_metrics: pd.DataFrame,
    mimic3_refit: pd.DataFrame,
    cox_results: pd.DataFrame,
    rcs_results: pd.DataFrame,
) -> str:
    lines = [
        "MIMIC-III external-validation results",
        "=" * 72,
        "",
        "Analysis protocol",
        "-----------------",
        "All reported validation results were obtained using statistical methods.",
        "Univariable logistic-regression models were fitted in MIMIC-IV and",
        "applied unchanged to MIMIC-III. No neural network or SMOTE was used.",
        f"MIMIC-III endpoint source: {outcome_source_description}.",
        "",
    ]

    if not cohort_summary.empty:
        mimic3_summary = cohort_summary[
            cohort_summary["cohort"]
            == "MIMIC-III"
        ]

        if not mimic3_summary.empty:
            lines.extend(
                [
                    "MIMIC-III cohort summary",
                    "------------------------",
                ]
            )

            for _, row in mimic3_summary.iterrows():
                lines.append(
                    f"{row['outcome_label']}: "
                    f"n={int(row['n_valid'])}, "
                    f"events={int(row['events'])}, "
                    f"event rate={row['event_rate']:.3f}"
                )

            lines.append("")

    if not correlations.empty:
        primary_correlations = correlations[
            (
                correlations["outcome"]
                == "death_if"
            )
            & (
                correlations["index"]
                == PRIMARY_INDEX
            )
        ]

        if not primary_correlations.empty:
            row = primary_correlations.iloc[0]

            lines.extend(
                [
                    "Primary-outcome association",
                    "---------------------------",
                    (
                        f"For {PRIMARY_INDEX}, the MIMIC-III primary-outcome "
                        f"correlations were Spearman r={row['spearman_coef']:.4f} "
                        f"(P={format_p(row['spearman_p'])}), "
                        f"Kendall tau={row['kendall_coef']:.4f} "
                        f"(P={format_p(row['kendall_p'])}), and "
                        f"Pearson r={row['pearson_coef']:.4f} "
                        f"(P={format_p(row['pearson_p'])})."
                    ),
                    "",
                ]
            )

    if not external_metrics.empty:
        primary_metrics = external_metrics[
            (
                external_metrics["outcome"]
                == "death_if"
            )
            & (
                external_metrics["index"]
                == PRIMARY_INDEX
            )
            & (
                external_metrics["status"]
                == "OK"
            )
        ]

        if not primary_metrics.empty:
            row = primary_metrics.iloc[0]

            lines.extend(
                [
                    "Primary-outcome external discrimination",
                    "---------------------------------------",
                    (
                        f"The MIMIC-IV-fitted {PRIMARY_INDEX} logistic model "
                        f"achieved an external ROC-AUC of {row['roc_auc']:.4f} "
                        f"(95% bootstrap CI "
                        f"{row.get('roc_auc_ci_low', np.nan):.4f}-"
                        f"{row.get('roc_auc_ci_high', np.nan):.4f}) in MIMIC-III. "
                        f"At the fixed probability threshold of 0.5, "
                        f"accuracy was {row['accuracy']:.4f}, "
                        f"recall was {row['recall']:.4f}, "
                        f"specificity was {row['specificity']:.4f}, and "
                        f"F1 score was {row['f1']:.4f}."
                    ),
                    "",
                ]
            )

    if not mimic3_refit.empty:
        association = mimic3_refit[
            (
                mimic3_refit["outcome"]
                == "death_if"
            )
            & (
                mimic3_refit["index"]
                == PRIMARY_INDEX
            )
            & (
                mimic3_refit["status"]
                == "OK"
            )
        ]

        if not association.empty:
            row = association.iloc[0]

            lines.extend(
                [
                    "MIMIC-III logistic association",
                    "------------------------------",
                    (
                        f"In a cohort-specific univariable logistic model, "
                        f"each one-unit increase in {PRIMARY_INDEX} was "
                        f"associated with an odds ratio of "
                        f"{row['odds_ratio']:.4f} "
                        f"(95% CI {row['or_ci_low']:.4f}-"
                        f"{row['or_ci_high']:.4f}, "
                        f"P={format_p(row['p_value'])})."
                    ),
                    "",
                ]
            )

    if not cox_results.empty and "status" in cox_results.columns:
        valid_cox = cox_results[
            cox_results["status"]
            == "OK"
        ]

        if not valid_cox.empty:
            lines.extend(
                [
                    "MIMIC-III Cox results",
                    "---------------------",
                ]
            )

            for _, row in valid_cox.iterrows():
                lines.append(
                    f"{row['outcome_label']}, {row['index']}, "
                    f"{row['model']}: HR={row['hazard_ratio_per_unit']:.4f} "
                    f"(95% CI {row['hr_ci_low_per_unit']:.4f}-"
                    f"{row['hr_ci_high_per_unit']:.4f}), "
                    f"P={format_p(row['p_value'])}; "
                    f"n={int(row['n'])}, events={int(row['events'])}."
                )

            lines.append("")

    if not rcs_results.empty and "status" in rcs_results.columns:
        valid_rcs = rcs_results[
            rcs_results["status"]
            == "OK"
        ]

        if not valid_rcs.empty:
            lines.extend(
                [
                    "MIMIC-III RCS results",
                    "---------------------",
                ]
            )

            for _, row in valid_rcs.iterrows():
                lines.append(
                    f"{row['outcome_label']}: "
                    f"nonlinearity P={format_p(row['nonlinearity_p'])}; "
                    f"n={int(row['n'])}, events={int(row['events'])}."
                )

            lines.append("")

    lines.extend(
        [
            "Interpretation note",
            "-------------------",
            "External discrimination should be reported from 06_external_validation_metrics.csv.",
            "The cohort-specific ORs in 07_mimic3_refit_logistic_associations.csv",
            "describe association replication and should not replace external prediction metrics.",
            "Cox/RCS claims should be made only for rows with status=OK.",
            "",
        ]
    )

    return "\n".join(
        lines
    )


# =============================================================================
# 15. MAIN
# =============================================================================

def main() -> None:
    print_banner(
        "MIMIC-III ALL STATISTICAL VALIDATION"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = OUTPUT_DIR / "figures"
    rcs_curve_dir = OUTPUT_DIR / "rcs_curves"

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inputs = resolve_inputs()

    audit_text = build_input_audit(
        inputs
    )

    write_text(
        OUTPUT_DIR / "00_input_audit.txt",
        audit_text,
    )

    print(
        audit_text
    )

    print_banner(
        "Building patient-level composite indices"
    )

    mimic4_indices = build_patient_level_indices(
        crp_path=inputs.mimic4_crp,
        wbc_path=inputs.mimic4_wbc,
        subject_path=inputs.mimic4_subjects,
        death_if_method=MIMIC4_DEATH_IF_AGGREGATION,
        cohort_name="MIMIC-IV",
    )

    mimic3_indices = build_patient_level_indices(
        crp_path=inputs.mimic3_crp,
        wbc_path=inputs.mimic3_wbc,
        subject_path=inputs.mimic3_subjects,
        death_if_method=MIMIC3_DEATH_IF_AGGREGATION,
        cohort_name="MIMIC-III",
    )

    mimic4 = enrich_mimic4_outcomes(
        indices=mimic4_indices,
        outcome_path=inputs.mimic4_outcomes,
        core_path=inputs.mimic4_core,
    )

    mimic3, mimic3_outcome_source = enrich_mimic3_outcomes(
        indices=mimic3_indices,
        inputs=inputs,
    )

    mimic4.to_csv(
        OUTPUT_DIR / "01_mimic4_patient_level_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mimic3.to_csv(
        OUTPUT_DIR / "02_mimic3_patient_level_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print_banner(
        "Cohort summaries"
    )

    cohort_summary = pd.DataFrame(
        cohort_summary_rows(
            mimic4,
            "MIMIC-IV",
        )
        + cohort_summary_rows(
            mimic3,
            "MIMIC-III",
        )
    )

    cohort_summary.to_csv(
        OUTPUT_DIR / "03_cohort_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not cohort_summary.empty:
        print(
            cohort_summary.to_string(
                index=False
            )
        )

    print_banner(
        "MIMIC-III correlation analysis"
    )

    correlations = calculate_correlations(
        mimic3,
        "MIMIC-III",
    )

    correlations.to_csv(
        OUTPUT_DIR / "04_mimic3_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not correlations.empty:
        print(
            correlations.to_string(
                index=False
            )
        )
    else:
        print(
            "No valid MIMIC-III correlation result could be calculated."
        )

    print_banner(
        "Derivation logistic models and external validation"
    )

    (
        derivation_models,
        external_metrics,
        mimic3_refit,
    ) = run_logistic_external_validation(
        mimic4=mimic4,
        mimic3=mimic3,
        figure_dir=figure_dir,
    )

    derivation_models.to_csv(
        OUTPUT_DIR / "05_derivation_logistic_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    external_metrics.to_csv(
        OUTPUT_DIR / "06_external_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mimic3_refit.to_csv(
        OUTPUT_DIR / "07_mimic3_refit_logistic_associations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not external_metrics.empty:
        display_columns = [
            column
            for column in [
                "outcome",
                "index",
                "n",
                "events",
                "accuracy",
                "recall",
                "specificity",
                "f1",
                "roc_auc",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
            ]
            if column in external_metrics.columns
        ]

        print(
            external_metrics[
                display_columns
            ].to_string(
                index=False
            )
        )
    else:
        print(
            "No external logistic result could be calculated."
        )

    print_banner(
        "MIMIC-III Cox models"
    )

    cox_results = run_cox_models(
        mimic3
    )

    cox_results.to_csv(
        OUTPUT_DIR / "08_mimic3_cox_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not cox_results.empty:
        print(
            cox_results.to_string(
                index=False
            )
        )

    print_banner(
        "MIMIC-III restricted cubic spline models"
    )

    rcs_results = run_rcs_models(
        mimic3=mimic3,
        curve_dir=rcs_curve_dir,
        figure_dir=figure_dir,
    )

    rcs_results.to_csv(
        OUTPUT_DIR / "09_mimic3_rcs_nonlinearity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not rcs_results.empty:
        print(
            rcs_results.to_string(
                index=False
            )
        )

    article_text = generate_article_ready_text(
        outcome_source_description=mimic3_outcome_source,
        cohort_summary=cohort_summary,
        correlations=correlations,
        external_metrics=external_metrics,
        mimic3_refit=mimic3_refit,
        cox_results=cox_results,
        rcs_results=rcs_results,
    )

    write_text(
        OUTPUT_DIR / "10_article_ready_results.txt",
        article_text,
    )

    run_manifest = {
        "protocol": {
            "analysis_type": "statistical external validation",
            "neural_network_used": False,
            "smote_used": False,
            "logistic_training_cohort": "MIMIC-IV",
            "external_validation_cohort": "MIMIC-III",
            "probability_threshold": PROBABILITY_THRESHOLD,
            "bootstrap_replicates": N_BOOTSTRAP,
            "primary_index": PRIMARY_INDEX,
            "indices": INDEX_COLUMNS,
        },
        "inputs": {
            key: safe_json_value(value)
            for key, value in inputs.__dict__.items()
        },
        "mimic3_outcome_source": mimic3_outcome_source,
        "outputs": {
            "directory": str(OUTPUT_DIR),
        },
    }

    write_text(
        OUTPUT_DIR / "run_manifest.json",
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
        ),
    )

    print_banner(
        "COMPLETED"
    )

    print(
        article_text
    )

    print(
        f"\nAll outputs were saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print_banner(
            "FAILED"
        )

        traceback.print_exc()

        try:
            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            write_text(
                OUTPUT_DIR / "FAILED_TRACEBACK.txt",
                traceback.format_exc(),
            )

            print(
                "\nThe complete traceback was also saved to:\n"
                f"{OUTPUT_DIR / 'FAILED_TRACEBACK.txt'}"
            )
        except Exception:
            pass

        sys.exit(1)
