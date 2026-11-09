#!/usr/bin/env python3
# =============================================================================
#  Speciation- and duration-aware assessment of the olive flounder blood panel
#  for arsenic biomonitoring.
#
#  SINGLE-FILE PIPELINE.  Produces every result, figure and LaTeX table used in
#  the accompanying paper.
#
#      python3 arsenic_pipeline.py --uploads /path/to/workbooks
#
#  Dependencies: numpy pandas scipy scikit-learn matplotlib openpyxl
#  No deep-learning framework is required; the graph convolutional networks are
#  implemented directly with analytic gradients, which is what makes the exact
#  architectural equivalence check in Section 3 possible.
#
#  Outputs:  out/*.csv      every numerical result
#            figs/*.pdf     every figure (Computer Modern, bold throughout)
#            tables/*.tex   every LaTeX table, column counts self-validated
# =============================================================================
from __future__ import annotations

import os

# -----------------------------------------------------------------------------
#  Windows guard: joblib/loky probes physical cores by shelling out to `wmic`,
#  which Windows 11 removed. The subprocess call then raises inside
#  loky.backend.context._count_physical_cores. Setting these before scikit-learn
#  is imported skips the probe entirely. Harmless on Linux and macOS.
# -----------------------------------------------------------------------------
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")


import argparse
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT, FIGS, TABS = ROOT / "out", ROOT / "figs", ROOT / "tables"
for _d in (OUT, FIGS, TABS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
INF = math.inf


# =============================================================================
#  PROGRESS LOGGING
#
#  Windows buffers stdout aggressively when output is redirected, so every
#  write is flushed explicitly. Without that the pipeline looks frozen for
#  twenty minutes. Set ARSENIC_QUIET=1 to silence the detail lines.
# =============================================================================
import time as _time

_T0 = _time.time()
_QUIET = os.environ.get("ARSENIC_QUIET", "") == "1"
_STEP = {"n": 0, "total": 0}


def log(msg: str = "", level: int = 0, always: bool = False):
    """level 0 = heading, 1 = step, 2 = detail."""
    if _QUIET and level >= 2 and not always:
        return
    el = _time.time() - _T0
    stamp = f"[{int(el // 60):02d}:{el % 60:05.2f}]"
    prefix = {0: "", 1: "   - ", 2: "       . "}[min(level, 2)]
    print(f"{stamp} {prefix}{msg}", flush=True)


def log_rule(title: str = ""):
    print("=" * 78, flush=True)
    if title:
        print(f"  {title}", flush=True)
        print("=" * 78, flush=True)


def log_step(title: str):
    _STEP["n"] += 1
    n, t = _STEP["n"], _STEP["total"]
    bar = f"[{n}/{t}]" if t else f"[{n}]"
    log("")
    log(f"{bar} {title}", 0)


# =============================================================================
#  SECTION 0.  Figure style: Computer Modern, bold everywhere.
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.default": "bf",
    "axes.formatter.use_mathtext": True,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.1,
    "axes.unicode_minus": False,
    "figure.titleweight": "bold",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.3",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def bolden(ax, size=9):
    """Force every tick label and legend entry on an axis to bold."""
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_fontweight("bold")
        lab.set_fontsize(size)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontweight("bold")
            t.set_fontsize(size - 1)


PALETTE = {"As3_D10": "#15476b", "As3_D20": "#4a90c4",
           "As5_D10": "#8c2f39", "As5_D20": "#d98324"}

# =============================================================================
#  SECTION 1.  Data: workbooks, biomarkers, experimental design.
# =============================================================================
BIOMARKERS = ["Hb", "Hct", "RBC count", "Calcium", "Magnesium", "Glucose",
              "Cholesterol", "Total protein", "AST", "ALT", "ALP"]

SYMBOL = {"Hb": "Hb", "Hct": "Hct", "RBC count": "RBC", "Calcium": "Ca",
          "Magnesium": "Mg", "Glucose": "Gluc", "Cholesterol": "Chol",
          "Total protein": "TP", "AST": "AST", "ALT": "ALT", "ALP": "ALP"}

CATEGORY = {"Hb": "haematological", "Hct": "haematological",
            "RBC count": "haematological", "Calcium": "electrolyte",
            "Magnesium": "electrolyte", "Glucose": "metabolic",
            "Cholesterol": "metabolic", "Total protein": "biochemical",
            "AST": "hepatic", "ALT": "hepatic", "ALP": "hepatic"}

HEPATIC = ["AST", "ALT", "ALP"]
NON_HEPATIC = [b for b in BIOMARKERS if b not in HEPATIC]

# -----------------------------------------------------------------------------
#  CORRECTED LOEC TABLE.
#
#  Supplied by the experimentalists and transcribed from their revision note.
#  Each entry is the LOWEST tested concentration (mg/L) at which the group mean
#  differs significantly from the control group (one-way ANOVA with Tukey's
#  test, p < 0.05), i.e. a Lowest Observed Effect Concentration.  `INF` denotes
#  "no significant difference at any tested concentration", so that biomarker
#  can never flag abnormality under that condition.
#
#  Three values differ from Table 3 of the submitted manuscript:
#     As5+ Day-10 haemoglobin   2.0  ->  INF   (no significant difference)
#     As3+ Day-20 magnesium     2.0  ->  INF   (no significant difference)
#     As3+ Day-10 cholesterol   2.0  ->  8.0   (first significant at 8 mg/L)
# -----------------------------------------------------------------------------
LOEC = {
    ("As3", 10): {"Hb": 2.0, "Hct": 2.0, "RBC count": 1.0, "Calcium": 2.0,
                  "Magnesium": 8.0, "Glucose": 8.0, "Cholesterol": 8.0,
                  "Total protein": 8.0, "AST": 4.0, "ALT": 4.0, "ALP": 8.0},
    ("As3", 20): {"Hb": 0.5, "Hct": 0.5, "RBC count": 0.5, "Calcium": 0.5,
                  "Magnesium": INF, "Glucose": 2.0, "Cholesterol": 0.5,
                  "Total protein": 0.5, "AST": 1.0, "ALT": 2.0, "ALP": 2.0},
    ("As5", 10): {"Hb": INF, "Hct": 4.0, "RBC count": 1.0, "Calcium": 2.0,
                  "Magnesium": 8.0, "Glucose": 8.0, "Cholesterol": 8.0,
                  "Total protein": 4.0, "AST": 2.0, "ALT": 8.0, "ALP": INF},
    ("As5", 20): {"Hb": 4.0, "Hct": 2.0, "RBC count": 2.0, "Calcium": 0.5,
                  "Magnesium": 4.0, "Glucose": 4.0, "Cholesterol": 2.0,
                  "Total protein": 0.5, "AST": 4.0, "ALT": 4.0, "ALP": 4.0},
}

#  Direction of the significant change, as reported alongside each LOEC
#  (감소 = decrease, 증가 = increase).  Consistent across all four conditions.
DIRECTION = {"Hb": -1, "Hct": -1, "RBC count": -1, "Calcium": -1,
             "Magnesium": -1, "Cholesterol": -1, "Total protein": -1,
             "Glucose": +1, "AST": +1, "ALT": +1, "ALP": +1}

#  Concentration ladders actually run (the As5+ arms omit 0.1 mg/L by design).
LADDER = {"As3": [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0],
          "As5": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]}
N_DESIGN = 7                        # fish allocated per concentration group

DAY_HEADERS = {"10일": 10, "20일": 20}


# =============================================================================
#  PROGRESS LOGGING
#
#  Every stage reports what it is doing, on what data, and what it found.
#  Output is flushed immediately so progress is visible on Windows consoles,
#  which buffer aggressively when stdout is redirected.
# =============================================================================
VERBOSE = True
_T0 = None


def _elapsed():
    import time
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0


def banner(text):
    if not VERBOSE:
        return
    print("\n" + "=" * 78, flush=True)
    print(f"  {text}", flush=True)
    print("=" * 78, flush=True)


def step(text):
    if VERBOSE:
        print(f"\n[{_elapsed():6.1f}s] {text}", flush=True)


def log(text, indent=1):
    if VERBOSE:
        print("  " * indent + text, flush=True)


def kv(label, value, indent=2, width=38):
    if VERBOSE:
        print("  " * indent + f"{label:<{width}} {value}", flush=True)




@dataclass(frozen=True)
class Condition:
    species: str
    day: int

    @property
    def key(self):
        return f"{self.species}_D{self.day}"

    @property
    def tex(self):
        """Label for figures: IUPAC oxidation-state notation."""
        rn = "III" if self.species == "As3" else "V"
        return f"As({rn}) D-{self.day}"

    @property
    def plain(self):
        """Label for LaTeX tables."""
        rn = "III" if self.species == "As3" else "V"
        return f"As({rn}) Day-{self.day}"


CONDITIONS = [Condition("As3", 10), Condition("As3", 20),
              Condition("As5", 10), Condition("As5", 20)]
CKEYS = [c.key for c in CONDITIONS]
CTEX = {c.key: c.plain for c in CONDITIONS}
CTEX_PLAIN = {c.key: ("As(III)" if c.species == "As3" else "As(V)")
              + f" D-{c.day}" for c in CONDITIONS}


def load_workbook(path, species):
    ws = openpyxl.load_workbook(path, data_only=True)["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    frames = []
    for i, row in enumerate(rows):
        if row[0] in DAY_HEADERS and i + 1 < len(rows) and \
                isinstance(rows[i + 1][0], str) and rows[i + 1][0].startswith("concentration"):
            day = DAY_HEADERS[row[0]]
            conc_row, samp_row = rows[i + 1], rows[i + 2]
            n = sum(1 for v in samp_row[1:] if v is not None)
            rec = {"species": species, "day": day,
                   "concentration": [float(v) for v in conc_row[1:1 + n]],
                   "fish_id": [f"{species}_D{day}_{int(v):03d}" for v in samp_row[1:1 + n]]}
            for r2 in rows[i + 3: i + 23]:
                if r2[0] in BIOMARKERS:
                    rec[r2[0]] = [None if v is None else float(v) for v in r2[1:1 + n]]
                elif r2[0] is None and len(rec) > 4:
                    break
            frames.append(pd.DataFrame(rec))
    df = pd.concat(frames, ignore_index=True)
    df["dose_group"] = (df.species + "_D" + df.day.astype(str)
                        + "_c" + df.concentration.astype(str))
    return df


def find_workbooks(upload_dir=None):
    """Locate the two workbooks without depending on an absolute path.

    Searches, in order: the directory given on the command line, a `data`
    folder beside this script, the script's own folder, the current working
    directory, and finally the container path used during development.
    Filenames are matched by glob, so `as3_raw_data.xlsx`,
    `as3___raw_data.xlsx` and similar all work.
    """
    here = Path(__file__).resolve().parent
    candidates = []
    if upload_dir:
        candidates.append(Path(upload_dir).expanduser())
    candidates += [here / "data", here, Path.cwd() / "data", Path.cwd(),
                   Path("/mnt/user-data/uploads")]

    tried, seen = [], set()
    for d in candidates:
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if not d.is_dir():
            continue
        a3 = sorted(d.glob("as3*raw*data*.xlsx")) or sorted(d.glob("*as3*.xlsx"))
        a5 = sorted(d.glob("as5*raw*data*.xlsx")) or sorted(d.glob("*as5*.xlsx"))
        if a3 and a5:
            return a3[0], a5[0]

    raise FileNotFoundError(
        "Could not find the two exposure workbooks.\n"
        "Expected one file matching 'as3*.xlsx' and one matching 'as5*.xlsx'.\n"
        "Searched:\n  " + "\n  ".join(tried) + "\n\n"
        "Fix: pass the folder explicitly, e.g.\n"
        '  python arsenic_pipeline.py --uploads "C:/path/to/data"\n'
        "or place the workbooks in a 'data' folder beside this script.")


def load_all(upload_dir=None):
    a3, a5 = find_workbooks(upload_dir)
    log(f"data folder: {a3.parent}", 1)
    log(f"workbooks: {a3.name}, {a5.name}", 2)
    return pd.concat([load_workbook(a3, "As3"),
                      load_workbook(a5, "As5")],
                     ignore_index=True)


def subset(df, cond):
    return df[(df.species == cond.species) & (df.day == cond.day)].reset_index(drop=True)


# =============================================================================
#  SECTION 2.  Outcome definitions.
# =============================================================================
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
#  2b.  Control-referenced physiological abnormality.
#
#  Reference intervals come from CONTROL fish only, estimated robustly because
#  n_control = 7.  A fish is flagged on biomarker m when it deviates in the
#  direction the LOEC table reports for that biomarker, and is abnormal when at
#  least k_min of the eleven flags fire.  Nothing here reads the dose.
# -----------------------------------------------------------------------------
K_SD, K_MIN = 2.5, 2


def reference_intervals(control_df, k=K_SD, markers=None):
    out = {}
    for m in (markers or BIOMARKERS):
        x = control_df[m].dropna().to_numpy()
        med = float(np.median(x))
        sd = 1.4826 * float(np.median(np.abs(x - med)))
        if sd <= 0:
            sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
        out[m] = (med - k * sd, med + k * sd, med, sd)
    return out


def deviation_flags(d, iv, directional=True):
    f = {}
    for m, (lo, hi, _, _) in iv.items():
        x = d[m].to_numpy()
        f[m] = ((x < lo).astype(int) if DIRECTION[m] < 0 else (x > hi).astype(int)) \
            if directional else ((x < lo) | (x > hi)).astype(int)
    return pd.DataFrame(f, index=d.index)


def abnormal_label(d, iv, k_min=K_MIN, directional=True):
    return (deviation_flags(d, iv, directional).to_numpy().sum(1) >= k_min).astype(int)


def ibr_v2(d, control_df, markers=None):
    """Integrated Biomarker Response: summed absolute control-referenced z."""
    s = np.zeros(len(d))
    for m in (markers or BIOMARKERS):
        mu, sd = float(control_df[m].mean()), float(control_df[m].std(ddof=1))
        if sd > 0:
            s += np.abs((d[m].to_numpy() - mu) / sd)
    return s


#  2c.  Design-derived outcomes: no biomarker enters the label.
def exposed_label(d):
    return (d.concentration.to_numpy() > 0).astype(int)


def log_dose(d, floor=0.05):
    return np.log10(np.maximum(d.concentration.to_numpy(), floor))


# =============================================================================
#  SECTION 3.  The proposed model: Directional Control-Referenced Panel (DCRP).
#
#  The LOEC table supplies one piece of information the raw panel does not: the
#  SIGN of the physiological response to arsenic for each biomarker.  DCRP uses
#  it.  Each biomarker is re-expressed as a signed control-referenced deviation
#
#        z_m(x) = s_m * ( x_m - med_m^ctrl ) / sd_m^ctrl ,   s_m in {-1,+1},
#
#  so that a LARGE POSITIVE value always means "displaced in the direction
#  arsenic displaces it", whatever the biomarker.  A summed severity term
#  S(x) = sum_m z_m(x) is appended, giving a 12-dimensional representation that
#  is comparable across biomarkers, across conditions and across species.  A
#  class-weighted penalised logistic model is fitted on that representation.
#
#  All reference constants are estimated from TRAINING-fold controls only, so
#  the transform is refitted inside every cross-validation fold.
# =============================================================================
class DirectionalReference(BaseEstimator, TransformerMixin):
    """Signed control-referenced standardisation plus a severity summary."""

    def __init__(self, markers=None, add_severity=True, k_sd=K_SD):
        self.markers = markers
        self.add_severity = add_severity
        self.k_sd = k_sd

    def fit(self, X, y=None, control_mask=None):
        self.markers_ = list(self.markers or BIOMARKERS)
        X = np.asarray(X, dtype=float)
        ref = X if control_mask is None or control_mask.sum() < 3 else X[control_mask]
        self.med_ = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - self.med_), axis=0) * 1.4826
        fallback = np.std(ref, axis=0, ddof=1) if len(ref) > 1 else np.ones(X.shape[1])
        self.sd_ = np.where(mad > 0, mad, np.where(fallback > 0, fallback, 1.0))
        self.sign_ = np.array([DIRECTION[m] for m in self.markers_], dtype=float)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        z = self.sign_ * (X - self.med_) / self.sd_
        if self.add_severity:
            z = np.column_stack([z, z.sum(axis=1)])
        return z


class DCRP(BaseEstimator):
    """The proposed model: DirectionalReference followed by penalised logistic
    regression.  Exposes the sklearn classifier interface."""

    def __init__(self, C=1.0, penalty="l2", add_severity=True, seed=SEED):
        self.C, self.penalty, self.add_severity, self.seed = C, penalty, add_severity, seed

    def fit(self, X, y, control_mask=None):
        self.tf_ = DirectionalReference(add_severity=self.add_severity)
        self.tf_.fit(X, control_mask=control_mask)
        Z = self.tf_.transform(X)
        solver = "liblinear" if self.penalty == "l1" else "lbfgs"
        self.clf_ = LogisticRegression(C=self.C, penalty=self.penalty, solver=solver,
                                       max_iter=5000, class_weight="balanced",
                                       random_state=self.seed).fit(Z, y)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.tf_.transform(X))

    @property
    def coef_(self):
        return self.clf_.coef_


# =============================================================================
#  SECTION 4.  Baseline model zoo (the comparisons the review asked for).
# =============================================================================
def _pipe(est):
    return Pipeline([("scale", StandardScaler()), ("clf", est)])


def baseline_models(seed=SEED):
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic (L2)": _pipe(LogisticRegression(max_iter=5000,
                                                  class_weight="balanced")),
        "Logistic (L1, CV)": _pipe(LogisticRegressionCV(
            Cs=10, cv=3, penalty="l1", solver="liblinear", max_iter=5000,
            class_weight="balanced", scoring="roc_auc", random_state=seed)),
        "Random forest": RandomForestClassifier(n_jobs=1, n_estimators=500, min_samples_leaf=2,
                                                class_weight="balanced",
                                                random_state=seed),
        "Grad. boosting": HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=7, learning_rate=0.08,
            min_samples_leaf=3, random_state=seed),
        "MLP (32--32)": _pipe(MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-3,
                                            max_iter=4000, random_state=seed)),
    }


# =============================================================================
#  SECTION 6.  Validation protocols, metrics and resampling inference.
# =============================================================================
def random_folds(y, k=5, seed=SEED):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y))


def lodgo(conc, seed=SEED):
    """Leave-One-Dose-Group-Out.

    Fish sharing a nominal concentration shared a tank, water, handling and
    sampling occasion, so they are pseudo-replicates.  Each fold holds out one
    entire exposed dose group; control fish are partitioned once across folds
    so every test set carries both classes while no control fish is ever in
    training and test simultaneously.
    """
    conc = np.asarray(conc, float)
    idx = np.arange(len(conc))
    levels = sorted(set(conc[conc > 0]))
    ctrl = np.random.default_rng(seed).permutation(idx[conc == 0])
    folds = []
    for lvl, part in zip(levels, np.array_split(ctrl, len(levels))):
        te = np.sort(np.concatenate([idx[conc == lvl], part]).astype(int))
        folds.append((np.setdiff1d(idx, te), te))
    return folds


def lodgo_exposed(conc):
    conc = np.asarray(conc, float); idx = np.arange(len(conc))
    return [(np.setdiff1d(idx, idx[conc == l]), idx[conc == l])
            for l in sorted(set(conc))]


def fast_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ss = s[order]
    i, r = 0, 1.0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (2 * r + (j - i)) / 2.0
        r += j - i + 1; i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_bacc(y, s, t=0.5):
    y = np.asarray(y); p = (np.asarray(s) >= t).astype(int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float(0.5 * (((p == 1) & (y == 1)).sum() / n1 +
                        ((p == 0) & (y == 0)).sum() / n0))


def metrics(y, s, t=0.5):
    y = np.asarray(y).astype(int); p = (np.asarray(s) >= t).astype(int)
    out = dict(n=len(y), n_pos=int(y.sum()),
               accuracy=accuracy_score(y, p),
               macro_f1=f1_score(y, p, average="macro", zero_division=0),
               balanced_accuracy=balanced_accuracy_score(y, p),
               roc_auc=roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan)
    if len(np.unique(y)) > 1:
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def cluster_bootstrap(y, s, groups, n_boot=1000, seed=0, alpha=0.05):
    """Percentile intervals resampling whole dose groups, not individual fish."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups); y = np.asarray(y); s = np.asarray(s, float)
    member = [np.where(groups == g)[0] for g in np.unique(groups)]
    A, B = [], []
    for _ in range(n_boot):
        idx = np.concatenate([member[k] for k in
                              rng.integers(0, len(member), len(member))])
        yt, st = y[idx], s[idx]
        if 0 < yt.sum() < len(yt):
            A.append(fast_auc(yt, st)); B.append(fast_bacc(yt, st))
    q = lambda v: (np.nan, np.nan) if not v else (
        float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
    return q(A), q(B)


def fit_predict_oof(make_model, X, y, folds, conc=None):
    """Pooled out-of-fold scores.  DCRP receives the training-fold control mask."""
    X = np.asarray(X, float); y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model()
        if isinstance(model, DCRP) and conc is not None:
            model.fit(X[tr], y[tr], control_mask=(np.asarray(conc)[tr] == 0))
        else:
            model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def permutation_test(score_fn, y, n_perm=300, seed=1):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    obs = fast_auc(y, score_fn(y))
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        v = fast_auc(yp, score_fn(yp))
        if np.isfinite(v):
            null.append(v)
    null = np.asarray(null)
    return dict(observed=obs, p_value=float((1 + (null >= obs).sum()) / (1 + len(null))),
                null_mean=float(null.mean()), null_sd=float(null.std()),
                n_perm=len(null))


# =============================================================================
#  SECTION 7.  LaTeX table writer (self-validating column counts).
# =============================================================================
def write_table(name, rows, spec, header):
    NL = chr(92) * 2
    body = [("  " + r.rstrip().rstrip(chr(92)).rstrip() + " " + NL)
            for r in rows if r.strip()]
    ncol = max(r.count("&") for r in body) + 1
    nspec = len([c for c in re.findall(r"[lcr]|p\{[^}]*\}", spec)])
    assert nspec == ncol, f"{name}: spec {nspec} cols vs data {ncol}"
    assert header.count("&") + 1 == ncol, f"{name}: header width"
    txt = ([chr(92) + "begin{tabular}{" + spec + "}", "  " + chr(92) + "toprule",
            "  " + header + " " + NL, "  " + chr(92) + "midrule"] + body +
           ["  " + chr(92) + "bottomrule", chr(92) + "end{tabular}"])
    (TABS / name).write_text(chr(10).join(txt) + chr(10))
    log(f"table {name}  ({len(body)} rows x {ncol} cols)", 2)


def fmt(v, nd=2, dash="--"):
    return dash if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


# =============================================================================
#  EXPERIMENT 1.  Effect thresholds (LOEC) and response directions.
# =============================================================================
def exp1_loec_table(df):
    """Effect thresholds and response directions -- Table 1 of the paper."""
    log_step("E1  LOEC table and response directions")
    rows = []
    for m in BIOMARKERS:
        r = {"biomarker": m, "symbol": SYMBOL[m], "category": CATEGORY[m],
             "direction": "increase" if DIRECTION[m] > 0 else "decrease"}
        for c in CONDITIONS:
            r[c.key] = LOEC[(c.species, c.day)][m]
        rows.append(r)
    loec_df = pd.DataFrame(rows)
    loec_df.to_csv(OUT / "e1_loec_table.csv", index=False)

    for c in CONDITIONS:
        tau = LOEC[(c.species, c.day)]
        finite = {m: v for m, v in tau.items() if np.isfinite(v)}
        ns = [SYMBOL[m] for m, v in tau.items() if not np.isfinite(v)]
        lowest = min(finite.values())
        first = [SYMBOL[m] for m, v in finite.items() if v == lowest]
        log(f"{CTEX_PLAIN[c.key]}: {len(finite)}/11 biomarkers respond; "
            f"lowest at {lowest:g} mg/L ({', '.join(first)})"
            + (f"; n.s.: {', '.join(ns)}" if ns else ""))

    arrow = {1: r"$\uparrow$", -1: r"$\downarrow$"}
    loec_rows = []
    for _, r in loec_df.iterrows():
        cells = " & ".join("n.s." if not np.isfinite(r[k]) else f"{r[k]:g}"
                           for k in CKEYS)
        loec_rows.append(f"{r['symbol']} & {r['category']} & "
                         f"{arrow[DIRECTION[r['biomarker']]]} & {cells}" + r" \\")
    write_table("t1_loec.tex", loec_rows, "llc" + "c" * 4,
                r"Biomarker & Category & Dir. & " + " & ".join(CTEX[k] for k in CKEYS))
    return loec_df


# =============================================================================
#  EXPERIMENT 3.  Calibration of the control-referenced abnormality rule.
# =============================================================================
def exp3_calibration(df):
    log_step("E3  calibration of the abnormality rule")
    rows = []
    for c in CONDITIONS:
        d = subset(df, c); ctrl = d[d.concentration == 0]
        for directional in (False, True):
            for k in (2.0, 2.5, 3.0):
                loo = []
                for i in range(len(ctrl)):
                    ref = reference_intervals(ctrl.drop(ctrl.index[i]), k=k)
                    loo.append(abnormal_label(ctrl.iloc[[i]], ref,
                                              directional=directional)[0])
                iv = reference_intervals(ctrl, k=k)
                y = abnormal_label(d, iv, directional=directional)
                top = (d.concentration == d.concentration.max()).to_numpy()
                rows.append(dict(condition=c.key, directional=directional, k_sd=k,
                                 loo_control_fp=float(np.mean(loo)),
                                 prevalence=float(y.mean()),
                                 top_dose_rate=float(y[top].mean())))
    for x in rows:
        if x["directional"]:
            log(f"k={x['k_sd']:.1f} {CTEX_PLAIN[x['condition']]}: "
                f"LOO control FP={x['loo_control_fp']:.2f}, "
                f"top-dose rate={x['top_dose_rate']:.2f}")
    log(f"adopted: k={K_SD}, k_min={K_MIN}, directional flags", 1)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "e3_calibration.csv", index=False)
    sel = r[r.directional]
    write_table("t4_calibration.tex",
                [f"{k:g} & " + " & ".join(
                    f"{sel[(sel.k_sd == k) & (sel.condition == cc)].loo_control_fp.iloc[0]:.2f} / "
                    f"{sel[(sel.k_sd == k) & (sel.condition == cc)].top_dose_rate.iloc[0]:.2f}"
                    for cc in CKEYS) + r" \\"
                 for k in sorted(sel.k_sd.unique())],
                "lcccc",
                r"$k$ (robust SD) & " + " & ".join(CTEX[k] for k in CKEYS))
    return r


# =============================================================================
#  EXPERIMENT 4.  TASK A -- exposure detection and dose reconstruction.
#
#  Outcome: a design variable (was this fish exposed?).  Predictors: the eleven
#  biomarkers.  Outcome and predictors share no information by construction,
#  so the circularity of the submitted formulation cannot recur.
# =============================================================================
def model_factories(seed=SEED):
    base = {k: (lambda v=v: clone(v)) for k, v in baseline_models(seed).items()}
    base["DCRP (proposed)"] = lambda: DCRP(C=1.0, penalty="l2")
    base["DCRP, no severity"] = lambda: DCRP(C=1.0, penalty="l2", add_severity=False)
    return base


MODEL_ORDER = ["Majority class", "Logistic (L2)", "Logistic (L1, CV)",
               "Random forest", "Grad. boosting", "MLP (32--32)",
               "DCRP, no severity", "DCRP (proposed)"]


def exp4_task_a(df, n_perm=300, n_boot=1000):
    log_step("E4  Task A: exposure detection and dose reconstruction")
    res, inf_rows, reg_rows, oof_store = [], [], [], {}
    for c in CONDITIONS:
        d = subset(df, c)
        X, y = d[BIOMARKERS].to_numpy(), exposed_label(d)
        conc, groups = d.concentration.to_numpy(), d.dose_group.to_numpy()
        step(f"{CTEX_PLAIN[c.key]}  (n={len(y)}, exposed={int(y.sum())}, "
             f"control={int(len(y)-y.sum())})")
        protocols = {"Random 5-fold": random_folds(y),
                     "LODGO": lodgo(conc)}
        for pname, folds in protocols.items():
            log(f"protocol: {pname}  ({len(folds)} folds)", 2)
            for name in MODEL_ORDER:
                log(f"{pname}: {name}", 2)
                oof = fit_predict_oof(model_factories()[name], X, y, folds, conc)
                ok = ~np.isnan(oof)
                (alo, ahi), (blo, bhi) = cluster_bootstrap(y[ok], oof[ok], groups[ok],
                                                           n_boot=n_boot)
                m_ = metrics(y[ok], oof[ok])
                kv(name, f"AUC={m_['roc_auc']:.3f} [{alo:.2f},{ahi:.2f}]  "
                         f"BAcc={m_['balanced_accuracy']:.3f}  "
                         f"F1={m_['macro_f1']:.3f}", indent=3, width=30)
                res.append(dict(condition=c.key, protocol=pname, model=name,
                                **m_, auc_lo=alo, auc_hi=ahi,
                                bacc_lo=blo, bacc_hi=bhi))
                if pname == "LODGO":
                    oof_store[(c.key, name)] = oof

        folds = protocols["LODGO"]
        log(f"{c.key}: permutation test, {n_perm} relabelled CV runs", 1)
        sf = lambda yy: np.nan_to_num(
            fit_predict_oof(model_factories()["DCRP (proposed)"], X, yy, folds, conc),
            nan=0.5)
        log(f"permutation test ({n_perm} relabellings of the full CV loop)", 2)
        _pt = permutation_test(sf, y, n_perm=n_perm)
        kv("observed vs null", f"{_pt['observed']:.3f} vs "
           f"{_pt['null_mean']:.3f}+/-{_pt['null_sd']:.3f}  "
           f"p={_pt['p_value']:.4f}", indent=3, width=30)
        inf_rows.append(dict(condition=c.key, model="DCRP (proposed)", **_pt))
        yl = log_dose(d)
        for rn, est in {"Ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
                        "Random forest": RandomForestRegressor(n_jobs=1, n_estimators=500, min_samples_leaf=2,
                            random_state=SEED)}.items():
            pred = np.full(len(yl), np.nan)
            for tr, te in folds:
                mdl = Pipeline([("s", StandardScaler()), ("m", clone(est))])
                mdl.fit(X[tr], yl[tr]); pred[te] = mdl.predict(X[te])
            ok = ~np.isnan(pred)
            _rho = float(spearmanr(yl[ok], pred[ok]).statistic)
            _mae = float(np.mean(np.abs(yl[ok] - pred[ok])))
            kv(f"dose reconstruction ({rn})",
               f"rho={_rho:.3f}  MAE={_mae:.3f} log10", indent=3, width=30)
            reg_rows.append(dict(condition=c.key, model=rn,
                                 spearman=_rho, mae_log10=_mae))
    R = pd.DataFrame(res); I = pd.DataFrame(inf_rows); G = pd.DataFrame(reg_rows)
    R.to_csv(OUT / "e4_taskA_results.csv", index=False)
    I.to_csv(OUT / "e4_taskA_permutation.csv", index=False)
    G.to_csv(OUT / "e4_taskA_dose_regression.csv", index=False)

    L = R[R.protocol == "LODGO"]
    best = {cc: L[(L.condition == cc) & (L.model != "Majority class")].roc_auc.max()
            for cc in CKEYS}
    rows = []
    for mname in MODEL_ORDER:
        cells = []
        for cc in CKEYS:
            s = L[(L.condition == cc) & (L.model == mname)]
            if s.empty:
                cells.append("--"); continue
            s = s.iloc[0]
            t = f"{s.roc_auc:.2f} [{s.auc_lo:.2f},{s.auc_hi:.2f}]"
            cells.append(r"\textbf{" + t + "}" if np.isclose(s.roc_auc, best[cc]) else t)
        rows.append(f"{mname} & " + " & ".join(cells) + r" \\")
    write_table("t5_taskA.tex", rows, "lcccc",
                r"Model & " + " & ".join(CTEX[k] for k in CKEYS))

    gr = G[G.model == "Ridge"].set_index("condition")
    write_table("t6_taskA_inference.tex",
                [f"{CTEX[x.condition]} & {x.observed:.3f} & {x.null_mean:.3f} & "
                 f"{x.null_sd:.3f} & {x.p_value:.3f} & "
                 f"{gr.loc[x.condition].spearman:.2f} & "
                 f"{gr.loc[x.condition].mae_log10:.2f} \\\\" for x in I.itertuples()],
                "lccccrr",
                (r"Condition & Observed AUC & Null mean & Null SD & $p$ & "
                 r"Spearman $\rho$ & MAE ($\log_{10}$)"))
    return R, I, G, oof_store


# =============================================================================
#  EXPERIMENT 5.  Held-out-cohort transfer across conditions.
# =============================================================================
def exp5_transfer(df, n_boot=1000):
    log_step("E5  held-out-cohort transfer")
    data = {}
    for c in CONDITIONS:
        d = subset(df, c)
        data[c.key] = (d[BIOMARKERS].to_numpy(), exposed_label(d),
                       d.dose_group.to_numpy(), d.concentration.to_numpy())
    rows = []
    for src in CKEYS:
        Xtr, ytr, _, ctr = data[src]
        for dst in CKEYS:
            if src == dst:
                continue
            Xte, yte, gte, _ = data[dst]
            kind = ("duration" if src[:3] == dst[:3] else
                    "speciation" if src[-3:] == dst[-3:] else "both")
            for name in ["Logistic (L2)", "Random forest", "DCRP (proposed)"]:
                mdl = model_factories()[name]()
                if isinstance(mdl, DCRP):
                    mdl.fit(Xtr, ytr, control_mask=(ctr == 0))
                else:
                    mdl.fit(Xtr, ytr)
                s = mdl.predict_proba(Xte)[:, 1]
                (alo, ahi), _ = cluster_bootstrap(yte, s, gte, n_boot=n_boot, seed=3)
                _auc = roc_auc_score(yte, s)
                if name == "DCRP (proposed)":
                    log(f"{CTEX_PLAIN[src]:>14}  ->  {CTEX_PLAIN[dst]:<14} "
                        f"AUC={_auc:.3f}  ({kind})", 2)
                rows.append(dict(train=src, test=dst, transfer=kind, model=name,
                                 roc_auc=_auc, auc_lo=alo, auc_hi=ahi,
                                 balanced_accuracy=fast_bacc(yte, s)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "e5_transfer.csv", index=False)
    rows = []
    for name in ["Logistic (L2)", "DCRP (proposed)"]:
        piv = T[T.model == name].pivot(index="train", columns="test", values="roc_auc")
        for tr in CKEYS:
            cells = ["--" if tr == te else f"{piv.loc[tr, te]:.2f}" for te in CKEYS]
            rows.append(f"{name} & {CTEX[tr]} & " + " & ".join(cells) + r" \\")
    write_table("t7_transfer.tex", rows, "llcccc",
                r"Model & Trained on & " + " & ".join(CTEX[k] for k in CKEYS))
    return T


# =============================================================================
#  EXPERIMENT 6.  TASK B -- panel-split hepatic outcome (disjoint predictors).
# =============================================================================
def exp6_task_b(df, n_perm=300):
    log_step("E6  Task B: panel-split hepatic outcome")
    res, lab, perm = [], [], []
    for c in CONDITIONS:
        d = subset(df, c); ctrl = d[d.concentration == 0]
        iv = reference_intervals(ctrl, k=2.0, markers=HEPATIC)
        exp = d[d.concentration > 0].reset_index(drop=True)
        y = (deviation_flags(exp, iv).to_numpy().sum(1) >= 1).astype(int)
        groups = exp.dose_group.to_numpy()
        folds = lodgo_exposed(exp.concentration.to_numpy())
        lab.append(dict(condition=c.key, n_exposed=len(exp), n_abnormal=int(y.sum()),
                        prevalence=float(y.mean())))
        if len(np.unique(y)) < 2:
            continue
        sets = {"concentration only": exp[["concentration"]].to_numpy(),
                "biomarkers only": exp[NON_HEPATIC].to_numpy(),
                "both": exp[["concentration"] + NON_HEPATIC].to_numpy()}
        for fname, X in sets.items():
            oof = fit_predict_oof(lambda: clone(baseline_models()["Logistic (L2)"]),
                                  X, y, folds)
            ok = ~np.isnan(oof)
            (alo, ahi), _ = cluster_bootstrap(y[ok], oof[ok], groups[ok])
            _m = metrics(y[ok], oof[ok])
            kv(f"{CTEX_PLAIN[c.key]} / {fname}",
               f"AUC={_m['roc_auc']:.3f} [{alo:.2f},{ahi:.2f}]", indent=2, width=42)
            res.append(dict(condition=c.key, features=fname, **_m,
                            auc_lo=alo, auc_hi=ahi))
        Xb = exp[NON_HEPATIC].to_numpy()
        sf = lambda yy: np.nan_to_num(fit_predict_oof(
            lambda: clone(baseline_models()["Logistic (L2)"]), Xb, yy, folds), nan=0.5)
        perm.append(dict(condition=c.key, **permutation_test(sf, y, n_perm=n_perm, seed=2)))
    R, L, P = pd.DataFrame(res), pd.DataFrame(lab), pd.DataFrame(perm)
    R.to_csv(OUT / "e6_taskB.csv", index=False)
    L.to_csv(OUT / "e6_taskB_labels.csv", index=False)
    P.to_csv(OUT / "e6_taskB_permutation.csv", index=False)
    Li, Pi = L.set_index("condition"), P.set_index("condition")
    rows = []
    for cc in CKEYS:
        cells = []
        for f in ["concentration only", "biomarkers only", "both"]:
            s = R[(R.condition == cc) & (R.features == f)]
            cells.append("--" if s.empty else
                         f"{s.iloc[0].roc_auc:.2f} [{s.iloc[0].auc_lo:.2f},{s.iloc[0].auc_hi:.2f}]")
        pv = Pi.loc[cc].p_value if cc in Pi.index else np.nan
        rows.append(f"{CTEX[cc]} & {int(Li.loc[cc].n_exposed)} & "
                    f"{int(Li.loc[cc].n_abnormal)} ({Li.loc[cc].prevalence:.2f}) & "
                    + " & ".join(cells) + f" & {fmt(pv, 3)} \\\\")
    write_table("t8_taskB.tex", rows, "lcccccc",
                (r"Condition & $n$ exp. & abnormal (prev.) & Conc.\ only & "
                 r"Biomarkers only & Both & $p$"))
    return R, L, P


# =============================================================================
#  EXPERIMENT 7.  Dose-response: benchmark doses, and BMD vs LOEC agreement.
# =============================================================================
def m_hill(x, a, b, c, d):
    return a + b * x ** d / (c ** d + x ** d + 1e-12)


def m_exp(x, a, b, c):
    return a * np.exp(b * (1 - np.exp(-c * x)))


def m_power(x, a, b, c):
    return a + b * np.sign(x) * np.abs(x) ** c


def m_loglin(x, a, b):
    return a + b * np.log1p(x)


CURVES = {"Hill": (m_hill, 4), "Exponential": (m_exp, 3),
          "Power": (m_power, 3), "Log-linear": (m_loglin, 2)}


def model_average(x, y):
    """AIC-weighted average over four dose-response families."""
    n = len(y); fits = {}
    for nm, (fn, k) in CURVES.items():
        p0 = [float(np.mean(y[x == x.min()]))] + [0.5] * (k - 1)
        if k >= 2:
            p0[1] = float(np.mean(y[x == x.max()]) - p0[0]) or 0.1
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
        except Exception:
            continue
        rss = float(np.sum((y - fn(x, *popt)) ** 2))
        if rss <= 0:
            continue
        fits[nm] = dict(popt=popt, fn=fn, aic=n * np.log(rss / n) + 2 * k)
    if not fits:
        return None
    amin = min(f["aic"] for f in fits.values())
    tot = sum(np.exp(-0.5 * (f["aic"] - amin)) for f in fits.values())
    for f in fits.values():
        f["w"] = float(np.exp(-0.5 * (f["aic"] - amin)) / tot)
    return fits


def avg_curve(fits, grid):
    return sum(f["w"] * f["fn"](grid, *f["popt"]) for f in fits.values())


def bmd_continuous(x, y, control_sd, bmr_sd=1.0, gmax=None, n=4001):
    fits = model_average(x, y)
    if fits is None or control_sd <= 0:
        return np.nan, None
    grid = np.linspace(0, gmax or float(x.max()), n)
    cur = avg_curve(fits, grid)
    hit = np.where(np.abs(cur - cur[0]) >= bmr_sd * control_sd)[0]
    return (float(grid[hit[0]]) if len(hit) else np.nan), fits


def exp7_dose_response(df, n_boot=300):
    log_step("E7  dose-response and benchmark doses")
    bmd_rows, grp_rows = [], []
    for c in CONDITIONS:
        d = subset(df, c).copy(); ctrl = d[d.concentration == 0]
        iv = reference_intervals(ctrl)
        d["IBR"] = ibr_v2(d, ctrl)
        d["abnormal"] = abnormal_label(d, iv)
        x = d.concentration.to_numpy()
        g = d.groupby("concentration").agg(
            n=("IBR", "size"), IBR_mean=("IBR", "mean"),
            IBR_se=("IBR", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
            abnormal_rate=("abnormal", "mean")).reset_index()
        g.insert(0, "condition", c.key); grp_rows.append(g)

        log(f"{CTEX_PLAIN[c.key]}: fitting 11 biomarkers", 1)
        groups = d.dose_group.to_numpy()
        member = [np.where(groups == q)[0] for q in np.unique(groups)]
        rng = np.random.default_rng(7)
        log(f"{c.key}: BMD for 11 biomarkers, {n_boot} bootstrap each", 1)
        for mk in BIOMARKERS:
            log(f"BMD {mk}", 2)
            yv = d[mk].to_numpy(); csd = float(ctrl[mk].std(ddof=1))
            pt, fits = bmd_continuous(x, yv, csd)
            boots = []
            for _ in range(n_boot):
                idx = np.concatenate([mem[rng.integers(0, len(mem), len(mem))]
                                      for mem in member])
                v, _ = bmd_continuous(x[idx], yv[idx], csd, gmax=float(x.max()))
                if np.isfinite(v):
                    boots.append(v)
            lo, hi = ((np.nan, np.nan) if len(boots) < 20 else
                      (float(np.quantile(boots, .05)), float(np.quantile(boots, .95))))
            loec = LOEC[(c.species, c.day)][mk]
            kv(SYMBOL[mk], f"BMD={fmt(pt)} [{fmt(lo)},{fmt(hi)}]  "
               f"LOEC={'n.s.' if not np.isfinite(loec) else f'{loec:g}'}",
               indent=2, width=8)
            bmd_rows.append(dict(condition=c.key, biomarker=mk, BMD_1SD=pt,
                                 BMDL=lo, BMDU=hi, LOEC=loec,
                                 loec_finite=np.isfinite(loec),
                                 bmd_below_loec=(np.isfinite(loec) and np.isfinite(pt)
                                                 and pt <= loec)))
    B = pd.DataFrame(bmd_rows); G = pd.concat(grp_rows, ignore_index=True)
    B.to_csv(OUT / "e7_bmd.csv", index=False); G.to_csv(OUT / "e7_groups.csv", index=False)

    sub = B[B.loec_finite & np.isfinite(B.BMD_1SD)]
    agree = dict(n_pairs=len(sub),
                 spearman=float(spearmanr(sub.LOEC, sub.BMD_1SD).statistic),
                 p_value=float(spearmanr(sub.LOEC, sub.BMD_1SD).pvalue),
                 frac_bmd_below_loec=float(sub.bmd_below_loec.mean()),
                 median_ratio=float(np.median(sub.BMD_1SD / sub.LOEC)))
    step("BMD vs LOEC agreement")
    kv("pairs with a finite LOEC", agree["n_pairs"])
    kv("Spearman rho", f"{agree['spearman']:.3f}  (p={agree['p_value']:.2e})")
    kv("fraction with BMD <= LOEC", f"{agree['frac_bmd_below_loec']:.2f}")
    kv("median BMD/LOEC ratio", f"{agree['median_ratio']:.3f}")
    pd.DataFrame([agree]).to_csv(OUT / "e7_bmd_loec_agreement.csv", index=False)

    rows = []
    for mk in BIOMARKERS:
        cells = []
        for cc in CKEYS:
            s = B[(B.condition == cc) & (B.biomarker == mk)]
            cells.append("--" if s.empty or not np.isfinite(s.iloc[0].BMD_1SD)
                         else f"{s.iloc[0].BMD_1SD:.2f} [{fmt(s.iloc[0].BMDL)},{fmt(s.iloc[0].BMDU)}]")
        rows.append(f"{SYMBOL[mk]} & " + " & ".join(cells) + r" \\")
    write_table("t9_bmd.tex", rows, "lcccc",
                r"Biomarker & " + " & ".join(CTEX[k] for k in CKEYS))
    return B, G, agree


# =============================================================================
#  EXPERIMENT 8.  Survival: an independently measured adverse outcome.
#
#  Seven fish were allocated per group; groups with fewer rows lost animals,
#  and the manuscript states that only survivors entered the analysis.
#  THIS INFERENCE REQUIRES CONFIRMATION BY THE EXPERIMENTALISTS.
# =============================================================================
def log_logistic(x, lc50, slope):
    return 1.0 / (1.0 + np.exp(-slope * (np.log10(np.maximum(x, 1e-3))
                                         - np.log10(max(lc50, 1e-3)))))


def exp8_survival(df, n_boot=400):
    log_step("E8  survival and LC50")
    rows, summ = [], []
    for c in CONDITIONS:
        obs = subset(df, c).groupby("concentration").size().to_dict()
        conc = np.array(LADDER[c.species], float)
        surv = np.array([obs.get(float(v), 0) for v in conc])
        dead = N_DESIGN - surv
        for cc, s_, d_ in zip(conc, surv, dead):
            rows.append(dict(condition=c.key, concentration=float(cc),
                             n_design=N_DESIGN, n_survivors=int(s_),
                             n_dead=int(d_), mortality=float(d_) / N_DESIGN))
        use = conc > 0

        def fit(dv):
            try:
                popt, _ = curve_fit(log_logistic, conc[use], (dv / N_DESIGN)[use],
                                    p0=[2.0, 2.0], bounds=([1e-3, .1], [50, 50]),
                                    maxfev=20000)
                return float(popt[0])
            except Exception:
                return np.nan

        pt = fit(dead) if dead[use].sum() > 0 else np.nan
        rng = np.random.default_rng(0)
        bs = [fit(rng.binomial(N_DESIGN, np.clip(dead / N_DESIGN, 0, 1)))
              for _ in range(n_boot)] if np.isfinite(pt) else []
        bs = np.array([b for b in bs if np.isfinite(b)])
        lo, hi = ((np.nan, np.nan) if len(bs) < 50 else
                  (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))))
        surv_txt = ", ".join(f"{cc:g}mg/L:{int(sv)}/7"
                             for cc, sv in zip(conc, surv) if sv < N_DESIGN)
        log(f"{CTEX_PLAIN[c.key]}: {int(dead.sum())} deaths in "
            f"{int((dead > 0).sum())} groups" +
            (f"  [{surv_txt}]" if surv_txt else "  [no mortality]"), 1)
        kv("LC50", "n/a" if not np.isfinite(pt)
           else f"{pt:.2f} mg/L  [{fmt(lo)}, {fmt(hi)}]", indent=2, width=8)
        summ.append(dict(condition=c.key, total_dead=int(dead.sum()),
                         groups_with_deaths=int((dead > 0).sum()),
                         LC50=pt, LC50_lo=lo, LC50_hi=hi))
    R, S = pd.DataFrame(rows), pd.DataFrame(summ)
    R.to_csv(OUT / "e8_survival.csv", index=False)
    S.to_csv(OUT / "e8_lc50.csv", index=False)
    piv = R.pivot(index="concentration", columns="condition", values="n_survivors")
    Si = S.set_index("condition")
    body = []
    for cc in sorted(piv.index):
        cells = [("n/a" if not np.isfinite(piv.loc[cc, k]) else f"{int(piv.loc[cc,k])}/7")
                 for k in CKEYS]
        body.append(f"{cc:g} & " + " & ".join(cells) + r" \\")
    body.append(r"\textbf{LC$_{50}$ (mg/L)} & " + " & ".join(
        ("--" if not np.isfinite(Si.loc[k].LC50)
         else f"\\textbf{{{Si.loc[k].LC50:.1f}}}") for k in CKEYS) + r" \\")
    write_table("t10_survival.tex", body, "lcccc",
                r"Conc.\ (mg/L) & " + " & ".join(CTEX[k] for k in CKEYS))
    return R, S


# =============================================================================
#  EXPERIMENT 10.  Biomarker attribution on the non-circular task.
# =============================================================================
def exp10_attribution(df, n_repeat=50):
    log_step("E10 permutation importance and dependency graphs")
    rng = np.random.default_rng(0); rows = []
    for c in CONDITIONS:
        d = subset(df, c)
        X, y = d[BIOMARKERS].to_numpy(), exposed_label(d)
        conc = d.concentration.to_numpy(); folds = lodgo(conc)
        fitted = []
        for tr, te in folds:
            mdl = DCRP().fit(X[tr], y[tr], control_mask=(conc[tr] == 0))
            fitted.append((mdl, te))
        oof = np.full(len(y), np.nan)
        for mdl, te in fitted:
            oof[te] = mdl.predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof); base = fast_auc(y[ok], oof[ok])
        for k, mk in enumerate(BIOMARKERS):
            log(f"permuting {mk} ({n_repeat} repeats)", 2)
            drops = []
            for _ in range(n_repeat):
                pr = np.full(len(y), np.nan)
                for mdl, te in fitted:
                    Xt = X[te].copy()
                    Xt[:, k] = Xt[rng.permutation(len(te)), k]
                    pr[te] = mdl.predict_proba(Xt)[:, 1]
                drops.append(base - fast_auc(y[ok], pr[ok]))
            drops = np.asarray(drops)
            rows.append(dict(condition=c.key, biomarker=mk, category=CATEGORY[mk],
                             base_auc=base, importance=float(drops.mean()),
                             lo=float(np.quantile(drops, .025)),
                             hi=float(np.quantile(drops, .975))))
        top = sorted([r for r in rows if r["condition"] == c.key],
                     key=lambda z: -z["importance"])[:3]
        log(f"{CTEX_PLAIN[c.key]} (base AUC={base:.3f}): top = "
            + ", ".join(f"{SYMBOL[t['biomarker']]} {t['importance']:+.3f}"
                        for t in top), 1)
    I = pd.DataFrame(rows)
    I.to_csv(OUT / "e10_importance.csv", index=False)
    rk = (I.assign(r=I.groupby("condition").importance.rank(ascending=False))
            .pivot(index="biomarker", columns="condition", values="r"))
    rk["mean_rank"] = rk.mean(1); rk = rk.sort_values("mean_rank")
    rk.to_csv(OUT / "e10_ranks.csv")
    write_table("t13_attribution.tex",
                [f"{SYMBOL[mk]} & " + " & ".join(
                    f"{I[(I.condition==k)&(I.biomarker==mk)].importance.iloc[0]:+.3f} "
                    f"({int(rk.loc[mk,k])})" for k in CKEYS)
                 + f" & {rk.loc[mk,'mean_rank']:.2f} \\\\" for mk in rk.index],
                "lccccc",
                r"Biomarker & " + " & ".join(CTEX[k] for k in CKEYS) + r" & Mean rank")
    return I, rk


# =============================================================================
#  SECTION 8.  Figures (Computer Modern, bold text throughout).
# =============================================================================


def fig2_bmd_vs_loec(B):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for c in CONDITIONS:
        s = B[(B.condition == c.key) & B.loec_finite & np.isfinite(B.BMD_1SD)]
        ax.scatter(s.LOEC, s.BMD_1SD, s=46, color=PALETTE[c.key], alpha=.85,
                   edgecolors="k", linewidths=.5, label=c.tex)
    lim = [0.05, 12]
    ax.plot(lim, lim, "k--", lw=1.1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("LOEC (mg/L), ANOVA", fontweight="bold", fontsize=10)
    ax.set_ylabel("BMD$_{1\\mathrm{SD}}$ (mg/L), model-averaged",
                  fontweight="bold", fontsize=10)
    ax.set_title("Benchmark dose against the reported LOEC",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.6); bolden(ax)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_bmd_loec.pdf"); plt.close(fig)


def fig3_dose_response(G):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.3))
    for ax, c in zip(axes, CONDITIONS):
        g = G[G.condition == c.key]
        ax.errorbar(g.concentration, g.IBR_mean, yerr=g.IBR_se, marker="o", ms=5,
                    lw=1.6, capsize=3, color=PALETTE[c.key], label="IBRv2")
        a2 = ax.twinx()
        a2.plot(g.concentration, g.abnormal_rate, ls=":", marker="s", ms=4,
                lw=1.4, color="0.25", label="abnormal rate")
        a2.set_ylim(-.05, 1.05)
        for t in a2.get_yticklabels():
            t.set_fontweight("bold"); t.set_fontsize(8)
        ax.set_title(c.tex, fontweight="bold", fontsize=11)
        ax.set_xlabel("mg/L", fontweight="bold", fontsize=9)
        bolden(ax, 8)
    axes[0].set_ylabel("IBRv2 (control SD)", fontweight="bold", fontsize=9)
    fig.suptitle("Control-referenced severity and abnormality rate against dose",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_dose_response.pdf"); plt.close(fig)


def fig4_survival(R, S):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    Si = S.set_index("condition")
    for c in CONDITIONS:
        r = R[R.condition == c.key].sort_values("concentration")
        ax.plot(r.concentration, r.mortality, marker="o", ms=6, lw=1.8,
                color=PALETTE[c.key],
                label=f"{c.tex}  LC$_{{50}}$=" +
                      ("n/a" if not np.isfinite(Si.loc[c.key].LC50)
                       else f"{Si.loc[c.key].LC50:.1f}"))
    ax.set_xlabel("arsenic concentration (mg/L)", fontweight="bold", fontsize=10)
    ax.set_ylabel("mortality (dead / 7)", fontweight="bold", fontsize=10)
    ax.set_ylim(-.05, 1.08)
    ax.set_title("Survival: an independently measured adverse outcome",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.6); bolden(ax)
    fig.tight_layout(); fig.savefig(FIGS / "fig4_survival.pdf"); plt.close(fig)


def fig5_task_a(R):
    L = R[R.protocol == "LODGO"]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.9), sharex=True)
    for ax, c in zip(axes, CONDITIONS):
        s = L[(L.condition == c.key) & (L.model != "Majority class")]
        s = s.set_index("model").loc[[m for m in MODEL_ORDER if m != "Majority class"]]
        cols = ["#b03a2e" if m == "DCRP (proposed)" else PALETTE[c.key]
                for m in s.index]
        ax.barh(range(len(s)), s.roc_auc,
                xerr=[s.roc_auc - s.auc_lo, s.auc_hi - s.roc_auc],
                color=cols, alpha=.9, error_kw=dict(lw=.8, capsize=2))
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontweight="bold", fontsize=8)
        ax.axvline(.5, color="k", ls="--", lw=.9)
        ax.set_xlim(.3, 1.03); ax.set_title(c.tex, fontweight="bold", fontsize=11)
        bolden(ax, 8)
    axes[0].set_xlabel("out-of-fold ROC-AUC (95\\% cluster bootstrap)"
                       .replace("\\%", "%"), fontweight="bold", fontsize=9)
    fig.suptitle("Task A: exposure detection under dose-blocked cross-validation",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig5_task_a.pdf"); plt.close(fig)


def fig6_transfer(T):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, name in zip(axes, ["Logistic (L2)", "DCRP (proposed)"]):
        piv = T[T.model == name].pivot(index="train", columns="test",
                                       values="roc_auc").reindex(index=CKEYS,
                                                                 columns=CKEYS)
        M = piv.to_numpy(float)
        im = ax.imshow(M, cmap="viridis", vmin=.6, vmax=1.0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "--" if not np.isfinite(M[i, j]) else f"{M[i,j]:.2f}",
                        ha="center", va="center", fontweight="bold", fontsize=9,
                        color="w" if np.isfinite(M[i, j]) and M[i, j] < .9 else "k")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([c.tex for c in CONDITIONS], fontweight="bold",
                           fontsize=8, rotation=20)
        ax.set_yticklabels([c.tex for c in CONDITIONS], fontweight="bold", fontsize=8)
        ax.set_xlabel("tested on", fontweight="bold", fontsize=10)
        ax.set_ylabel("trained on", fontweight="bold", fontsize=10)
        ax.set_title(name, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=.03)
    cb.set_label("ROC-AUC", fontweight="bold")
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")
    fig.suptitle("Held-out-cohort transfer across conditions",
                 fontweight="bold", fontsize=12)
    fig.savefig(FIGS / "fig6_transfer.pdf"); plt.close(fig)


def fig7_importance(I):
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8))
    for ax, c in zip(axes, CONDITIONS):
        s = I[I.condition == c.key].sort_values("importance")
        ax.barh([SYMBOL[m] for m in s.biomarker], s.importance,
                xerr=[s.importance - s.lo, s.hi - s.importance],
                color=PALETTE[c.key], alpha=.9, error_kw=dict(lw=.7, capsize=2))
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{c.tex}  (AUC={s.base_auc.iloc[0]:.2f})",
                     fontweight="bold", fontsize=10)
        bolden(ax, 8)
    axes[0].set_xlabel("drop in out-of-fold AUC when permuted",
                       fontweight="bold", fontsize=9)
    fig.suptitle("Permutation importance on the non-circular exposure task",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "fig7_importance.pdf"); plt.close(fig)






# =============================================================================
#  DRIVER
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uploads", default=None,
                    help="folder holding the two .xlsx workbooks; auto-detected if omitted")
    ap.add_argument("--quick", action="store_true",
                    help="fewer bootstrap/permutation replicates")
    ap.add_argument("--only", default=None,
                    help="comma-separated experiment numbers to run, e.g. 8,9,10")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-step progress output")
    a = ap.parse_args()
    nb, npm = (200, 60) if a.quick else (1000, 300)

    log_rule("Arsenic biomonitoring pipeline -- single-file reproduction")
    log(f"python {sys.version.split()[0]} on {sys.platform}", 1)
    try:
        import sklearn, matplotlib
        log(f"numpy {np.__version__}, pandas {pd.__version__}, "
            f"sklearn {sklearn.__version__}, matplotlib {matplotlib.__version__}", 1)
    except Exception:
        pass
    log(f"bootstrap {nb} replicates, permutation {npm} replicates", 1)
    log(f"outputs: {OUT}  |  {FIGS}  |  {TABS}", 1)
    if _QUIET:
        log("ARSENIC_QUIET=1: detail lines suppressed", 1)

    df = load_all(a.uploads)
    log(f"loaded {len(df)} fish in {df.dose_group.nunique()} dose groups", 1)
    for c in CONDITIONS:
        sub = subset(df, c)
        log(f"{CTEX_PLAIN[c.key]}: n={len(sub)}, doses "
            f"{[float(v) for v in sorted(sub.concentration.unique())]} mg/L", 2)

    want = (set(int(x) for x in a.only.split(",")) if a.only
            else {1, 3, 4, 5, 6, 7, 8, 10})
    _STEP["total"] = len(want) + 1          # experiments plus the figure stage
    log(f"running experiments: {sorted(want) if want else 'none (figures only)'}", 1)

    def cached(nums, path, builder):
        """Run `builder` when requested, otherwise reload from out/."""
        if nums & want:
            return builder()
        return pd.read_csv(OUT / path)

    loec_df = cached({1}, "e1_loec_table.csv", lambda: exp1_loec_table(df))
    if 3 in want:
        exp3_calibration(df)
    if 4 in want:
        R, I4, G4, _ = exp4_task_a(df, n_perm=npm, n_boot=nb)
    else:
        R = pd.read_csv(OUT / "e4_taskA_results.csv")
        I4 = pd.read_csv(OUT / "e4_taskA_permutation.csv")
    T = cached({5}, "e5_transfer.csv", lambda: exp5_transfer(df, n_boot=nb))
    if 6 in want:
        exp6_task_b(df, n_perm=npm)
    if 7 in want:
        B, G7, agree = exp7_dose_response(df, n_boot=300 if not a.quick else 80)
    else:
        B = pd.read_csv(OUT / "e7_bmd.csv"); G7 = pd.read_csv(OUT / "e7_groups.csv")
        agree = pd.read_csv(OUT / "e7_bmd_loec_agreement.csv").iloc[0].to_dict()
    if 8 in want:
        SR, SL = exp8_survival(df)
    else:
        SR = pd.read_csv(OUT / "e8_survival.csv"); SL = pd.read_csv(OUT / "e8_lc50.csv")
    if 10 in want:
        IMP, RK = exp10_attribution(df)
    else:
        IMP = pd.read_csv(OUT / "e10_importance.csv")

    log_step("Figures")
    fig2_bmd_vs_loec(B)
    fig3_dose_response(G7)
    fig4_survival(SR, SL)
    fig5_task_a(R)
    fig6_transfer(T)
    fig7_importance(IMP)
    for p in sorted(FIGS.glob("*.pdf")):
        log(f"figure {p.name}  ({p.stat().st_size // 1024} kB)", 2)

    log("")
    log_rule("SUMMARY")
    L = R[R.protocol == 'LODGO']
    d = L[L.model == 'DCRP (proposed)'].set_index('condition').roc_auc
    log(f"Task A, DCRP, LODGO AUC                           : "
          f"{', '.join(f'{d[k]:.2f}' for k in CKEYS)}")
    log(f"Task A permutation p-values                       : "
          f"{', '.join(f'{v:.3f}' for v in I4.p_value)}")
    log(f"held-out-cohort transfer AUC (DCRP), min-max      : "
          f"{T[T.model=='DCRP (proposed)'].roc_auc.min():.2f}"
          f"-{T[T.model=='DCRP (proposed)'].roc_auc.max():.2f}")
    log(f"BMD vs LOEC Spearman                              : "
          f"{agree['spearman']:.2f} (p={agree['p_value']:.1e})")
    log(f"LC50 (mg/L)                                       : "
          f"{', '.join(fmt(v,1) for v in SL.LC50)}")
    log(f"wrote {len(list(OUT.glob('*.csv')))} CSVs, "
        f"{len(list(FIGS.glob('*.pdf')))} figures, "
        f"{len(list(TABS.glob('*.tex')))} LaTeX tables", 0)
    log(f"total elapsed {_time.time() - _T0:.1f} s", 0)
    log_rule()


if __name__ == "__main__":
    main()


# =============================================================================
#  EXPERIMENT 11.  SPECIATION DISCRIMINATION:  As(III) versus As(V).
#
#  Given the blood panel of an exposed fish, which arsenic species was it
#  exposed to?  The label is a design variable (which tank series the animal
#  belonged to), so no predictor encodes it.
#
#  Two confounds are controlled explicitly:
#    (i)  DOSE.  The two arms do not span identical ladders, so the analysis is
#         restricted to concentrations present in BOTH arms at that day, and
#         folds hold out one nominal dose level from both species at once.  A
#         dose-only baseline is reported; it should be uninformative.
#    (ii) REFERENCE.  Standardising each species against its own controls would
#         erase the very difference being tested, so DCRP here references a
#         POOLED control set drawn from both arms at that day.
# =============================================================================
def matched_doses(df, day):
    a3 = set(subset(df, Condition("As3", day)).concentration.unique())
    a5 = set(subset(df, Condition("As5", day)).concentration.unique())
    return sorted(c for c in (a3 & a5) if c > 0)


def speciation_frame(df, day):
    """Exposed fish of both species at matched concentrations, plus the pooled
    control set used only to fit the reference transform."""
    keep = matched_doses(df, day)
    d = df[(df.day == day)].copy()
    exposed = d[d.concentration.isin(keep)].reset_index(drop=True)
    controls = d[d.concentration == 0].reset_index(drop=True)
    return exposed, controls, keep


def exp11_speciation(df, n_perm=300, n_boot=1000):
    log_step("E11 speciation discrimination As(III) vs As(V)")
    rows, inf_rows, imp_rows = [], [], []

    for day in (10, 20, "pooled"):
        days = [10, 20] if day == "pooled" else [day]
        frames, ctrls, ladders = [], [], {}
        for dd in days:
            e, c, keep = speciation_frame(df, dd)
            frames.append(e); ctrls.append(c); ladders[dd] = keep
        exposed = pd.concat(frames, ignore_index=True)
        controls = pd.concat(ctrls, ignore_index=True)

        X = exposed[BIOMARKERS].to_numpy()
        y = (exposed.species == "As3").astype(int).to_numpy()
        conc = exposed.concentration.to_numpy()
        groups = exposed.dose_group.to_numpy()
        # Fold = one nominal dose level, held out from BOTH species at once.
        blocks = (exposed.day.astype(str) + "_c" + conc.astype(str)).to_numpy()
        folds = [(np.where(blocks != b)[0], np.where(blocks == b)[0])
                 for b in np.unique(blocks)]

        Xc = controls[BIOMARKERS].to_numpy()

        def dcrp_pooled():
            """DCRP whose reference is the pooled control set (species-agnostic)."""
            class _M(DCRP):
                def fit(self, Xf, yf, control_mask=None):
                    self.tf_ = DirectionalReference(add_severity=self.add_severity)
                    self.tf_.fit(Xc, control_mask=np.ones(len(Xc), bool))
                    Z = self.tf_.transform(Xf)
                    self.clf_ = LogisticRegression(
                        C=self.C, penalty="l2", solver="lbfgs", max_iter=5000,
                        class_weight="balanced",
                        random_state=self.seed).fit(Z, yf)
                    return self
            return _M()

        zoo = {"Dose only (control)": lambda: clone(baseline_models()["Logistic (L2)"]),
               "Logistic (L2)": lambda: clone(baseline_models()["Logistic (L2)"]),
               "Random forest": lambda: clone(baseline_models()["Random forest"]),
               "Grad. boosting": lambda: clone(baseline_models()["Grad. boosting"]),
               "DCRP (proposed)": dcrp_pooled}

        for name, fac in zoo.items():
            Xu = conc.reshape(-1, 1) if name.startswith("Dose only") else X
            oof = fit_predict_oof(fac, Xu, y, folds)
            ok = ~np.isnan(oof)
            (alo, ahi), (blo, bhi) = cluster_bootstrap(y[ok], oof[ok], groups[ok],
                                                       n_boot=n_boot)
            rows.append(dict(day=str(day), model=name, **metrics(y[ok], oof[ok]),
                             auc_lo=alo, auc_hi=ahi, bacc_lo=blo, bacc_hi=bhi,
                             n_doses=len(np.unique(conc))))

        sf = lambda yy: np.nan_to_num(fit_predict_oof(dcrp_pooled, X, yy, folds),
                                      nan=0.5)
        inf_rows.append(dict(day=str(day), **permutation_test(sf, y, n_perm=n_perm,
                                                              seed=5)))

        # Which biomarkers separate the two species?
        rng = np.random.default_rng(0)
        fitted = []
        for tr, te in folds:
            fitted.append((dcrp_pooled().fit(X[tr], y[tr]), te))  # noqa: E501
        oof = np.full(len(y), np.nan)
        for mdl, te in fitted:
            oof[te] = mdl.predict_proba(X[te])[:, 1]
        ok = ~np.isnan(oof); base = fast_auc(y[ok], oof[ok])
        for k, mk in enumerate(BIOMARKERS):
            drops = []
            for _ in range(30):
                pr = np.full(len(y), np.nan)
                for mdl, te in fitted:
                    Xt = X[te].copy()
                    Xt[:, k] = Xt[rng.permutation(len(te)), k]
                    pr[te] = mdl.predict_proba(Xt)[:, 1]
                drops.append(base - fast_auc(y[ok], pr[ok]))
            imp_rows.append(dict(day=str(day), biomarker=mk, base_auc=base,
                                 importance=float(np.mean(drops)),
                                 lo=float(np.quantile(drops, .025)),
                                 hi=float(np.quantile(drops, .975))))

    R, I, M = pd.DataFrame(rows), pd.DataFrame(inf_rows), pd.DataFrame(imp_rows)
    R.to_csv(OUT / "e11_speciation.csv", index=False)
    I.to_csv(OUT / "e11_speciation_permutation.csv", index=False)
    M.to_csv(OUT / "e11_speciation_importance.csv", index=False)

    DAYS = ["10", "20", "pooled"]
    hdr = {"10": "Day-10", "20": "Day-20", "pooled": "Both days"}
    order = ["Dose only (control)", "Logistic (L2)", "Random forest",
             "Grad. boosting", "DCRP (proposed)"]
    best = {d: R[(R.day == d) & (R.model != "Dose only (control)")].roc_auc.max()
            for d in DAYS}
    body = []
    for name in order:
        cells = []
        for d in DAYS:
            s = R[(R.day == d) & (R.model == name)]
            if s.empty:
                cells.append("--"); continue
            s = s.iloc[0]
            t = f"{s.roc_auc:.2f} [{s.auc_lo:.2f},{s.auc_hi:.2f}]"
            cells.append(r"\textbf{" + t + "}" if np.isclose(s.roc_auc, best[d]) else t)
        body.append(f"{name} & " + " & ".join(cells) + r" \\")
    write_table("t14_speciation.tex", body, "lccc",
                r"Model & " + " & ".join(hdr[d] for d in DAYS))

    write_table("t15_speciation_inference.tex",
                [f"{hdr[x.day]} & {x.observed:.3f} & {x.null_mean:.3f} & "
                 f"{x.null_sd:.3f} & {x.p_value:.3f} \\\\" for x in I.itertuples()],
                "lcccc",
                r"Subset & Observed AUC & Null mean & Null SD & $p$")

    mp = M[M.day == "pooled"].sort_values("importance", ascending=False)
    write_table("t16_speciation_importance.tex",
                [f"{SYMBOL[r.biomarker]} & {CATEGORY[r.biomarker]} & "
                 f"{r.importance:+.3f} & [{r.lo:+.3f}, {r.hi:+.3f}] \\\\"
                 for r in mp.itertuples()],
                "llcc",
                (r"Biomarker & Category & $\Delta$AUC when permuted & 95\% interval"))
    return R, I, M


def fig10_speciation(R, M):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    DAYS = ["10", "20", "pooled"]
    labs = ["Day-10", "Day-20", "Both days"]
    order = ["Dose only (control)", "Logistic (L2)", "Random forest",
             "Grad. boosting", "DCRP (proposed)"]
    w = 0.15
    ax = axes[0]
    for i, name in enumerate(order):
        vals = [R[(R.day == d) & (R.model == name)].roc_auc.iloc[0] for d in DAYS]
        los = [R[(R.day == d) & (R.model == name)].auc_lo.iloc[0] for d in DAYS]
        his = [R[(R.day == d) & (R.model == name)].auc_hi.iloc[0] for d in DAYS]
        col = "#b03a2e" if name == "DCRP (proposed)" else \
              "#8d99ae" if name.startswith("Dose") else "#15476b"
        ax.bar(np.arange(3) + (i - 2) * w, vals, w, color=col, alpha=.9,
               yerr=[np.array(vals) - los, np.array(his) - vals],
               error_kw=dict(lw=.7, capsize=2), label=name)
    ax.axhline(.5, ls="--", color="k", lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontweight="bold", fontsize=9)
    ax.set_ylim(.3, 1.05)
    ax.set_ylabel("ROC-AUC, As(III) vs As(V)", fontweight="bold", fontsize=10)
    ax.set_title("(a) speciation discrimination", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, loc="lower right"); bolden(ax, 9)

    ax = axes[1]
    mp = M[M.day == "pooled"].sort_values("importance")
    ax.barh([SYMBOL[b] for b in mp.biomarker], mp.importance,
            xerr=[mp.importance - mp.lo, mp.hi - mp.importance],
            color="#15476b", alpha=.9, error_kw=dict(lw=.7, capsize=2))
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("drop in AUC when permuted", fontweight="bold", fontsize=9)
    ax.set_title("(b) which biomarkers separate the species",
                 fontweight="bold", fontsize=11)
    bolden(ax, 9)
    fig.tight_layout(); fig.savefig(FIGS / "fig10_speciation.pdf"); plt.close(fig)


# =============================================================================
#  EXPERIMENT 12.  FEASIBILITY OF A TOXICITY-PREDICTION ENDPOINT.
#
#  Part (a) asks whether the present design can support an adverse-outcome
#  model at all.  Mortality is the only measured harm available, so the
#  question is how many dose groups experienced deaths AND retain surviving
#  animals from which a blood panel was drawn.
#
#  Part (b) is a power analysis for the confirmatory study.  It is calibrated
#  on the REAL within-dose-group biomarker covariance measured here, so the
#  sample sizes it returns are specific to this panel and this species rather
#  than generic rules of thumb.
# =============================================================================
def exp12a_mortality_feasibility(df):
    print("[E12a] feasibility of mortality as a prediction target")
    s = pd.read_csv(OUT / "e8_survival.csv")
    d = s[s.n_dead > 0]
    usable = d[d.n_survivors > 0]
    out = dict(n_groups_total=int(len(s)),
               n_groups_with_deaths=int(len(d)),
               n_groups_with_deaths_and_survivors=int(len(usable)),
               n_groups_wiped_out=int((d.n_survivors == 0).sum()),
               surviving_fish_in_positive_groups=int(usable.n_survivors.sum()))
    pd.DataFrame([out]).to_csv(OUT / "e12a_mortality_feasibility.csv", index=False)
    write_table("t17_feasibility.tex",
                [f"Dose groups in the design & {out['n_groups_total']} \\\\",
                 f"Groups experiencing any mortality & {out['n_groups_with_deaths']} \\\\",
                 f"\\quad of which lost every animal & {out['n_groups_wiped_out']} \\\\",
                 f"\\textbf{{Groups with deaths and surviving sampled fish}} & "
                 f"\\textbf{{{out['n_groups_with_deaths_and_survivors']}}} \\\\",
                 f"Surviving fish in those groups & "
                 f"{out['surviving_fish_in_positive_groups']} \\\\"],
                "lc", r"Quantity & Count")
    return out


# -----------------------------------------------------------------------------
def observed_effect_size(df):
    """Cohen's d on the DCRP severity axis, control vs exposed, per condition.

    Anchors the simulated effect sizes in something measured rather than
    assumed.
    """
    rows = []
    for c in CONDITIONS:
        d = subset(df, c)
        ctrl = d[d.concentration == 0]
        iv = reference_intervals(ctrl)
        sev = np.zeros(len(d))
        for m in BIOMARKERS:
            _, _, med, sd = iv[m]
            sev += DIRECTION[m] * (d[m].to_numpy() - med) / sd
        g0, g1 = sev[d.concentration.to_numpy() == 0], sev[d.concentration.to_numpy() > 0]
        sp = np.sqrt(((len(g0) - 1) * g0.var(ddof=1) +
                      (len(g1) - 1) * g1.var(ddof=1)) / (len(g0) + len(g1) - 2))
        rows.append(dict(condition=c.key, cohens_d=float((g1.mean() - g0.mean()) / sp)))
    return pd.DataFrame(rows)


def within_group_correlation(df):
    """Biomarker correlation after removing each dose group's mean."""
    blocks = []
    for c in CONDITIONS:
        d = subset(df, c)
        X = d[BIOMARKERS].to_numpy(float)
        for g in d.dose_group.unique():
            s = (d.dose_group == g).to_numpy()
            if s.sum() > 1:
                blocks.append(X[s] - X[s].mean(0))
    Z = np.vstack(blocks)
    Z = Z / np.where(Z.std(0, ddof=1) > 0, Z.std(0, ddof=1), 1.0)
    R = np.corrcoef(Z, rowvar=False)
    # nearest positive-definite repair
    w, V = np.linalg.eigh(R)
    R = V @ np.diag(np.maximum(w, 1e-6)) @ V.T
    dg = np.sqrt(np.diag(R))
    return R / np.outer(dg, dg)


def simulate_study(R, n_doses, n_tanks, n_per_tank, effect_d, icc, rng,
                   prevalence=0.35, null=False):
    """One synthetic confirmatory study with the measured covariance structure.

    Animals are nested in tanks, tanks in concentrations.  A latent severity
    rises with concentration; the adverse outcome is drawn from a logistic
    model on that severity; the eleven biomarkers are drawn around it with the
    real within-group correlation plus a tank random intercept of
    intraclass correlation `icc`.
    """
    p = R.shape[0]
    L = np.linalg.cholesky(R)
    conc_levels = np.linspace(0, 1, n_doses)
    sev, tank_id, dose_id, Xs = [], [], [], []
    t = 0
    for di, cl in enumerate(conc_levels):
        for _ in range(n_tanks):
            tank_eff = rng.normal(0, np.sqrt(icc))
            for _ in range(n_per_tank):
                s_i = 2.0 * cl + tank_eff + rng.normal(0, np.sqrt(max(1 - icc, 1e-6)))
                sev.append(s_i); tank_id.append(t); dose_id.append(di)
                Xs.append(L @ rng.normal(size=p))
            t += 1
    sev = np.asarray(sev); X = np.asarray(Xs)
    # Biomarkers carry the severity signal on a shared loading direction.
    loading = np.ones(p) / np.sqrt(p)
    X = X + (effect_d * sev)[:, None] * loading[None, :]
    beta = 0.0 if null else 1.6
    eta = beta * (sev - sev.mean()) / (sev.std() + 1e-9)
    eta = eta + (np.log(prevalence / (1 - prevalence)) - eta.mean())
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return X, y, np.asarray(tank_id), np.asarray(dose_id)


def blocked_auc(X, y, tank_id):
    """Leave-one-tank-out pooled out-of-fold AUC."""
    if len(np.unique(y)) < 2:
        return np.nan
    oof = np.full(len(y), np.nan)
    for t in np.unique(tank_id):
        te = tank_id == t
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        mdl = Pipeline([("s", StandardScaler()),
                        ("m", LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"))])
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return fast_auc(y[ok], oof[ok]) if len(np.unique(y[ok])) > 1 else np.nan


def exp12b_power(df, n_rep=150, n_doses=5, seed=11):
    print("[E12b] power analysis for the confirmatory study")
    R = within_group_correlation(df)
    np.save(OUT / "e12_within_group_corr.npy", R)
    eff = observed_effect_size(df)
    eff.to_csv(OUT / "e12_observed_effect_size.csv", index=False)
    print(f"    measured Cohen's d (control vs exposed, severity axis): "
          f"{', '.join(f'{v:.2f}' for v in eff.cohens_d)}")

    rng = np.random.default_rng(seed)
    rows = []
    for icc in (0.0, 0.15):
        for d_eff in (0.3, 0.5, 0.8):
            for n_tanks in (1, 2, 3):
                for n_per in (7, 10, 15):
                    null_aucs, alt_aucs = [], []
                    for _ in range(n_rep):
                        X, y, tk, _ = simulate_study(R, n_doses, n_tanks, n_per,
                                                     d_eff, icc, rng, null=True)
                        v = blocked_auc(X, y, tk)
                        if np.isfinite(v):
                            null_aucs.append(v)
                        X, y, tk, _ = simulate_study(R, n_doses, n_tanks, n_per,
                                                     d_eff, icc, rng, null=False)
                        v = blocked_auc(X, y, tk)
                        if np.isfinite(v):
                            alt_aucs.append(v)
                    if len(null_aucs) < 20 or len(alt_aucs) < 20:
                        continue
                    crit = float(np.quantile(null_aucs, 0.95))
                    rows.append(dict(
                        icc=icc, effect_d=d_eff, n_tanks=n_tanks,
                        n_per_tank=n_per,
                        n_total=n_doses * n_tanks * n_per,
                        mean_auc=float(np.mean(alt_aucs)),
                        auc_q05=float(np.quantile(alt_aucs, .05)),
                        null_crit=crit,
                        power=float(np.mean(np.asarray(alt_aucs) > crit))))
                    print(f"    icc={icc} d={d_eff} tanks={n_tanks} n/tank={n_per} "
                          f"-> AUC {rows[-1]['mean_auc']:.2f} power {rows[-1]['power']:.2f}",
                          flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "e12b_power.csv", index=False)

    main = P[P.icc == 0.15]
    body = []
    for d_eff in sorted(main.effect_d.unique()):
        for n_tanks in sorted(main.n_tanks.unique()):
            cells = []
            for n_per in sorted(main.n_per_tank.unique()):
                s = main[(main.effect_d == d_eff) & (main.n_tanks == n_tanks)
                         & (main.n_per_tank == n_per)]
                cells.append("--" if s.empty else
                             f"{s.iloc[0].power:.2f} ({s.iloc[0].mean_auc:.2f})")
            body.append(f"{d_eff:g} & {n_tanks} & " + " & ".join(cells) + r" \\")
    write_table("t18_power.tex", body, "llccc",
                (r"Effect $d$ & Tanks/conc. & 7 fish/tank & 10 fish/tank "
                 r"& 15 fish/tank"))
    return P, eff, R


