"""
Pareto front sweep over STAB_THRESH.
Produces accuracy / stability / compactness trade-off curves for each dataset.

Usage:
    conda run -n spectroscopy-research python experiments/pareto_sweep.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.data import load_dataset, snv
from src.regions import make_regions, region_features
from src.stability import rent_selection

# ── Config ───────────────────────────────────────────────────────────────────
DATASETS = [
    ("NIR",  "Dry"),
    ("NIR",  "Wet"),
    ("FTIR", "Dry"),
    ("FTIR", "Wet"),
]
STAB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_REGIONS   = 50
N_MODELS    = 100
SUBSAMPLE   = 0.7
C_LASSO     = 0.1
N_FOLDS     = 5
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def run_one(X_reg, y, threshold):
    """Run 5-fold CV for a single STAB_THRESH value. Returns summary dict."""
    n_region_features = X_reg.shape[1]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    accs, kis, sels = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_reg, y)):
        X_train, X_test = X_reg[train_idx], X_reg[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        selected_mask, ki, _ = rent_selection(
            X_train_s, y_train,
            n_models=N_MODELS,
            subsample_ratio=SUBSAMPLE,
            stability_threshold=threshold,
            C=C_LASSO,
            random_state=RANDOM_SEED + fold,
        )

        n_selected = selected_mask.sum()
        if n_selected == 0:
            continue

        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
        clf.fit(X_train_s[:, selected_mask], y_train)
        y_pred = clf.predict(X_test_s[:, selected_mask])

        accs.append(accuracy_score(y_test, y_pred))
        kis.append(ki)
        sels.append(n_selected)

    if not accs:
        return None

    return {
        "stab_threshold": threshold,
        "accuracy_mean":    float(np.mean(accs)),
        "accuracy_std":     float(np.std(accs)),
        "kuncheva_mean":    float(np.mean(kis)),
        "kuncheva_std":     float(np.std(kis)),
        "n_selected_mean":  float(np.mean(sels)),
        "compactness_mean": float(np.mean(sels)) / n_region_features,
    }


def run_dataset(modality, condition):
    print(f"\n=== Pareto Sweep | {condition}-{modality} ===")

    X, y, wavenumbers, classes = load_dataset(modality, condition)
    X = snv(X)
    regions = make_regions(wavenumbers, N_REGIONS)
    X_reg = region_features(X, regions)

    print(f"{'Thresh':>7}  {'Acc':>6}  {'KI':>6}  {'Selected':>8}  {'Compact':>7}")
    print("-" * 50)

    points = []
    for thresh in STAB_THRESHOLDS:
        result = run_one(X_reg, y, thresh)
        if result is None:
            print(f"  {thresh:.1f}: no features selected, skipped")
            continue
        points.append(result)
        print(
            f"  {thresh:.1f}    "
            f"{result['accuracy_mean']:.3f}  "
            f"{result['kuncheva_mean']:.3f}  "
            f"{result['n_selected_mean']:>6.1f}    "
            f"{result['compactness_mean']:.3f}"
        )

    os.makedirs("results", exist_ok=True)
    out_path = f"results/pareto_{condition}_{modality}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "modality": modality, "condition": condition,
                "n_regions": N_REGIONS, "n_models": N_MODELS,
                "stab_thresholds": STAB_THRESHOLDS,
            },
            "points": points,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    for modality, condition in DATASETS:
        run_dataset(modality, condition)
