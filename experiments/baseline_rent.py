"""
RENT-style stability selection baseline.
Floor track from proposal: region-level features + post-hoc stability selection.

Usage:
    conda run -n spectroscopy-research python experiments/baseline_rent.py
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
from src.stability import rent_selection, kuncheva_index

# ── Config ──────────────────────────────────────────────────────────────────
MODALITY    = "NIR"
CONDITION   = "Dry"
N_REGIONS   = 50       # divide spectrum into 50 equal regions
N_MODELS    = 100      # number of RENT subsampled models
SUBSAMPLE   = 0.7      # fraction of samples per model
STAB_THRESH = 0.5      # min selection frequency to keep a region feature
C_LASSO     = 0.1      # L1 regularization strength
N_FOLDS     = 5        # cross-validation folds
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def run():
    print(f"=== RENT Baseline | {CONDITION}-{MODALITY} ===\n")

    # 1. Load and preprocess
    X, y, wavenumbers, classes = load_dataset(MODALITY, CONDITION)
    X = snv(X)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} wavenumbers, {len(classes)} classes")

    # 2. Region-level features
    regions = make_regions(wavenumbers, N_REGIONS)
    X_reg = region_features(X, regions)
    n_region_features = X_reg.shape[1]
    print(f"Region features: {N_REGIONS} regions × 2 descriptors = {n_region_features} features\n")

    # 3. Outer cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_reg, y)):
        X_train, X_test = X_reg[train_idx], X_reg[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 4. RENT stability selection on training fold only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        selected_mask, ki, subsets = rent_selection(
            X_train_s, y_train,
            n_models=N_MODELS,
            subsample_ratio=SUBSAMPLE,
            stability_threshold=STAB_THRESH,
            C=C_LASSO,
            random_state=RANDOM_SEED + fold,
        )

        n_selected = selected_mask.sum()

        if n_selected == 0:
            print(f"  Fold {fold+1}: no features selected, skipping")
            continue

        # 5. Train SVM on selected features
        X_tr_sel = X_train_s[:, selected_mask]
        X_te_sel = X_test_s[:, selected_mask]

        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
        clf.fit(X_tr_sel, y_train)
        y_pred = clf.predict(X_te_sel)

        acc = accuracy_score(y_test, y_pred)
        compactness = n_selected / n_region_features  # fraction of features kept

        fold_results.append({
            "fold": fold + 1,
            "accuracy": acc,
            "kuncheva_index": ki,
            "n_selected": int(n_selected),
            "compactness": compactness,
        })

        print(
            f"  Fold {fold+1}: "
            f"Acc={acc:.3f}  "
            f"KI={ki:.3f}  "
            f"Selected={n_selected}/{n_region_features}  "
            f"Compact={compactness:.3f}"
        )

    # 6. Summary
    print("\n=== Summary ===")
    accs = [r["accuracy"] for r in fold_results]
    kis  = [r["kuncheva_index"] for r in fold_results]
    sels = [r["n_selected"] for r in fold_results]

    print(f"Accuracy:        {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Kuncheva Index:  {np.mean(kis):.3f} ± {np.std(kis):.3f}")
    print(f"Features selected: {np.mean(sels):.1f} ± {np.std(sels):.1f} / {n_region_features}")
    print(f"Compactness:     {np.mean([r['compactness'] for r in fold_results]):.3f}")

    # 7. Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/baseline_rent_{CONDITION}_{MODALITY}.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "modality": MODALITY, "condition": CONDITION,
                "n_regions": N_REGIONS, "n_models": N_MODELS,
                "subsample_ratio": SUBSAMPLE, "stability_threshold": STAB_THRESH,
            },
            "folds": fold_results,
            "summary": {
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "kuncheva_mean": float(np.mean(kis)),
                "kuncheva_std": float(np.std(kis)),
                "n_selected_mean": float(np.mean(sels)),
            }
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
