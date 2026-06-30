import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression


def kuncheva_index(S: list[set], p: int) -> float:
    """
    Kuncheva stability index for a list of feature subsets.

    Args:
        S: list of selected feature index sets
        p: total number of features

    Returns:
        KI in [-1, 1]; higher = more stable
    """
    k = len(S)
    if k < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            si, sj = len(S[i]), len(S[j])
            intersect = len(S[i] & S[j])
            r = si * sj / p
            denom = max(si, sj) - r
            if denom == 0:
                continue
            total += (intersect - r) / denom
            count += 1

    return total / count if count > 0 else 0.0


def _fit_one(X, y, subsample_ratio, C, seed):
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    idx = rng.choice(n_samples, size=int(n_samples * subsample_ratio), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    if len(np.unique(y_sub)) < 2:
        return None

    clf = LogisticRegression(
        l1_ratio=1.0,
        C=C,
        solver="saga",
        max_iter=3000,
        tol=1e-2,
        random_state=int(rng.integers(1e6)),
    )
    clf.fit(X_sub, y_sub)
    selected = (np.abs(clf.coef_).sum(axis=0) > 0)
    return set(np.where(selected)[0])


def rent_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_models: int = 100,
    subsample_ratio: float = 0.7,
    stability_threshold: float = 0.5,
    C: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[np.ndarray, float, list[set]]:
    """
    RENT-style stability selection via repeated subsampled L1 logistic regression.
    Runs in parallel across all available CPU cores.

    Returns:
        selected_mask: boolean array of shape (n_features,)
        ki: Kuncheva stability index
        subsets: list of selected feature index sets per model
    """
    seeds = np.random.SeedSequence(random_state).spawn(n_models)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(X, y, subsample_ratio, C, seed)
        for seed in seeds
    )

    subsets = [r for r in results if r is not None]
    n_features = X.shape[1]
    selection_counts = np.zeros(n_features)
    for s in subsets:
        for i in s:
            selection_counts[i] += 1

    freq = selection_counts / n_models
    selected_mask = freq >= stability_threshold
    ki = kuncheva_index(subsets, n_features)

    return selected_mask, ki, subsets
