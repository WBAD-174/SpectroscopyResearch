import numpy as np


def make_regions(wavenumbers: np.ndarray, n_regions: int) -> list[tuple[int, int]]:
    """
    Divide the wavenumber axis into n_regions equal-width regions.
    Returns list of (start_idx, end_idx) index pairs.
    """
    n = len(wavenumbers)
    edges = np.linspace(0, n, n_regions + 1, dtype=int)
    return [(edges[i], edges[i + 1]) for i in range(n_regions)]


def region_features(X: np.ndarray, regions: list[tuple[int, int]]) -> np.ndarray:
    """
    Compute shape descriptors for each region.
    Descriptor = [mean, slope] concatenated across all regions.

    Returns:
        X_reg: (n_samples, n_regions * 2)
    """
    features = []
    for start, end in regions:
        segment = X[:, start:end]
        mean = segment.mean(axis=1)

        # slope via linear regression on region index
        idx = np.arange(end - start, dtype=np.float64)
        idx_c = idx - idx.mean()
        denom = (idx_c ** 2).sum()
        if denom > 0:
            slope = (segment * idx_c).sum(axis=1) / denom
        else:
            slope = np.zeros(X.shape[0])

        features.extend([mean, slope])

    return np.column_stack(features)
