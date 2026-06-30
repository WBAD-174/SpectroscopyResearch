import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def load_dataset(modality: str, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spectroscopy dataset.

    Args:
        modality: 'NIR' or 'FTIR'
        condition: 'Dry' or 'Wet'

    Returns:
        X: (n_samples, n_wavelengths) spectral matrix
        y: (n_samples,) class labels as integers
        wavenumbers: (n_wavelengths,) wavenumber axis
    """
    path = DATA_DIR / f"2023{condition}-{modality}.xlsx"
    df = pd.read_excel(path)

    sample_col = "sample" if "sample" in df.columns else "sanmple"
    meta_cols = [sample_col, "group"]
    wavenumber_cols = [c for c in df.columns if c not in meta_cols]

    X = df[wavenumber_cols].values.astype(np.float64)
    wavenumbers = np.array(wavenumber_cols, dtype=np.float64)

    labels = df["group"].values
    classes, y = np.unique(labels, return_inverse=True)

    return X, y, wavenumbers, classes


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate normalization (per-sample)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-10)
