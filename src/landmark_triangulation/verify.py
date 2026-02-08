# src/landmark_triangulation/verify.py
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs

# Import from the local package relative to this file
from .core import LandmarkTriangulation


def run_diagnostic():
    print("--- Landmark Triangulation Linux/System Diagnostic ---")
    try:
        X, y = make_blobs(n_samples=500, n_features=20, centers=3, random_state=42)
        lt = LandmarkTriangulation(n_landmarks=50, landmark_mode="hybrid")
        embedding = lt.fit_transform(X)

        print(f"✓ Numerical Core: OK (Shape: {embedding.shape})")
        print(f"✓ Dependencies: NumPy {np.__version__} detected")
        return True
    except Exception as e:
        print(f"! Diagnostic Failed: {e}")
        return False


if __name__ == "__main__":
    success = run_diagnostic()
    exit(0 if success else 1)
