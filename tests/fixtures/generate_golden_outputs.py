"""Generate golden outputs from baseline commit for regression testing.

This script creates reference outputs (golden files) from a known-good version
of the algorithm. These outputs are used in regression tests to ensure that
code changes don't unintentionally alter the algorithm's behavior.

Usage:
    1. Checkout the baseline commit: git checkout 2e0552a
    2. Run this script: python tests/fixtures/generate_golden_outputs.py
    3. Type 'yes' when prompted
    4. Return to your branch: git checkout your-branch

Warning:
    This script will OVERWRITE existing golden outputs. Only run when you
    intentionally want to update the baseline (e.g., after an algorithm improvement).

Generated Files:
    - tests/fixtures/golden/input_data.npy: Fixed test input (500×20)
    - tests/fixtures/golden/random_mode_2d.npy: Random mode 2D embedding
    - tests/fixtures/golden/random_mode_3d.npy: Random mode 3D embedding
    - tests/fixtures/golden/synthetic_mode_2d.npy: Synthetic mode embedding
    - tests/fixtures/golden/hybrid_mode_2d.npy: Hybrid mode embedding
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from landmark_triangulation import LandmarkTriangulation

# Fixed random seed for reproducibility
SEED: int = 42
OUTPUT_DIR: Path = Path(__file__).parent / "golden"


def generate_golden_outputs() -> None:
    """Generate and save golden output files for regression testing.

    Creates fixed test data and generates embeddings using all landmark modes.
    Saves results as .npy files in the golden/ directory.

    Raises:
        SystemExit: If user doesn't confirm the operation.
    """
    # Safety check
    print("WARNING: This script will overwrite golden outputs!")
    print(
        "Only run this at baseline commit (2e0552a) or when intentionally updating baselines."
    )
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        sys.exit(0)

    # Generate test data
    np.random.seed(SEED)
    X: NDArray[np.float64] = np.random.rand(500, 20)  # 500 samples, 20 features

    # Save input data
    np.save(OUTPUT_DIR / "input_data.npy", X)
    print(f"✓ Saved input_data.npy: {X.shape}")

    # Test cases: (mode, n_components, n_landmarks)
    test_cases: list[tuple[str, int, int]] = [
        ("random", 2, 50),
        ("random", 3, 50),
        ("synthetic", 2, 50),
        ("hybrid", 2, 50),
    ]

    for mode, n_comp, n_land in test_cases:
        transformer = LandmarkTriangulation(
            n_components=n_comp,
            n_landmarks=n_land,
            landmark_mode=mode,
            random_state=SEED,
        )
        embedding: NDArray[np.float64] = transformer.fit_transform(X)

        filename = f"{mode}_mode_{n_comp}d.npy"
        np.save(OUTPUT_DIR / filename, embedding)
        print(f"✓ Saved {filename}: {embedding.shape}")

    print(f"\n✅ All golden outputs generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_golden_outputs()
