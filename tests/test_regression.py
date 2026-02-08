"""Regression tests to ensure algorithm changes don't break expected behavior.

Golden outputs generated from commit 2e0552a (baseline before refurbishment).
"""

from pathlib import Path

import numpy as np
import pytest

from landmark_triangulation import LandmarkTriangulation

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


@pytest.fixture
def input_data():
    """Load the fixed input data used for golden outputs."""
    return np.load(GOLDEN_DIR / "input_data.npy")


def test_random_mode_2d_regression(input_data):
    """Verify random mode 2D output matches baseline."""
    expected = np.load(GOLDEN_DIR / "random_mode_2d.npy")

    transformer = LandmarkTriangulation(
        n_components=2, n_landmarks=50, landmark_mode="random", random_state=42
    )
    result = transformer.fit_transform(input_data)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_random_mode_3d_regression(input_data):
    """Verify random mode 3D output matches baseline."""
    expected = np.load(GOLDEN_DIR / "random_mode_3d.npy")

    transformer = LandmarkTriangulation(
        n_components=3, n_landmarks=50, landmark_mode="random", random_state=42
    )
    result = transformer.fit_transform(input_data)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_synthetic_mode_2d_regression(input_data):
    """Verify synthetic mode 2D output matches baseline."""
    expected = np.load(GOLDEN_DIR / "synthetic_mode_2d.npy")

    transformer = LandmarkTriangulation(
        n_components=2, n_landmarks=50, landmark_mode="synthetic", random_state=42
    )
    result = transformer.fit_transform(input_data)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_hybrid_mode_2d_regression(input_data):
    """Verify hybrid mode 2D output matches baseline."""
    expected = np.load(GOLDEN_DIR / "hybrid_mode_2d.npy")

    transformer = LandmarkTriangulation(
        n_components=2, n_landmarks=50, landmark_mode="hybrid", random_state=42
    )
    result = transformer.fit_transform(input_data)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
