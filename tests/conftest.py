"""Shared pytest fixtures for testing."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_swiss_roll


@pytest.fixture
def random_state():
    """Fixed random state for reproducibility."""
    return 42


@pytest.fixture
def simple_data():
    """Small dataset for quick tests."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def large_data():
    """Larger dataset for performance tests."""
    np.random.seed(42)
    return np.random.randn(1000, 20)


@pytest.fixture
def blob_data(random_state):
    """Clustered data with known structure."""
    X, y = make_blobs(
        n_samples=500,
        n_features=15,
        centers=5,
        cluster_std=1.0,
        random_state=random_state,
    )
    return X, y


@pytest.fixture
def swiss_roll_data(random_state):
    """Swiss roll manifold data."""
    X, t = make_swiss_roll(n_samples=500, noise=0.1, random_state=random_state)
    return X, t


@pytest.fixture
def small_data():
    """Very small dataset for edge case testing."""
    return np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ]
    )
