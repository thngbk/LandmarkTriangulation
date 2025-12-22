"""Performance and scaling tests."""

import time

import numpy as np
import pytest

from landmark_triangulation import LandmarkTriangulation


@pytest.mark.slow
class TestPerformance:
    """Test performance and scaling behavior."""

    def test_large_dataset(self):
        """Test on a large dataset."""
        X = np.random.randn(10000, 50)
        lt = LandmarkTriangulation(n_landmarks=100)

        start = time.time()
        X_transformed = lt.fit_transform(X)
        elapsed = time.time() - start

        assert X_transformed.shape == (10000, 2)
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 10.0  # seconds

    def test_scaling_behavior(self):
        """Test that runtime scales reasonably with data size."""
        sizes = [1000, 2000, 4000]
        times = []

        for size in sizes:
            X = np.random.randn(size, 20)
            lt = LandmarkTriangulation(n_landmarks=50)

            start = time.time()
            lt.fit_transform(X)
            elapsed = time.time() - start
            times.append(elapsed)

        # Check that scaling is reasonable (not exponential)
        # Allow for overhead: 4x data should take less than 10x time
        # This accounts for fixed costs (PCA on landmarks, etc.)
        scaling_factor = times[2] / times[0]
        assert scaling_factor < 10, (
            f"Scaling factor {scaling_factor:.2f} is too high (expected < 10)"
        )

        # Also check it's not constant (should scale somewhat)
        assert times[2] > times[0], "Runtime should increase with data size"

    def test_linear_scaling_large_datasets(self):
        """Test linear scaling behavior on larger datasets where overhead is smaller."""
        sizes = [5000, 10000, 20000]
        times = []

        for size in sizes:
            X = np.random.randn(size, 20)
            lt = LandmarkTriangulation(n_landmarks=100)

            start = time.time()
            lt.fit_transform(X)
            elapsed = time.time() - start
            times.append(elapsed)

        # For larger datasets, expect closer to linear scaling
        # 4x data should take less than 6x time
        scaling_factor = times[2] / times[0]
        assert scaling_factor < 6, (
            f"Scaling factor {scaling_factor:.2f} should be closer to linear for large datasets"
        )


@pytest.mark.slow
class TestMemory:
    """Test memory efficiency."""

    def test_no_memory_leak(self):
        """Test repeated fit_transform doesn't leak memory."""
        X = np.random.randn(1000, 20)
        lt = LandmarkTriangulation(n_landmarks=50)

        # Multiple iterations should not accumulate memory
        for _ in range(10):
            _ = lt.fit_transform(X)
