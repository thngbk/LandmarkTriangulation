"""Tests for numerical stability and correctness."""

import numpy as np

from landmark_triangulation import LandmarkTriangulation


class TestNumericalStability:
    """Test numerical stability of the algorithm."""

    def test_no_nans(self, simple_data):
        """Test that output contains no NaNs."""
        lt = LandmarkTriangulation(n_landmarks=10)
        X_transformed = lt.fit_transform(simple_data)

        assert not np.any(np.isnan(X_transformed))

    def test_no_infs(self, simple_data):
        """Test that output contains no infinities."""
        lt = LandmarkTriangulation(n_landmarks=10)
        X_transformed = lt.fit_transform(simple_data)

        assert not np.any(np.isinf(X_transformed))

    def test_finite_landmarks(self, simple_data):
        """Test that landmarks are finite."""
        lt = LandmarkTriangulation(n_landmarks=10)
        lt.fit(simple_data)

        assert np.all(np.isfinite(lt.landmarks_high_))
        assert np.all(np.isfinite(lt.landmarks_low_))

    def test_scaled_data(self, simple_data):
        """Test with pre-scaled data."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(simple_data)

        lt = LandmarkTriangulation(n_landmarks=10)
        X_transformed = lt.fit_transform(X_scaled)

        assert np.all(np.isfinite(X_transformed))


class TestOutputProperties:
    """Test properties of the output embedding."""

    def test_output_variance(self, blob_data):
        """Test that output has reasonable variance."""
        X, _ = blob_data
        lt = LandmarkTriangulation(n_landmarks=50)
        X_transformed = lt.fit_transform(X)

        # Output should have non-zero variance
        variance = np.var(X_transformed, axis=0)
        assert np.all(variance > 0)

    def test_preserves_sample_count(self, simple_data):
        """Test that all samples are preserved."""
        lt = LandmarkTriangulation(n_landmarks=10)
        X_transformed = lt.fit_transform(simple_data)

        assert X_transformed.shape[0] == simple_data.shape[0]
