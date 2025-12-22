"""Tests for core LandmarkTriangulation functionality."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from landmark_triangulation import LandmarkTriangulation


class TestInitialization:
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default parameters."""
        lt = LandmarkTriangulation()
        assert lt.n_components == 2
        assert lt.n_landmarks == 50
        assert lt.landmark_mode == "random"
        assert lt.random_state == 42

    def test_custom_initialization(self):
        """Test custom parameters."""
        lt = LandmarkTriangulation(
            n_components=3, n_landmarks=100, landmark_mode="hybrid", random_state=123
        )
        assert lt.n_components == 3
        assert lt.n_landmarks == 100
        assert lt.landmark_mode == "hybrid"
        assert lt.random_state == 123


class TestFit:
    """Test fit method."""

    def test_fit_returns_self(self, simple_data):
        """Test that fit returns self for chaining."""
        lt = LandmarkTriangulation(n_landmarks=10)
        result = lt.fit(simple_data)
        assert result is lt

    def test_fit_creates_attributes(self, simple_data):
        """Test that fit creates expected attributes."""
        lt = LandmarkTriangulation(n_landmarks=10)
        lt.fit(simple_data)

        # Check all required attributes exist
        assert hasattr(lt, "landmarks_high_")
        assert hasattr(lt, "landmarks_high_scaled_")
        assert hasattr(lt, "landmarks_low_")
        assert hasattr(lt, "scaler_")
        assert hasattr(lt, "reference_landmark_")
        assert hasattr(lt, "landmark_sq_norms_")
        assert hasattr(lt, "solver_matrix_")

    def test_fit_shapes(self, simple_data):
        """Test that fitted attributes have correct shapes."""
        n_landmarks = 20
        n_components = 2
        lt = LandmarkTriangulation(n_landmarks=n_landmarks, n_components=n_components)
        lt.fit(simple_data)

        assert lt.landmarks_high_.shape == (n_landmarks, simple_data.shape[1])
        assert lt.landmarks_high_scaled_.shape == (n_landmarks, simple_data.shape[1])
        assert lt.landmarks_low_.shape == (n_landmarks, n_components)
        assert lt.reference_landmark_.shape == (n_components,)
        assert lt.landmark_sq_norms_.shape == (n_landmarks,)
        assert lt.solver_matrix_.shape == (n_components, n_landmarks - 1)

    def test_fit_too_few_samples(self):
        """Test error when n_samples < n_landmarks in random mode."""
        X = np.random.randn(10, 5)
        lt = LandmarkTriangulation(n_landmarks=20, landmark_mode="random")

        with pytest.raises(ValueError, match="n_samples.*must be >= n_landmarks"):
            lt.fit(X)

    def test_fit_too_few_landmarks(self, simple_data):
        """Test error when n_landmarks < 2."""
        lt = LandmarkTriangulation(n_landmarks=1)

        with pytest.raises(ValueError, match="n_landmarks must be at least 2"):
            lt.fit(simple_data)

    def test_fit_invalid_n_components(self, simple_data):
        """Test error when n_components >= n_landmarks."""
        lt = LandmarkTriangulation(n_components=10, n_landmarks=10)

        with pytest.raises(
            ValueError, match="n_components.*must be less than.*n_landmarks"
        ):
            lt.fit(simple_data)


class TestTransform:
    """Test transform method."""

    def test_transform_not_fitted(self, simple_data):
        """Test error when transform called before fit."""
        lt = LandmarkTriangulation()

        with pytest.raises(NotFittedError):
            lt.transform(simple_data)

    def test_transform_shape(self, simple_data):
        """Test output shape of transform."""
        n_components = 2
        lt = LandmarkTriangulation(n_components=n_components, n_landmarks=10)
        lt.fit(simple_data)

        X_transformed = lt.transform(simple_data)
        assert X_transformed.shape == (simple_data.shape[0], n_components)

    def test_transform_different_data(self, simple_data, random_state):
        """Test transform on different data than fit."""
        lt = LandmarkTriangulation(n_landmarks=10, random_state=random_state)
        lt.fit(simple_data)

        # New data with same number of features
        X_new = np.random.randn(50, simple_data.shape[1])
        X_transformed = lt.transform(X_new)

        assert X_transformed.shape == (50, 2)

    def test_fit_transform(self, simple_data):
        """Test fit_transform produces same result as fit then transform."""
        lt1 = LandmarkTriangulation(n_landmarks=10, random_state=42)
        lt2 = LandmarkTriangulation(n_landmarks=10, random_state=42)

        X_fit_transform = lt1.fit_transform(simple_data)

        lt2.fit(simple_data)
        X_transform = lt2.transform(simple_data)

        np.testing.assert_array_almost_equal(X_fit_transform, X_transform)


class TestLandmarkModes:
    """Test different landmark selection modes."""

    def test_random_mode(self, simple_data):
        """Test random landmark mode."""
        lt = LandmarkTriangulation(
            n_landmarks=10, landmark_mode="random", random_state=42
        )
        lt.fit(simple_data)

        # Landmarks should be from the input data
        assert lt.landmarks_high_.shape == (10, simple_data.shape[1])

    def test_synthetic_mode(self, simple_data):
        """Test synthetic landmark mode."""
        lt = LandmarkTriangulation(
            n_landmarks=10, landmark_mode="synthetic", random_state=42
        )
        lt.fit(simple_data)

        assert lt.landmarks_high_.shape == (10, simple_data.shape[1])

    def test_hybrid_mode(self, simple_data):
        """Test hybrid landmark mode."""
        lt = LandmarkTriangulation(
            n_landmarks=10, landmark_mode="hybrid", random_state=42
        )
        lt.fit(simple_data)

        # Should select real data points
        assert lt.landmarks_high_.shape == (10, simple_data.shape[1])

    def test_invalid_mode(self, simple_data):
        """Test error with invalid landmark mode."""
        lt = LandmarkTriangulation(landmark_mode="invalid")

        with pytest.raises(ValueError, match="Invalid landmark_mode"):
            lt.fit(simple_data)


class TestDeterminism:
    """Test deterministic behavior with fixed random state."""

    def test_reproducibility(self, simple_data):
        """Test that results are reproducible with same random_state."""
        lt1 = LandmarkTriangulation(n_landmarks=10, random_state=42)
        lt2 = LandmarkTriangulation(n_landmarks=10, random_state=42)

        X1 = lt1.fit_transform(simple_data)
        X2 = lt2.fit_transform(simple_data)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_different_random_states(self, simple_data):
        """Test that different random states produce different results."""
        lt1 = LandmarkTriangulation(n_landmarks=10, random_state=42)
        lt2 = LandmarkTriangulation(n_landmarks=10, random_state=123)

        X1 = lt1.fit_transform(simple_data)
        X2 = lt2.fit_transform(simple_data)

        # Results should be different
        assert not np.allclose(X1, X2)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_dimension_output(self, simple_data):
        """Test with n_components=1."""
        lt = LandmarkTriangulation(n_components=1, n_landmarks=10)
        X_transformed = lt.fit_transform(simple_data)

        assert X_transformed.shape == (simple_data.shape[0], 1)

    def test_high_dimensional_output(self, simple_data):
        """Test with n_components > 2."""
        lt = LandmarkTriangulation(n_components=5, n_landmarks=20)
        X_transformed = lt.fit_transform(simple_data)

        assert X_transformed.shape == (simple_data.shape[0], 5)

    def test_many_landmarks(self, large_data):
        """Test with many landmarks."""
        lt = LandmarkTriangulation(n_landmarks=500)
        X_transformed = lt.fit_transform(large_data)

        assert X_transformed.shape == (large_data.shape[0], 2)

    def test_minimum_configuration(self, small_data):
        """Test minimum viable configuration."""
        # Minimum: 2 landmarks, 1 component
        lt = LandmarkTriangulation(n_components=1, n_landmarks=2)
        X_transformed = lt.fit_transform(small_data)

        assert X_transformed.shape == (small_data.shape[0], 1)


class TestDataTypes:
    """Test handling of different data types."""

    def test_float32_input(self, simple_data):
        """Test with float32 input."""
        X_float32 = simple_data.astype(np.float32)
        lt = LandmarkTriangulation(n_landmarks=10)
        X_transformed = lt.fit_transform(X_float32)

        assert X_transformed.dtype == np.float64

    def test_list_input(self):
        """Test with list input (should be converted to array)."""
        X_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
        lt = LandmarkTriangulation(n_landmarks=3)
        X_transformed = lt.fit_transform(X_list)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == (5, 2)
