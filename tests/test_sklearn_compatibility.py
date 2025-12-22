"""Tests for scikit-learn compatibility."""

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from landmark_triangulation import LandmarkTriangulation


class TestSklearnAPI:
    """Test scikit-learn API compliance."""

    def test_estimator_interface(self):
        """Test that class follows estimator interface."""
        from sklearn.base import BaseEstimator, TransformerMixin

        lt = LandmarkTriangulation()
        assert isinstance(lt, BaseEstimator)
        assert isinstance(lt, TransformerMixin)

    def test_get_params(self):
        """Test get_params method."""
        lt = LandmarkTriangulation(
            n_components=3, n_landmarks=100, landmark_mode="hybrid"
        )

        params = lt.get_params()
        assert params["n_components"] == 3
        assert params["n_landmarks"] == 100
        assert params["landmark_mode"] == "hybrid"

    def test_set_params(self):
        """Test set_params method."""
        lt = LandmarkTriangulation()
        lt.set_params(n_components=3, n_landmarks=100)

        assert lt.n_components == 3
        assert lt.n_landmarks == 100

    def test_clone(self):
        """Test that estimator can be cloned."""
        lt = LandmarkTriangulation(n_components=3, random_state=42)
        lt_clone = clone(lt)

        assert lt_clone.n_components == 3
        assert lt_clone.random_state == 42
        assert lt_clone is not lt


class TestPipeline:
    """Test integration with sklearn pipelines."""

    def test_pipeline_basic(self, simple_data):
        """Test basic pipeline usage."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reducer", LandmarkTriangulation(n_landmarks=10)),
            ]
        )

        X_transformed = pipeline.fit_transform(simple_data)
        assert X_transformed.shape == (simple_data.shape[0], 2)

    def test_pipeline_chaining(self, simple_data):
        """Test that pipeline can be chained."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reducer", LandmarkTriangulation(n_landmarks=10)),
            ]
        )

        result = pipeline.fit(simple_data)
        assert result is pipeline

        X_transformed = pipeline.transform(simple_data)
        assert X_transformed.shape == (simple_data.shape[0], 2)
