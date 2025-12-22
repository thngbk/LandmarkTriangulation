"""Landmark-based dimensionality reduction implementation."""

from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class LandmarkTriangulation(BaseEstimator, TransformerMixin):
    """
    Landmark-based dimensionality reduction using triangulation and alpha-refinement.

    This transformer reduces dimensionality by selecting landmark points and using
    geometric triangulation to embed new points in a lower-dimensional space.

    Args:
        n_components: Number of dimensions for the output embedding. Default is 2.
        n_landmarks: Number of landmark points to sample. Default is 50.
        landmark_mode: Strategy for selecting landmarks. Must be one of:
            - 'random': Randomly samples points from the input data.
            - 'synthetic': Generates synthetic points based on a sine-wave manifold.
            - 'hybrid': Generates synthetic points and snaps them to nearest real points.
            Default is 'random'.
        random_state: Controls randomness of landmark selection/generation.
            Pass an int for reproducible output. Default is 42.

    Attributes:
        landmarks_high_: Selected landmark points in original high-dimensional space.
            Shape (n_landmarks, n_features).
        landmarks_high_scaled_: Scaled landmark points in high-dimensional space.
            Shape (n_landmarks, n_features).
        landmarks_low_: Landmark points in low-dimensional embedding space.
            Shape (n_landmarks, n_components).
        scaler_: StandardScaler fitted on the training data.
        reference_landmark_: First landmark in low-dimensional space (reference point).
            Shape (n_components,).
        landmark_sq_norms_: Squared norms of landmarks in low-dimensional space.
            Shape (n_landmarks,).
        solver_matrix_: Precomputed pseudoinverse for the triangulation system.
            Shape (n_components, n_landmarks-1).

    Examples:
        >>> import numpy as np
        >>> from landmark_triangulation import LandmarkTriangulation
        >>>
        >>> X = np.random.randn(100, 10)
        >>> transformer = LandmarkTriangulation(n_components=2, n_landmarks=20)
        >>> X_embedded = transformer.fit_transform(X)
        >>> X_embedded.shape
        (100, 2)

    References:
        Add your paper/method reference here if applicable.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_landmarks: int = 50,
        landmark_mode: Literal["random", "synthetic", "hybrid"] = "random",
        random_state: Optional[int] = 42,
    ) -> None:
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.landmark_mode = landmark_mode
        self.random_state = random_state

    def _generate_synthetic_landmarks(self, n_features: int) -> NDArray[np.float64]:
        """
        Generate synthetic sine-based landmarks.

        Args:
            n_features: Number of features in the original data space.

        Returns:
            Synthetically generated landmark points with shape (n_landmarks, n_features).
        """
        rng = np.random.RandomState(self.random_state)
        amplitudes = rng.uniform(0.5, 2.0, n_features)
        frequencies = rng.uniform(0.5, 1.5, n_features)
        phases = rng.uniform(0, 2 * np.pi, n_features)
        t = np.linspace(0, 2 * np.pi, self.n_landmarks)

        # Shape: (n_features, n_landmarks) -> transpose to (n_landmarks, n_features)
        landmarks = amplitudes[:, None] * np.sin(
            frequencies[:, None] * t + phases[:, None]
        )
        return landmarks.T

    def _select_landmarks(
        self, X: NDArray[np.float64], rng: np.random.RandomState
    ) -> NDArray[np.float64]:
        """
        Select landmarks based on the specified mode.

        Args:
            X: Input data with shape (n_samples, n_features).
            rng: Random number generator.

        Returns:
            Selected landmarks with shape (n_landmarks, n_features).

        Raises:
            ValueError: If landmark_mode is invalid or n_samples < n_landmarks.
        """
        n_samples, n_features = X.shape

        if self.landmark_mode == "synthetic":
            # Pure synthetic landmarks (sine-wave based)
            return self._generate_synthetic_landmarks(n_features)

        elif self.landmark_mode == "hybrid":
            # Generate synthetic points (ghosts) and snap to nearest real points
            synthetic = self._generate_synthetic_landmarks(n_features)

            # Snap ghosts to nearest real data points (Manifold Snapping)
            # Temporarily normalize for fair distance computation
            temp_scaler = StandardScaler().fit(X)
            X_normalized = temp_scaler.transform(X)

            # # Scale synthetic landmarks to match the normalized space
            # synthetic_normalized = temp_scaler.transform(synthetic)

            # # Calculate indices of the nearest real neighbor for every ghost
            # nearest_indices = pairwise_distances_argmin(
            #     synthetic_normalized, X_normalized
            # )
            # Calculate indices of the nearest real neighbor for every ghost
            nearest_indices = pairwise_distances_argmin(synthetic, X_normalized)
            return X[nearest_indices].copy()

        elif self.landmark_mode == "random":
            # Random sampling from input data
            if n_samples < self.n_landmarks:
                raise ValueError(
                    f"n_samples ({n_samples}) must be >= n_landmarks ({self.n_landmarks})"
                )
            indices = rng.choice(n_samples, size=self.n_landmarks, replace=False)
            return X[indices].copy()

        else:
            raise ValueError(
                f"Invalid landmark_mode: '{self.landmark_mode}'. "
                "Must be one of ['random', 'synthetic', 'hybrid']"
            )

    def _compute_scaling_factor(
        self,
        landmarks_high: NDArray[np.float64],
        landmarks_low: NDArray[np.float64],
    ) -> float:
        """
        Compute RMS scaling factor to match high-dimensional and low-dimensional scales.

        Args:
            landmarks_high: Landmarks in high-dimensional space.
            landmarks_low: Landmarks in low-dimensional space.

        Returns:
            Scaling factor.
        """

        def sum_squared_distances(arr: NDArray[np.float64]) -> float:
            """Calculate sum of squared pairwise distances."""
            diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
            return float(np.sum(diff**2))

        ss_high = sum_squared_distances(landmarks_high)
        ss_low = sum_squared_distances(landmarks_low)

        return np.sqrt(ss_high / ss_low) if ss_low > 1e-10 else 1.0

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "LandmarkTriangulation":
        """
        Fit the transformer by selecting landmarks and computing the embedding.

        Args:
            X: Training data with shape (n_samples, n_features).
            y: Ignored. Present for scikit-learn API consistency.

        Returns:
            Fitted transformer instance.

        Raises:
            ValueError: If n_samples < n_landmarks when using random mode,
                or if landmark_mode is invalid, or if n_landmarks < 2,
                or if n_components >= n_landmarks.
        """
        X = check_array(X, dtype=np.float64)

        # Validate parameters
        if self.n_landmarks < 2:
            raise ValueError(f"n_landmarks must be at least 2, got {self.n_landmarks}")

        if self.n_components >= self.n_landmarks:
            raise ValueError(
                f"n_components ({self.n_components}) must be less than "
                f"n_landmarks ({self.n_landmarks})"
            )

        rng = np.random.RandomState(self.random_state)

        # Step 1: Select landmarks
        self.landmarks_high_ = self._select_landmarks(X, rng)

        # Step 2: Fit scaler on training data and scale landmarks
        self.scaler_ = StandardScaler().fit(X)
        self.landmarks_high_scaled_ = self.scaler_.transform(self.landmarks_high_)

        # Step 3: Create initial low-dimensional embedding using PCA
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        landmarks_low_raw = pca.fit_transform(self.landmarks_high_scaled_)

        # Step 4: Apply RMS scaling to match scales
        scaling_factor = self._compute_scaling_factor(
            self.landmarks_high_scaled_, landmarks_low_raw
        )
        self.landmarks_low_ = landmarks_low_raw * scaling_factor

        # Step 5: Precompute solver terms for efficient transformation
        self.reference_landmark_ = self.landmarks_low_[0]
        self.landmark_sq_norms_ = np.sum(self.landmarks_low_**2, axis=1)

        A = 2 * (self.landmarks_low_[1:] - self.reference_landmark_)
        self.solver_matrix_ = np.linalg.pinv(A)

        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """
        Transform X into the low-dimensional space.

        Args:
            X: Data to transform with shape (n_samples, n_features).

        Returns:
            Transformed data with shape (n_samples, n_components).

        Raises:
            NotFittedError: If the transformer has not been fitted yet.
        """
        check_is_fitted(
            self,
            ["landmarks_high_scaled_", "landmarks_low_", "solver_matrix_", "scaler_"],
        )
        X = check_array(X, dtype=np.float64)

        # 1. Scale input data
        X_scaled = self.scaler_.transform(X)

        # 2. Compute squared distances to landmarks in high-dimensional space
        diff = (
            X_scaled[:, np.newaxis, :] - self.landmarks_high_scaled_[np.newaxis, :, :]
        )
        distances_sq = np.sum(diff**2, axis=2)

        # 3. First pass: solve triangulation system
        term1 = self.landmark_sq_norms_[1:] - self.landmark_sq_norms_[0]
        term2 = distances_sq[:, 1:] - distances_sq[:, 0:1]
        b = term1 - term2
        Y_initial = b @ self.solver_matrix_.T

        # 4. Alpha refinement: compute scaling correction
        diff_low = Y_initial[:, np.newaxis, :] - self.landmarks_low_[np.newaxis, :, :]
        distances_low = np.linalg.norm(diff_low, axis=2)
        distances_high = np.sqrt(distances_sq)

        numerator = np.sum(distances_low * distances_high)
        denominator = np.sum(distances_sq)
        alpha = numerator / denominator if denominator > 1e-10 else 1.0

        # 5. Second pass: refined transformation
        distances_sq_corrected = (alpha**2) * distances_sq
        term2_corrected = distances_sq_corrected[:, 1:] - distances_sq_corrected[:, 0:1]
        b_corrected = term1 - term2_corrected
        Y_final = b_corrected @ self.solver_matrix_.T

        return Y_final
