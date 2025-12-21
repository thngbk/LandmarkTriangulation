import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import pairwise_distances_argmin

class LandmarkTriangulation(BaseEstimator, TransformerMixin):
    """
    Landmark-based Dimensionality Reduction using Triangulation and Alpha-Refinement.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the output embedding.
    
    n_landmarks : int, default=50
        Number of landmark points to sample (k).
    
    landmark_mode : {'random', 'synthetic', 'hybrid'}, default='random'
        The strategy for selecting landmarks:
        - 'random': Randomly samples k points from the input data.
        - 'synthetic': Generates k synthetic points based on a sine-wave manifold 
          (ignores input data distribution for the skeleton).
        - 'hybrid': Generates synthetic sine-wave points and 'snaps' them to the 
          nearest actual data point. This ensures landmarks exist on the real 
          manifold while maintaining a topological structure.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the landmark selection/generation.
    """

    def __init__(self, n_components=2, n_landmarks=50, landmark_mode='random', random_state=42):
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.landmark_mode = landmark_mode
        self.random_state = random_state
        self.scaler = StandardScaler()

    def _generate_synthetic_landmarks(self, n_features):
        """Generates synthetic sine-based landmarks."""
        rng = np.random.RandomState(self.random_state)
        a = rng.uniform(0.5, 2.0, n_features)
        omega = rng.uniform(0.5, 1.5, n_features)
        phi = rng.uniform(0, 2 * np.pi, n_features)
        t = np.linspace(0, 2 * np.pi, self.n_landmarks)
        
        # result shape: (n_dims, n_landmarks) -> Transpose to (n_landmarks, n_dims)
        gamma = a[:, None] * np.sin(omega[:, None] * t + phi[:, None])
        return gamma.T

    def fit(self, X, y=None):
        """
        Selects landmarks and computes the embedding skeleton.
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # --- STEP 1: Select Landmarks based on Mode ---
        if self.landmark_mode == 'synthetic':
            # Pure hallucination (Eq 1 in original logic)
            self.landmarks_high_ = self._generate_synthetic_landmarks(n_features)
            
        elif self.landmark_mode == 'hybrid':
            # Generate ghosts
            ghosts = self._generate_synthetic_landmarks(n_features)
            
            # Snap ghosts to nearest real data points (Manifold Snapping)
            # We must normalize temporarily to ensure Euclidean distance is fair
            temp_scaler = StandardScaler().fit(X)
            X_temp = temp_scaler.transform(X)
            
            # Assuming ghosts are roughly in standard normal range, or we map them.
            # Since _gamma generates values roughly [-2, 2], they align well with standardized X.
            # We calculate indices of the nearest real neighbor for every ghost.
            closest_indices = pairwise_distances_argmin(ghosts, X_temp)
            
            # Select the REAL points
            self.landmarks_high_ = X[closest_indices].copy()
            
        else: # 'random' (Default)
            if n_samples < self.n_landmarks:
                raise ValueError(f"n_samples ({n_samples}) must be >= n_landmarks ({self.n_landmarks})")
            
            idx = rng.choice(n_samples, size=self.n_landmarks, replace=False)
            self.landmarks_high_ = X[idx].copy()

        # --- STEP 2: PCA Skeleton on RAW Data ---
        # Logic: We fit PCA on raw data to capture magnitude-based variance (The "Old" Look)
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        landmarks_low_raw = pca.fit_transform(self.landmarks_high_)

        # --- STEP 3: Standardize Everything ---
        # Now we fit the main scaler on X and transform the landmarks for the Solver
        self.scaler.fit(X)
        self.landmarks_high_scaled_ = self.scaler.transform(self.landmarks_high_)

        # --- STEP 4: RMS Scaling (Global Scale Correction) ---
        def get_sum_sq_dist(arr):
            diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
            return np.sum(diff ** 2)

        ss_high = get_sum_sq_dist(self.landmarks_high_scaled_)
        ss_low = get_sum_sq_dist(landmarks_low_raw)
        
        scaling_factor = np.sqrt(ss_high / ss_low) if ss_low > 1e-10 else 1.0
        self.landmarks_low_ = landmarks_low_raw * scaling_factor

        # --- STEP 5: Precompute Linear Solver Terms ---
        self.L0_low_ = self.landmarks_low_[0]
        self.L_low_sq_norms_ = np.sum(self.landmarks_low_**2, axis=1)
        
        self.A_ = 2 * (self.landmarks_low_[1:] - self.L0_low_)
        self.solver_matrix_ = np.linalg.pinv(self.A_)

        return self

    def transform(self, X):
        """
        Projects X into the low-dimensional space.
        """
        check_is_fitted(self, ['landmarks_high_scaled_', 'landmarks_low_', 'solver_matrix_'])
        X = check_array(X)
        
        # 1. Apply the scaler fitted during fit()
        X_scaled = self.scaler.transform(X)

        # 2. Calculate squared distances from X to all High-D Landmarks
        diff = X_scaled[:, np.newaxis, :] - self.landmarks_high_scaled_[np.newaxis, :, :]
        delta_sq = np.sum(diff**2, axis=2)

        # 3. Construct RHS vector b (Pass 1)
        term1 = self.L_low_sq_norms_[1:] - self.L_low_sq_norms_[0]
        term2 = delta_sq[:, 1:] - delta_sq[:, 0:1]
        b = term1 - term2 

        # 4. Solve for Raw Coordinates
        Y_raw = b @ self.solver_matrix_.T

        # 5. Alpha Refinement
        diff_low = Y_raw[:, np.newaxis, :] - self.landmarks_low_[np.newaxis, :, :]
        dist_low = np.linalg.norm(diff_low, axis=2) 
        dist_high = np.sqrt(delta_sq)

        numerator = np.sum(dist_low * dist_high)
        denominator = np.sum(delta_sq)
        
        alpha = numerator / denominator if denominator > 1e-10 else 1.0

        # 6. Final Mapping (Pass 2)
        delta_sq_corrected = (alpha**2) * delta_sq
        term2_corr = delta_sq_corrected[:, 1:] - delta_sq_corrected[:, 0:1]
        b_corr = term1 - term2_corr
        
        Y_final = b_corr @ self.solver_matrix_.T
        
        return Y_final