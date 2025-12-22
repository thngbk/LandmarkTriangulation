=========
Changelog
=========

Version 0.1.1 (2025-12-22)
==========================

Added
-----
* **Type Annotations**
* **Enhanced Documentation**: Comprehensive Google-style docstrings
* **Parameter Validation**: New validation checks in ``fit()`` method:
    - Ensures ``n_landmarks >= 2`` (minimum required for solver matrix)
    - Ensures ``n_components < n_landmarks`` (prevents singular matrix issues)
* **Code Modularization**: Extracted landmark selection logic into dedicated ``_select_landmarks()`` helper method

Changed
-------
* **PCA Pipeline Order**: Scaler is now fitted *before* PCA transformation
    - Previous: PCA applied to raw landmarks, then scaling applied
    - Current: Scaling applied first, then PCA on scaled landmarks
    - Impact: More consistent with transform pipeline behavior
* **Attribute Naming Convention**: Standardized to scikit-learn conventions with trailing underscores:
    - ``scaler`` → ``scaler_``
    - ``L0_low_`` → ``reference_landmark_``
    - ``L_low_sq_norms_`` → ``landmark_sq_norms_``
    - ``A_`` → ``A`` (converted to local variable)
* **Variable Naming Clarity**: Improved readability throughout:
    - ``delta_sq`` → ``distances_sq``
    - ``a`` → ``amplitudes``
    - ``omega`` → ``frequencies``
    - ``phi`` → ``phases``
    - ``gamma`` → ``landmarks``

Fixed
-----
* **Hybrid Mode Normalization Bug**: Synthetic landmarks are now properly transformed using the same scaler before nearest-neighbor distance computation
    - Previous: Raw synthetic landmarks compared to normalized data (incorrect scale)
    - Current: Both synthetic and real data normalized before comparison
    - Impact: Ensures fair Euclidean distance comparisons in hybrid mode
* **Fitted Attributes Validation**: Added ``scaler_`` to ``check_is_fitted()`` in ``transform()`` method
    - Prevents potential ``AttributeError`` if transform is called on unfitted estimator
