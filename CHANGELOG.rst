
Changelog
=========

Version 1.0.1 (2025-12-22)
==========================

Added
-----
* **Type annotations** in ``src/landmark_triangulation/core.py``.
* **Enhanced documentation**: Comprehensive Google‑style docstrings in ``src/landmark_triangulation/core.py``.
* **Parameter validation**: Added validation checks in ``LandmarkTriangulation.fit()``:
  - Ensures ``n_landmarks >= 2`` (minimum required for solver matrix).
  - Ensures ``n_components < n_landmarks`` (prevents singular‑matrix issues).
* **Code modularization**: Extracted landmark‑selection logic from ``LandmarkTriangulation`` into a dedicated ``_select_landmarks()`` helper method.
* Added extensive tests under the ``tests/`` directory.
* Added ``CHANGELOG.rst`` to track changes.

Changed
-------
* **Attribute naming conventions** updated to follow scikit‑learn’s trailing‑underscore pattern:
  - ``scaler`` → ``scaler_``
  - ``L0_low_`` → ``reference_landmark_``
  - ``L_low_sq_norms_`` → ``landmark_sq_norms_``
  - ``A_`` → (removed as an attribute; now a local variable ``A``)
* **Improved variable clarity** throughout ``LandmarkTriangulation``:
  - ``delta_sq`` → ``distances_sq``
  - ``a`` → ``amplitudes``
  - ``omega`` → ``frequencies``
  - ``phi`` → ``phases``
  - ``gamma`` → ``landmarks``
* Updated ``pyproject.toml`` with additional ``keywords``, ``classifiers``, Ruff configuration, and clarified dependencies.
* Removed ``requirements.txt`` (no longer needed).

Fixed
-----
* **Fitted attribute validation**: Added ``scaler_`` to ``check_is_fitted()`` inside ``transform()``,

