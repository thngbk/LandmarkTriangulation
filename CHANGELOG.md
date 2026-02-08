# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Releasing a New Version

1. Update version in `pyproject.toml`
2. Update this CHANGELOG with release date and changes
3. Commit changes: `git commit -am "Release v1.X.X"`
4. Create git tag: `git tag v1.X.X`
5. Push: `git push && git push --tags`

---

## [1.1.0] - 2026-02-08

### ⚠️ Breaking Changes
- **Minimum Python version raised to 3.9**: Required for modern type hint syntax (`int | None`) and dependency updates.
- **Build Backend Migration**: Switched from `setuptools` to `hatchling` for PEP 517 compliance.

### Added
- **Built-in Diagnostic Tool**: Added `landmark_triangulation.verify` module and `lt-check` console script for system verification.
- **Regression tests**: Golden output tests to prevent algorithm changes from breaking expected behavior.
  - Test fixtures generated from baseline commit (`2e0552a`).
  - Covers all landmark modes (random, synthetic, hybrid) in 2D and 3D.
- **Pre-commit integration**: Added `ruff` for automated linting and formatting.
- **CI/CD Pipeline**: GitHub Actions for automated testing, linting, and PyPI publishing.
- **Test Documentation**: Created `tests/TESTING.md` with instructions for 100% coverage suite.
- **Windows Support**: Added "Common Errors" section to README regarding C++ build tools.

### Changed
- **CLI Entry Points**: Configured `project.scripts` in `pyproject.toml` to expose verification utilities.
- **Dependency Management**: Moved dev dependencies to `[dependency-groups]` (modern `uv` standard).
- **Project Structure**: Renamed main notebook to `tsne_benchmark.ipynb` and reorganized README for better onboarding.
- **Typing**: Modernized all type hints to use PEP 604 syntax (`from __future__ import annotations`).
- **Version Logic**: `_version.py` now dynamically reads from package metadata.

### Fixed
- **Notebook Reliability**: Removed fragile `sys.path` hacks in favor of proper installation.
- **Standards**: Added missing paper citations and fixed `pytest` strict marker warnings.

## [1.0.1] - 2025-12-22

### Added
- **Type annotations** in `src/landmark_triangulation/core.py`.
- **Enhanced documentation**: Comprehensive Google-style docstrings in `src/landmark_triangulation/core.py`.
- **Parameter validation**: Added validation checks in `LandmarkTriangulation.fit()`:
  - Ensures `n_landmarks >= 2` (minimum required for solver matrix).
  - Ensures `n_components < n_landmarks` (prevents singular-matrix issues).
- **Code modularization**: Extracted landmark-selection logic from `LandmarkTriangulation` into a dedicated `_select_landmarks()` helper method.
- Added extensive tests under the `tests/` directory.
- Added `CHANGELOG.rst` to track changes.

### Changed
- **Attribute naming conventions** updated to follow scikit-learn's trailing-underscore pattern:
  - `scaler` → `scaler_`
  - `L0_low_` → `reference_landmark_`
  - `L_low_sq_norms_` → `landmark_sq_norms_`
  - `A_` → (removed as an attribute; now a local variable `A`)
- **Improved variable clarity** throughout `LandmarkTriangulation`:
  - `delta_sq` → `distances_sq`
  - `a` → `amplitudes`
  - `omega` → `frequencies`
  - `phi` → `phases`
  - `gamma` → `landmarks`
- Updated `pyproject.toml` with additional `keywords`, `classifiers`, Ruff configuration, and clarified dependencies.
- Removed `requirements.txt` (no longer needed).

### Fixed
- **Fitted attribute validation**: Added `scaler_` to `check_is_fitted()` inside `transform()`.
