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

## [1.0.2] - 2025-02-08

### Added
- **Regression tests**: Golden output tests to prevent algorithm changes from breaking expected behavior
  - Test fixtures generated from baseline commit (2e0552a)
  - Covers all landmark modes (random, synthetic, hybrid) in 2D and 3D
  - Runs first in CI pipeline as a gate before other tests
  - Generation script included for updating baselines when algorithm is intentionally changed
- Pre-commit hooks with ruff for automated code quality
- GitHub Actions CI/CD for testing, linting, and PyPI publishing
- Test documentation (`tests/TESTING.md`) with 100% coverage (44 tests)
- 90% minimum coverage threshold enforcement
- Enhanced notebook with markdown explanations and error handling
- Common Errors section in README for Windows C++ build tools

### Changed
- **BREAKING**: Minimum Python version raised from 3.8 to 3.9
- Build backend migrated from `setuptools` to `hatchling`
- Dev dependencies moved to `[dependency-groups]` (modern uv standard)
- `jupyterlab` moved from core to optional `[examples]` dependencies
- README reorganized (Development & Testing before Contributing)
- README installation prioritizes PyPI over source
- Notebook renamed to `tsne_benchmark.ipynb` (Python convention)
- Type hints modernized (`int | None` syntax)
- CHANGELOG format converted from `.rst` to `.md`
- Version handling: `_version.py` now reads from package metadata automatically

### Fixed
- Pytest configuration: added `pythonpath` and `--strict-markers`
- Docstring reference: added actual paper citation
- Notebook imports: removed fragile sys.path manipulation

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
