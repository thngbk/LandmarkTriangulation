# Landmark Triangulation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/landmark-triangulation.svg)](https://pypi.org/project/landmark-triangulation/)
[![Tests](https://github.com/thngbk/LandmarkTriangulation/actions/workflows/test.yml/badge.svg)](https://github.com/thngbk/LandmarkTriangulation/actions/workflows/test.yml)

> **A deterministic, linear-time alternative to t-SNE for dimensionality reduction.**

**Landmark Triangulation** is a dimensionality reduction algorithm designed for speed, stability, and massive scalability. Unlike t-SNE or UMAP, which rely on iterative optimization and O(N²) pairwise comparisons, this method uses **landmark triangulation** against a topological skeleton to map points in **O(N·k)** linear time.

This approach makes it possible to generate embeddings for millions of points in seconds, without needing a GPU.

📖 **Read the Full Story:** [A Linear-Time Alternative To t-SNE for Dimensionality Reduction and Fast Visualisation](https://medium.com/towards-artificial-intelligence/a-linear-time-alternative-to-t-sne-for-dimensionality-reduction-and-fast-visualisation-5cd1a7219d6f)

---

## ⚡ Benchmarks

Comparison against Scikit-Learn's t-SNE on a synthetic dataset of 2,000 samples with 5 clusters (50 features):

| Method          | Time (sec) | Speedup | Silhouette Score |
|-----------------|-----------:|--------:|-----------------:|
| Random Mode     |      0.25s |     84x |             0.81 |
| Synthetic Mode  |      0.25s |     84x |             0.33 |
| Hybrid Mode     |      0.21s |    100x |             0.61 |
| t-SNE           |     21.16s |      1x |             0.84 |

**Key Takeaways:**

- 🚀 **~85× faster** than t-SNE on this dataset
- 🎯 **96% of t-SNE's clustering quality** (0.81 vs 0.84 score) in a fraction of the time

![Benchmark Visualization](./resources/images/benchmark.png)

📊 Reproduce this benchmark with the notebook in the `examples/` folder.

---

## 🚀 Key Features

- **⚡ Linear Time Complexity O(N·k)**: Scales linearly with dataset size
- **🎯 Deterministic & Stable**: No random initialization that changes results between runs
- **🔧 Scikit-learn Compatible**: Drop-in replacement for TSNE/UMAP in existing pipelines
- **📐 Alpha Refinement**: Global stress-correction to minimize distortion
- **👻 Ghost Manifolds (Hybrid Mode)**: Novel "Manifold Snapping" technique that fits a sine-wave skeleton to your data distribution

---

## 🛠 Prerequisites & Installation

### 1. System Requirements

- **Python**: 3.9 or higher.
- **Windows Users**: You must have the **Visual Studio Build Tools** installed to compile dependencies like `numpy`. 
    - [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **"Desktop development with C++"** workload.

### 2. Installation

**From PyPI (recommended):**

```bash
pip install landmark-triangulation
```

**From source:**

We recommend using [uv](https://docs.astral.sh/uv/) for development:

```bash
# Clone the repository
git clone https://github.com/thngbk/LandmarkTriangulation.git
cd LandmarkTriangulation

# Synchronize the environment (creates .venv and installs everything)
uv sync
```

**For development (editable mode with all tools):**

```bash
# For development (includes testing tools, linter, and examples):
uv sync --dev --extra examples
```

### 3. Verify Installation

Once installed, verify that the library and its dependencies are communicating correctly with your system architecture:

**Standard Install (pip):**
```bash
# Run the built-in diagnostic command
lt-check

# OR using the python module directly
python -m landmark_triangulation.verify
```

**Development Install (uv):**

```bash
# Run via uv to ensure the local .venv is used
uv run lt-check

# OR
uv run python -m landmark_triangulation.verify
```

### 4. Dependencies

- NumPy ≥ 1.20.0
- Scikit-learn ≥ 1.0.0

---

## 💻 Quick Start

### Basic Usage

Landmark Triangulation provides a scikit-learn-compatible transformer:

```python
import numpy as np
from landmark_triangulation import LandmarkTriangulation

# Generate sample high-dimensional data
X = np.random.rand(10000, 50)  # 10k samples, 50 features

# Initialize transformer (hybrid mode recommended)
transformer = LandmarkTriangulation(
    n_components=2, 
    n_landmarks=150, 
    landmark_mode='hybrid',
    random_state=42
)

# Fit and transform in linear time
embedding = transformer.fit_transform(X)

print(f"Embedding shape: {embedding.shape}")  
# Output: (10000, 2)
```

### Scikit-learn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from landmark_triangulation import LandmarkTriangulation

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', LandmarkTriangulation(n_components=2, n_landmarks=100))
])

X_embedded = pipeline.fit_transform(X)
```

### Visualization Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# Generate Swiss roll dataset
X, color = make_swiss_roll(n_samples=2000, noise=0.1, random_state=42)

# Apply Landmark Triangulation
lt = LandmarkTriangulation(n_components=2, n_landmarks=100, landmark_mode='hybrid')
X_embedded = lt.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 2], c=color, cmap='viridis', s=10)
plt.title('Original Swiss Roll')
plt.xlabel('X')
plt.ylabel('Z')

plt.subplot(122)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap='viridis', s=10)
plt.title('Landmark Triangulation Embedding')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.tight_layout()
plt.show()
```

---

## 🧠 How It Works

This algorithm acts like a **GPS system** rather than preserving all pairwise distances:

1. **Landmark Selection**: Select k "satellite" points (landmarks) from your data
2. **Skeleton Discovery**: Use PCA on landmarks to determine global structure
3. **Triangulation**: For each point, measure distances to landmarks and solve a linear system to find 2D coordinates
4. **Alpha Correction**: Calculate global error factor α and re-run triangulation to minimize distortion

```mermaid
graph TD
    subgraph Step 1: Landmark Selection
    A[Input Data X] -->|Mode: Hybrid| B(Generate Sine Ghosts)
    B --> C{Snap to Data?}
    C -->|Yes| D[Select Nearest Real Points]
    end

    subgraph Step 2: Skeleton Discovery
    D --> E[PCA on Landmarks]
    E --> F[Low-D Skeleton L']
    end

    subgraph Step 3: Triangulation
    A --> G{Measure Distances}
    D --> G
    G -->|Distance to Landmarks| H[Solve Linear System Ax=b]
    H --> I[Raw Embedding Y]
    end

    subgraph Step 4: Alpha Refinement
    I --> J[Compare High-D vs Low-D Distances]
    J --> K[Compute Alpha α]
    K --> L[Re-Triangulate]
    L --> M((Final Embedding))
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#bbf,stroke:#333,stroke-width:2px
```

### Landmark Selection Strategies

| Mode            | Description                                                                          | Best For                           | Performance        |
|-----------------|--------------------------------------------------------------------------------------|------------------------------------|--------------------|
| **`random`**    | Randomly selects k points from your dataset                                          | General purpose, dense clusters    | ⭐⭐⭐⭐⭐ (Best)  |
| **`synthetic`** | Generates a perfect sine-wave path through phase space                               | Visualizing theoretical manifolds  | ⭐⭐⭐             |
| **`hybrid`**    | **(Recommended)** Generates sine-wave "ghosts" and snaps them to nearest real points | Preserving topology with real data | ⭐⭐⭐⭐           |

```python
# Recommended: Hybrid mode for most use cases
transformer = LandmarkTriangulation(
    n_components=2,
    n_landmarks=100,        # More landmarks = better accuracy but slower
    landmark_mode='hybrid',
    random_state=42         # For reproducibility
)
```

---

## ⚠️ Common Errors

### 🪟 Windows: C++ Build Tools Missing

If you see an error like `error: Microsoft Visual C++ 14.0 or greater is required` during installation, it means the `numpy` build failed because your system lacks a C++ compiler.

**The Fix:**

1. Download the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Run the installer and select **"Desktop development with C++"**.
3. Restart your terminal and run `uv sync` again.

---

## 📂 Repository Structure

```text
landmark-triangulation/
├── src/
│   └── landmark_triangulation/
│       ├── __init__.py         # Package exports
│       ├── _version.py         # Dynamic version info
│       ├── core.py             # Main implementation
│       ├── py.typed            # Marker for PEP 561 (Type Hinting)
│       └── verify.py           # Diagnostic & installation validation
├── tests/                      # Unit & Regression tests
│   ├── fixtures/
│   │   ├── golden/             # Golden outputs for regression tests
│   │   └── generate_golden_outputs.py  # Script to regenerate baselines
│   ├── conftest.py              # Shared fixtures
│   ├── test_regression.py       # Regression tests (run first in CI)
│   ├── test_core.py             # Core functionality tests
│   ├── test_sklearn_compatibility.py  # Scikit-learn API tests
│   ├── test_numerical_stability.py    # Numerical correctness
│   └── test_performance.py      # Performance tests (marked @slow)
├── examples/                   # Jupyter notebooks
│   └── tsne_benchmark.ipynb
├── resources/
│   └── images/                 # Benchmark plots
├── pyproject.toml              # Build configuration
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## 🧪 Development & Testing

### 🛠️ Environment Setup

To ensure all development tools and hooks are correctly configured, run the following:

```bash
# 1. Sync the environment (installs dev tools and optional examples)
uv sync --dev --extra examples

# 2. Install the git pre-commit hooks
uv run pre-commit install
```

### 🧪 Testing

This project uses `pytest` for testing with high code coverage standards.

**Current Coverage: 94%** (48 tests)

```bash
# Run all tests (excludes slow performance tests by default)
uv run python -m pytest tests/

# Run with coverage report
uv run python -m pytest tests/ --cov=landmark_triangulation --cov-report=term-missing

# Run ALL tests including slow performance tests
uv run python -m pytest tests/ -m ""

# Run only fast tests explicitly
uv run python -m pytest tests/ -m "not slow"

# Run specific test file
uv run python -m pytest tests/test_core.py
```

#### Regression Tests

Regression tests ensure code changes don't break expected algorithm behavior. These tests compare current outputs against golden baselines from commit `2e0552a`:

```bash
# Run regression tests only
uv run python -m pytest tests/test_regression.py -v
```

**What's tested:** All landmark modes (random, synthetic, hybrid) in 2D and 3D using fixed synthetic data (500 samples × 20 features, random seed 42).

**Golden outputs:** Stored in `tests/fixtures/golden/` as `.npy` files. See `tests/fixtures/generate_golden_outputs.py` for how test data and baselines are created.

These tests run **first** in the CI pipeline and must pass before other tests execute.

See [tests/TESTING.md](tests/TESTING.md) for detailed testing documentation.

### Test Coverage

View the HTML coverage report:

```bash
# Generate report
uv run python -m pytest tests/ --cov=landmark_triangulation --cov-report=html

# Open in browser (Linux/Mac)
open htmlcov/index.html
```

### ✨ Code Quality (Pre-commit)

This project uses [pre-commit](https://pre-commit.com/) with [ruff](https://docs.astral.sh/ruff/) to automate code quality. Hooks run automatically before each commit; if issues are found, the commit is blocked until fixed.

**Manual execution:**

```bash
# Run hooks manually on all files
uv run pre-commit run --all-files
```

**What gets checked:**

- **ruff check** (`--fix`): Lints and auto-fixes code issues
  - `E`, `W`: PEP 8 errors and warnings
  - `F`: Pyflakes (unused imports, undefined names)
  - `I`: Import sorting (isort-compatible)
  - `B`: Bugbear (common bugs and design problems)
  - `UP`: pyupgrade (modern Python syntax)
  - `PYI`: Type hint checks
  - `TID`: Tidy imports
- **ruff format**: Ensures consistent code formatting (Black-compatible)

### 🚀 Continuous Integration (CI)

All pushes and pull requests are automatically tested via **GitHub Actions** on **Ubuntu** to ensure code integrity and maintain quality standards.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`uv run python -m pytest tests/`)
4. Format code (`uv run ruff format && uv run ruff check --fix`)
5. Commit changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

---

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@misc{ferrando2025slr_article,
  author       = {Ferrando-Llopis, Roman},
  title        = {A Linear-Time Alternative To t-SNE for Dimensionality Reduction and Fast Visualisation},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18007950},
  url          = {https://doi.org/10.5281/zenodo.18007950}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📮 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/thngbk/LandmarkTriangulation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/thngbk/LandmarkTriangulation/discussions)
- **Article**: [Medium Post](https://medium.com/towards-artificial-intelligence/a-linear-time-alternative-to-t-sne-for-dimensionality-reduction-and-fast-visualisation-5cd1a7219d6f)
