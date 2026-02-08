"""Version information."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("landmark-triangulation")
except PackageNotFoundError:
    # Fallback for development/editable installs
    __version__ = "1.1.0"
