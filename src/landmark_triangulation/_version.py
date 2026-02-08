"""Version information."""

from importlib.metadata import PackageNotFoundError, version

# The source of truth for the fallback/development version
__version__ = "1.1.0"

try:
    # Attempt to get the version from the installed package metadata
    __version__ = version("landmark-triangulation")
except PackageNotFoundError:
    # Package is not installed (e.g., running tests directly from source)
    pass
