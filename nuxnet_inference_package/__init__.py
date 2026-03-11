"""Top-level package for nuxnet-inference."""

from importlib.metadata import PackageNotFoundError, version

__author__ = "NuxNet contributors"
__email__ = "noreply@example.org"

try:
    __version__ = version("nuxnet-inference")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
