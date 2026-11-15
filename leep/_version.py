from __future__ import annotations

try:
    import importlib.metadata as metadata
except ImportError:
    # Fallback for Python < 3.8
    import importlib_metadata as metadata

__all__ = ("__version__",)

try:
    __version__ = metadata.version("leep")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
