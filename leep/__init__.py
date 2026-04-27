from __future__ import annotations

from .base import open, IGNORE, WARN, ERROR, RomError
from ._version import __version__ as _version

__all__ = [
    '__version__',
    'open',
    'IGNORE',
    'WARN',
    'ERROR',
    'RomError',
    '__version__'
]

__version__: str = _version
