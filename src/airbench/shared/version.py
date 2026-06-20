"""Single source of truth for the package version (R5).

The literal lives here once. ``__init__`` imports ``VERSION``; the build backend
(hatchling) reads ``__version__``; a test asserts they agree. Bump ``+0.01`` per change.
"""

__version__ = "1.10"
VERSION = __version__
