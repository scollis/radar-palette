"""Sphinx configuration for the radar-palette documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

# autosummary imports the package to introspect it. Prefer an installed
# radar-palette; fall back to the adjacent src/ tree so `make html` also works
# in a bare clone where `pip install -e .` has not been run.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

project = "radar-palette"
copyright = "2026, Radar-Palette Developers"  # noqa: A001
author = "Radar-Palette Developers"

try:
    release = _pkg_version("radar-palette")
except PackageNotFoundError:  # docs built from a source tree without an install
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_nb",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pyart": ("https://arm-doe.github.io/pyart", None),
}

# Fetching intersphinx inventories needs network access, and the Makefile builds
# with -W. Set RADAR_PALETTE_DOCS_OFFLINE=1 to drop cross-project links so an
# offline build stays warning-clean.
if os.environ.get("RADAR_PALETTE_DOCS_OFFLINE"):
    intersphinx_mapping = {}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/DIGR-Legacy/radar-palette",
}

nb_execution_mode = "off"
