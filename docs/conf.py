"""Sphinx configuration for the JAX BSSN documentation."""

from pathlib import Path
import sys


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

project = "JAX BSSN"
author = "Christopher Woolford"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

autodoc_mock_imports = [
    "openpmd_api",
]
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "JAX BSSN"
html_static_path = ["images"]
html_css_files = ["custom.css"]

logo_path = DOCS_DIR / "images" / "JAX_BSSN_logo.png"
if logo_path.exists():
    html_logo = "images/JAX_BSSN_logo.png"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0757d9",
        "color-brand-content": "#0757d9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#72a7ff",
        "color-brand-content": "#72a7ff",
    },
}
