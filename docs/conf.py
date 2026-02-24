# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "ModularML"
copyright = "2025, The ModularML Team"
author = "The ModularML Team"
version = importlib.metadata.version("modularml")
release = version
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "restructuredtext",
}

# Mock heavy optional dependencies so autodoc can import all modules
autodoc_mock_imports = [
    "matplotlib",
    "torch",
    "tensorflow",
    "scikit-learn",
    "sklearn",
    "optuna",
    "pandas",
    "pyarrow",
    "numexpr",
    "bottleneck",
    "rich",
]

nb_render_markdown_format = "myst"
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_fence_as_directive = ["mermaid"]
nb_execution_timeout = 120
nb_execution_mode = "cache"
nb_execution_cache_path = str(ROOT / ".jupyter_cache")

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "special-members": "__init__",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_logo = "_static/logos/modularml_logo.png"
html_title = f"{project} v{version} Manual"
html_last_updated_fmt = "%Y-%m-%d"

add_module_names = False

html_theme_options = {
    "logo": {
        "image_light": "_static/logos/modularml_logo_text-dark.png",
        "image_dark": "_static/logos/modularml_logo_text-light.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/REIL-UConn/modular-ml",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ModularML/",
            "icon": "fa-solid fa-box",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": [
        "copyright",
        "sphinx-version",
    ],
    "footer_end": [
        "theme-version",
        "last-updated",
    ],
}


def _add_noindex_to_docstring_attributes(_app, _what, _name, _obj, _options, lines):
    """Prevent duplicate attribute targets when docstrings list dataclass fields."""
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith((".. attribute::", ".. py:attribute::")):
            indent = line[: len(line) - len(stripped)]
            next_idx = idx + 1
            while next_idx < len(lines) and lines[next_idx].strip() == "":
                next_idx += 1
            has_noindex = (next_idx < len(lines)) and (
                lines[next_idx].lstrip().startswith(":no-index:")
            )
            if not has_noindex:
                lines.insert(idx + 1, f"{indent}   :no-index:")
                idx += 1
        idx += 1


def setup(app):
    app.connect("autodoc-process-docstring", _add_noindex_to_docstring_attributes)
