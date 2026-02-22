# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
]
source_suffix = {
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# Mock heavy optional dependencies so autodoc can import all modules
autodoc_mock_imports = ["matplotlib", "torch", "tensorflow", "scikit-learn"]

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

# -- nbsphinx config ---------------------------------------------------------
nbsphinx_execute = "never"  # do not execute notebooks during doc build
nbsphinx_requirejs_path = ""  # prevent conflicts with require.js

# Optional: avoid Pandoc by keeping Markdown cells as raw HTML
nbsphinx_allow_errors = True  # (not strictly required, but handy for CI)
