# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Shared scverse defaults (MyST, napoleon, autosummary, theme, ...) come from scverse-doc:
# https://github.com/scverse/scverse-doc

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

info = metadata("scib-metrics")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "scverse_doc",
    # scverse-doc resolves the [source] links from `source_repository`
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

bibtex_reference_style = "author_year"
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"

# scverse-doc's registry already maps anndata, matplotlib, numpy, pandas, python, scanpy,
# scipy and the other scverse packages, so only the rest is listed here.
intersphinx_mapping = intersphinx() | {
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "plottable": ("https://plottable.readthedocs.io/en/latest/", None),
}

# extlinks config
extlinks = {
    "issue": (f"{source_repository}/issues/%s", "#%s"),
    "pr": (f"{source_repository}/pull/%s", "#%s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]
html_css_files = ["css/custom.css", "css/override.css"]
html_js_files = ["js/custom.js"]

html_theme_options = {
    # The scverse theme's header, plus the download/fullscreen buttons that
    # sphinx-book-theme used to provide (see _templates/header-buttons.html).
    "article_header_end": [
        "header-buttons",
        "theme-switcher",
        "navbar-icon-links",
        "scverse-ecosystem",
        "toggle-secondary-sidebar",
    ],
}
# Copy sources under their own suffix, so the download button serves e.g. a usable .ipynb.
html_sourcelink_suffix = ""

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
]
