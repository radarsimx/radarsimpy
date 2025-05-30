"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(".."))

import radarsimpy  # pylint: disable=wrong-import-position

# -- Project information -----------------------------------------------------

project = "RadarSimPy"  # pylint: disable=invalid-name
copyright = (  # pylint: disable=redefined-builtin, invalid-name
    "2018 - " + str(datetime.datetime.now().year) + ", RadarSimX LLC"
)
version = radarsimpy.__version__  # pylint: disable=invalid-name
release = version  # pylint: disable=invalid-name

pygments_style = "sphinx"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",  # Add doctest extension
]

autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_favicon = "_static/radarsimdoc.png"  # pylint: disable=invalid-name

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"  # pylint: disable=invalid-name

html_theme_options = {
    "logo": {
        "text": "RadarSimPy v" + version,
        "image_light": "_static/radarsimdoc.svg",
        "image_dark": "_static/radarsimdoc.svg",
    },
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 8,
    "icon_links": [
        {
            "name": "RadarSimX",
            "url": "https://radarsimx.com/",
            "icon": "fas fa-globe",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/radarsimx/radarsimpy",
            "icon": "fab fa-github",
        },
    ],
    "footer_start": ["copyright"],
    "footer_center ": ["sphinx-version"],
    "footer_end": ["theme-version"],
}

html_context = {
    "default_mode": "auto",
    "display_github": True,
    "github_user": "radarsimx",
    "github_repo": "radarsimpy",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
