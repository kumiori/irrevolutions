# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Irreversible Solvers'
copyright = '2024, Andrés A León Baldelli'
author = 'Andrés A León Baldelli'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser",
              "sphinx.ext.duration",
              "sphinx.ext.autodoc",
              "autoapi.extension",
              "sphinx.ext.napoleon",
              ]

autoapi_type = 'python'
autoapi_dirs = ['../../']

templates_path = ['_templates']
exclude_patterns = [
                    ]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
