Rocky Worlds DDT Data Challenge
================================

.. image:: https://readthedocs.org/projects/rocky-worlds-data-challenge/badge/?version=latest
    :target: https://rocky-worlds-data-challenge.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/spacetelescope/rocky-worlds-data-challenge/actions/workflows/ci_workflows.yml/badge.svg
    :target: https://github.com/spacetelescope/rocky-worlds-data-challenge/actions/workflows/ci_workflows.yml
    :alt: GitHub Actions CI Status

.. image:: https://codecov.io/gh/spacetelescope/rocky-worlds-data-challenge/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/spacetelescope/rocky-worlds-data-challenge
    :alt: Codecov Coverage

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

Utilities and notebooks for preparing submissions to the
`Rocky Worlds DDT Data Challenge <https://www.kaggle.com/competitions/rocky-worlds-data-challenge>`_.

This package provides helpers for:

* describing eclipsing exoplanet systems and derived orbital quantities
* validating posterior samples, light curves, and questionnaire responses
* writing and reloading challenge submission ZIP archives
* interactively completing the submission form


Installation
------------

For standard use, install the package from PyPI:

.. code-block:: bash

    python -m pip install rocky-worlds-data-challenge

To validate an installed wheel with the package's bundled test suite, install
the optional test dependencies and run the installed tests:

.. code-block:: bash

    python -m pip install "rocky-worlds-data-challenge[test]"
    python -m pytest --pyargs rocky_worlds_data_challenge.tests

Local Development
~~~~~~~~~~~~~~~~~

To work on the package locally, clone the repository and install it in editable
mode with development, documentation, and test dependencies:

.. code-block:: bash

    git clone https://github.com/spacetelescope/rocky-worlds-data-challenge.git
    cd rocky-worlds-data-challenge
    python -m pip install -e ".[dev,docs,test]"


Quick Start
-----------

The best place to start is the tutorial in the hosted documentation:

* `Prepare Kaggle submissions <https://rocky-worlds-data-challenge.readthedocs.io/en/latest/tutorials/prepare-submission.html>`_

That tutorial walks through packaging posterior samples, photometry products,
completed forms, and the final submission ZIP archive.


Documentation
-------------

Documentation is built with Sphinx and published on Read the Docs.

To build the docs locally:

.. code-block:: bash

    python -m sphinx -W -b html docs docs/_build/html


Testing
-------

Run the test suite with:

.. code-block:: bash

    python -m pytest

The CI configuration also runs style checks, security checks, package build
checks, and documentation link checks through ``tox``.


Simulation Code
---------------

The ``rocky-worlds-data-challenge/simulator`` directory is reserved for the code used to generate the
simulated data products. The intention is to provide that code once the data
challenge is complete.


License
-------

See ``LICENSE.rst`` for more information.


Contributing
------------

Contributions are welcome. See ``CONTRIBUTING.md`` for guidance.
