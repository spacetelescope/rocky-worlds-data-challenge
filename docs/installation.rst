Installation
============

Standard Installation
---------------------

For standard use, install the package from PyPI:

.. code-block:: bash

    python -m pip install rocky-worlds-data-challenge

This is the recommended installation path for participants who only need to
prepare and validate challenge submissions.

Local Development
-----------------

To work on the package locally, clone the repository and install it in editable
mode with development, documentation, and test dependencies:

.. code-block:: bash

    git clone https://github.com/spacetelescope/rocky-worlds-data-challenge.git
    cd rocky-worlds-data-challenge
    python -m pip install -e ".[dev,docs,test]"

Build The Documentation
-----------------------

To build the documentation locally:

.. code-block:: bash

    python -m sphinx -W -b html docs docs/_build/html

Run The Tests
-------------

From a local source checkout, run the test suite with:

.. code-block:: bash

    python -m pytest

To validate an installed package instead of a local checkout, install the
optional test dependencies and run the bundled installed tests with
``--pyargs``:

.. code-block:: bash

    python -m pip install "rocky-worlds-data-challenge[test]"
    python -m pytest --pyargs rocky_worlds_data_challenge.tests
