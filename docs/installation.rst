Installation
============

From the repository root:

.. code-block:: bash

   python -m pip install -e .

Install the test-only dependencies, including pytest and SciPy, with:

.. code-block:: bash

   python -m pip install -e ".[test]"

SciPy is not a runtime solver dependency. It is currently used only by the
derivative regression tests.
