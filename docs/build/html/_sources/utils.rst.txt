Utils
=====

The :mod:`deckard.utils` module contains shared utilities used across the
public API, including stable config hashing, serialization helpers, dynamic
class loading, and parser generation helpers.

.. automodule:: deckard.utils
   :members:
   :show-inheritance:

Overview
--------

Utilities provide the shared primitives that keep Deckard configs and runtime
behavior deterministic across CLI and programmatic execution.

Key responsibilities include:

- stable hashing for config identity
- safe object/data serialization helpers
- dynamic class loading from import paths
- parser creation from callable signatures

Usage
-----

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard.utils import hash_conf_values, create_parser_from_function

   conf_hash = hash_conf_values({"a": 1, "b": [2, 3]})
   print(conf_hash)

   def fn(x: int, y: str = "demo"):
       return x, y

   parser = create_parser_from_function(fn)
   print(parser)

Internals
---------

The module emphasizes deterministic normalization (for hashing and persistence)
and defensive loading behavior so configs are portable across environments.

Troubleshooting
---------------

- Verify dotted import paths when using dynamic class loading helpers.
- Ensure serialized object/data formats match file extension and expected loader.
- Check hash normalization inputs when comparing run identity across platforms.

See also
~~~~~~~~

* :doc:`experiment`
* :doc:`file`
* :doc:`layers`
