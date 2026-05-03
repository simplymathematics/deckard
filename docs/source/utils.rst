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
- torch device resolution helpers for cpu/cuda/mps selection
- ConfigStore-safe registration helpers for Hydra config groups

Usage
-----

Programmatic example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from deckard.utils import (
      hash_conf_values,
      create_parser_from_function,
      resolve_torch_device,
      safe_store,
   )

   conf_hash = hash_conf_values({"a": 1, "b": [2, 3]})
   print(conf_hash)

   def fn(x: int, y: str = "demo"):
       return x, y

   parser = create_parser_from_function(fn)
   print(parser)

   device = resolve_torch_device("auto")
   print(device)

   # Safe duplicate-tolerant ConfigStore registration
   safe_store(group="score", name="my-score", node={"scorers": {}})

ConfigBase Helpers
~~~~~~~~~~~~~~~~~~

Most major config classes inherit from :class:`deckard.utils.ConfigBase`, which
provides shared persistence and serialization primitives:

- ``save(filepath)`` / ``load(filepath)`` for object persistence
- ``save_data`` / ``load_data`` for tabular artifacts
- ``save_scores`` / ``load_scores`` for score dictionaries
- stable hash computation via ``to_dict(for_hash=True)`` +
   :func:`deckard.utils.hash_conf_values`

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
