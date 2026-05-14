"""Data configuration module.

This module is kept for backward compatibility and optional sampler registration.
Canonical data configs are now loaded from examples/*/config/data/ YAML files
at runtime via deckard.declarations.register_configs().

Sampler configs can still be registered optionally if needed for legacy tests.
"""

from .sample import register_sampler_configs

# Optional: Register sampler configs for legacy test compatibility
# Sampler configs are typically not used in main configs, but this preserves
# backward compatibility if tests rely on them
try:
    register_sampler_configs()
except Exception:
    pass
