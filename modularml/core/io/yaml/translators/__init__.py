from typing import Any


def _to_yaml_safe(value: Any) -> Any:
    """Recursively convert tuples -> lists so PyYAML safe_load can round-trip them."""
    if isinstance(value, tuple):
        return [_to_yaml_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_yaml_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_yaml_safe(v) for v in value]
    return value
