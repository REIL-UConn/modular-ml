"""Utility registries with case-insensitive lookups."""

from typing import Any


class CaseInsensitiveRegistry(dict):
    """
    Dictionary-like registry with case-insensitive key lookup.

    Description:
        Keys are stored exactly as provided (preserving casing) while
        lookups (`[]`, :meth:`get`, membership checks) normalize to lowercase.
        Collisions on lowercase equivalents are rejected to avoid ambiguity.

    Attributes:
        _lower_map (dict[str, str]): Mapping of lowercased keys to canonical keys.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the registry and optionally seed it with values.

        Args:
            *args (Any): Optional iterable or mapping to preload entries.
            **kwargs (Any): Additional key-value pairs to register.

        """
        super().__init__()
        self._lower_map: dict[str, str] = {}
        if args or kwargs:
            self.update(*args, **kwargs)

    # ==========================================
    # Internal helpers
    # ==========================================
    def _normalize(self, key: str) -> str:
        """
        Normalize a key to lowercase and validate type.

        Args:
            key (str): Key to normalize.

        Returns:
            str: Lowercase representation of the key.

        Raises:
            TypeError: If `key` is not a string.

        """
        if not isinstance(key, str):
            msg = f"Registry keys must be strings, got {type(key)}"
            raise TypeError(msg)
        return key.lower()

    def get_original_key(self, key: str) -> str:
        """
        Return the stored original key matching this key (case-insensitive).

        Args:
            key (str): Lookup key whose canonical form should be retrieved.

        Returns:
            str | None: Canonical key or `None` if absent.

        """
        lk = self._normalize(key)
        return self._lower_map.get(lk)

    # ==========================================
    # Core dict overrides
    # ==========================================
    def __setitem__(self, key: str, value):
        """
        Insert an item while preventing lowercase collisions.

        Args:
            key (str): Registry key to insert.
            value (Any): Object to store.

        Raises:
            KeyError: If another key with the same lowercase form exists.

        """
        lk = self._normalize(key)

        # Enforce lowercase uniqueness
        if lk in self._lower_map and self._lower_map[lk] != key:
            msg = f"Cannot insert key '{key}' - lowercase equivalent collides with existing key '{self._lower_map[lk]}'"
            raise KeyError(msg)

        # Insert
        super().__setitem__(key, value)
        self._lower_map[lk] = key

    def __getitem__(self, key: str):
        """
        Retrieve an item using case-insensitive lookup.

        Args:
            key (str): Lookup key.

        Returns:
            Any: Stored value.

        Raises:
            KeyError: If no entry matches `key`.

        """
        orig = self.get_original_key(key)
        if orig is None:
            raise KeyError(key)
        return super().__getitem__(orig)

    def __delitem__(self, key: str):
        """
        Delete an item using case-insensitive lookup.

        Args:
            key (str): Lookup key.

        Raises:
            KeyError: If no entry matches `key`.

        """
        orig = self.get_original_key(key)
        if orig is None:
            raise KeyError(key)
        lk = orig.lower()
        del self._lower_map[lk]
        super().__delitem__(orig)

    def __contains__(self, key: str) -> bool:
        """
        Check membership using case-insensitive lookup.

        Args:
            key (str): Lookup key.

        Returns:
            bool: True if the key exists, False otherwise.

        """
        return self.get_original_key(key) is not None

    # ==========================================
    # Convenience methods
    # ==========================================
    def get(self, key: str, default=None):
        """
        Retrieve a value with case-insensitive lookup and default.

        Args:
            key (str): Lookup key.
            default (Any, optional): Fallback value returned when missing.

        Returns:
            Any: Stored value or `default`.

        """
        orig = self.get_original_key(key)
        if orig is None:
            return default
        return super().get(orig, default)

    def pop(self, key: str, default=None):
        """
        Remove a value with case-insensitive lookup.

        Args:
            key (str): Lookup key.
            default (Any, optional): Fallback returned when the key is missing.

        Returns:
            Any: Removed value or `default`.

        Raises:
            KeyError: If the key is missing and no `default` is provided.

        """
        orig = self.get_original_key(key)
        if orig is None:
            if default is None:
                raise KeyError(key)
            return default
        lk = orig.lower()
        self._lower_map.pop(lk, None)
        return super().pop(orig)

    def update(self, *args, **kwargs):
        """
        Update the registry while respecting case-insensitive uniqueness.

        Args:
            *args (Any): Optional iterable or mapping of entries.
            **kwargs (Any): Additional key-value pairs.

        """
        items = dict(*args, **kwargs)
        for k, v in items.items():
            self[k] = v

    # Optional: return keys case-insensitively or normally
    def original_keys(self):
        """
        Return keys as originally inserted.

        Returns:
            list[str]: Canonical keys in insertion order.

        """
        return list(self.keys())

    def register(self, name: str, obj: Any):
        """
        Register an object under a case-insensitive key.

        Args:
            name (str): Key to register with.
            obj (Any): Object to store.

        Raises:
            KeyError: If the key collides case-insensitively.

        """
        key_l = name.lower()
        if key_l in (k.lower() for k in self.keys()):
            msg = f"Duplicate registry key (case-insensitive): {name}"
            raise KeyError(msg)
        self[name] = obj
