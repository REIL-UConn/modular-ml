from __future__ import annotations

import logging

from .formatters import ModularMLBannerFormatter, WarningFormatter

"""
Example usage of logging:

```python
from modularml.utils.logging import get_logger

logger = get_logger("featureset")
logger.info("FeatureSet initialized")
```
"""


_LOGGER_NAME = "modularml"


def get_logger(
    name: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a configured ModularML logger instance.

    Args:
        name (str | None):
            Optional child logger name (e.g., "data", "models.encoder").
        level (int):
            Logging level for the root ModularML logger.

    Returns:
        logging.Logger:
            Configured logger instance.

    """
    logger_name = _LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}"
    logger = logging.getLogger(logger_name)

    # Configure only once
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False

        handler = logging.StreamHandler()

        if name == "warnings":
            handler.setFormatter(WarningFormatter())
        else:
            handler.setFormatter(ModularMLBannerFormatter())

        logger.addHandler(handler)

    return logger
