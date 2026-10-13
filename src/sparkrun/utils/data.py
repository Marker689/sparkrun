"""Detached data values for configuration and public snapshots; no I/O."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import PurePath
from types import MappingProxyType
from typing import Any


def freeze(value: Any, *, label: str = "Data") -> Any:
    """Make primitive mappings/sequences recursively read-only."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item, label=label) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item, label=label) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("%s must contain only mappings, sequences and scalar values" % label)


def thaw(value: Any) -> Any:
    """Copy read-only data into ordinary mutable mappings/lists."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def normalize_data(value: Any, *, path: str = "data") -> Any:
    """Copy serializable values; dates become ISO strings, paths become strings.

    Mappings require string keys; tuples become lists. Other application objects
    are rejected with a field path instead of leaking into a public snapshot.
    """
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("%s must have string keys" % path)
        return {key: normalize_data(item, path="%s.%s" % (path, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_data(item, path="%s[%d]" % (path, index)) for index, item in enumerate(value)]
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported %s value: %s" % (path, type(value).__name__))


def _numeric_setting[T](value: object, key: str, parse: Callable[[int | float | str], T]) -> T:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ValueError("Configuration %r must be numeric" % key)
    try:
        return parse(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Configuration %r must be numeric" % key) from error


def integer_setting(value: object, *, key: str) -> int:
    """Convert a numeric configuration scalar, naming invalid input in errors."""
    return _numeric_setting(value, key, int)


def positive_integer_setting(value: object, *, key: str, minimum: int = 1) -> int:
    """Parse a finite integral count without truncation, then enforce its bound."""
    from decimal import Decimal, InvalidOperation

    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value() or number < minimum:
            raise ValueError
        return int(number)
    except (ValueError, InvalidOperation, OverflowError) as error:
        raise ValueError("Configuration %r must be an integer >= %d" % (key, minimum)) from error


def float_setting(value: object, *, key: str) -> float:
    """Convert a numeric scalar; the consumer owns finite/range policy."""
    return _numeric_setting(value, key, float)


def byte_size_setting(value: object, *, key: str) -> int:
    """Positive byte count with vLLM's decimal/binary suffix convention."""
    import re
    from decimal import Decimal

    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmMgGtT]?)", value.strip())
        if match:
            number, suffix = match.groups()
            # Fractional values are supported only with decimal multipliers.
            if "." not in number or (suffix and suffix.islower()):
                power = "kmgt".index(suffix.lower()) + 1 if suffix else 0
                result = Decimal(number) * (1024 if suffix.isupper() else 1000) ** power
                if result > 0 and result == int(result):
                    return int(result)
    raise ValueError("Configuration %r must be a positive byte count (for example 8589934592 or '8G')" % key)
