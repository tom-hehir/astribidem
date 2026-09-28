"""Validate scalar radii and expand per-survey deduplication overrides."""

from collections.abc import Iterable, Mapping
from math import isfinite
from numbers import Real


def radius(value: float, *, name: str, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be a finite scalar radius {bound}")
    return float(value)


def dedupe_radii(
    names: Iterable[str],
    default: float,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Expand a scalar default and optional exceptions for spatial participants."""
    default = radius(default, name="dedupe_radius_arcsec", allow_zero=True)
    result = dict.fromkeys(names, default)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise TypeError("dedupe_radius_arcsec_overrides must be a mapping")
        for name, value in overrides.items():
            if name not in result:
                raise ValueError(
                    "dedupe_radius_arcsec_overrides names an unknown spatial "
                    f"participant: {name!r}"
                )
            result[name] = radius(
                value, name=f"dedupe override for {name!r}", allow_zero=True
            )
    return result
