"""Build stable-ID crossmatch indexes from in-memory coordinate arrays."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from math import isfinite

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from astro_crossmatch.edges import SurveyCoords, build_pair_edges, dedupe_survey
from astro_crossmatch.inputs import ResolveInputs
from astro_crossmatch.kernel import CatalogKernel, resolve_workers
from astro_crossmatch.modes import CrossmatchModeConfig, ModeResult, mode_to_mapping


def _validate_survey(survey: SurveyCoords) -> None:
    if not isinstance(survey.name, str) or not survey.name or "/" in survey.name:
        raise ValueError("survey names must be non-empty and must not contain '/'")
    ids = np.asarray(survey.ids)
    xyz = np.asarray(survey.xyz)
    if ids.ndim != 1 or xyz.shape != (len(ids), 3):
        raise ValueError(f"survey {survey.name!r}: ids and xyz must align")
    # from_pandas marks NaN/NaT as missing as well as ordinary Python None.
    # Keep Arrow's integer representation; nullable IDs must never pass through
    # a floating-point pandas column before they reach the emitted index.
    values = pa.array(ids, from_pandas=True)
    if values.null_count:
        raise ValueError(f"survey {survey.name!r}: IDs must be non-null")
    if pc.count_distinct(values).as_py() != len(values):
        raise ValueError(f"survey {survey.name!r}: IDs must be globally unique")
    if not np.isfinite(xyz).all():
        raise ValueError(f"survey {survey.name!r}: coordinates must be finite")


def crossmatch(
    surveys: Sequence[SurveyCoords],
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    mode: CrossmatchModeConfig,
    workers: int = 1,
    id_columns: Mapping[str, str] | None = None,
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
) -> ModeResult:
    """Resolve a coordinate collection into one stable-ID index.

    Every survey requires an explicit dedupe radius; ``0.0`` opts out.
    ``radius_arcsec`` is the default cross-survey radius. Optional pair
    overrides name exactly two surveys. The same unordered pair cannot be
    specified twice, even with equal radii.

    ``mode`` selects an Astral-derived subset join or exactly-once entity
    resolution. Matching uses complete in-radius candidate sets, and entity
    selection is applied only after all configured surveys are resolved.
    IDs retain their source Arrow-compatible types. Coordinates must already
    be cleaned; this function does not infer catalog-specific sentinel values.

    ``id_columns`` sets each survey's output field name (default ``id``).
    The result contains ``<survey>/<field>`` columns, optional separations or
    dispute reasons, and policy/provenance metadata. No payloads are read.
    """
    surveys = tuple(surveys)
    if not surveys:
        raise ValueError("at least one survey is required")
    for survey in surveys:
        _validate_survey(survey)
    names = tuple(survey.name for survey in surveys)
    known = set(names)
    if len(known) != len(names):
        raise ValueError("survey names must be unique")
    if set(dedupe_radius_arcsec) != known:
        raise ValueError("dedupe_radius_arcsec must name every survey exactly once")
    dedupe_radii = {name: float(dedupe_radius_arcsec[name]) for name in names}
    if any(not isfinite(value) or value < 0 for value in dedupe_radii.values()):
        raise ValueError("dedupe radii must be finite and >= 0")
    radius_arcsec = float(radius_arcsec)
    if not isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be finite and > 0")
    workers = resolve_workers(workers)
    if not isinstance(mode, CrossmatchModeConfig):
        raise TypeError("mode must be a CrossmatchModeConfig")
    unknown = set(mode.required_surveys()) - known
    if unknown:
        raise ValueError(f"mode names unknown surveys: {sorted(unknown)}")
    id_columns = dict(id_columns or {})
    if set(id_columns) - known:
        raise ValueError("id_columns names unknown surveys")
    columns = {name: id_columns.get(name, "id") for name in names}
    if any(
        not isinstance(value, str) or not value or "/" in value
        for value in columns.values()
    ):
        raise ValueError("ID column names must be non-empty and must not contain '/'")

    radii = {frozenset(pair): radius_arcsec for pair in combinations(names, 2)}
    overridden: set[frozenset[str]] = set()
    for pair, radius in (pair_radius_overrides or {}).items():
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("each radius override must name exactly two surveys")
        key = frozenset(pair)
        if len(key) != 2 or not key <= known:
            raise ValueError(f"radius override names invalid survey pair: {pair!r}")
        if key in overridden:
            raise ValueError(f"duplicate radius override for pair: {sorted(key)}")
        radius = float(radius)
        if not isfinite(radius) or radius <= 0:
            raise ValueError("override radii must be finite and > 0")
        overridden.add(key)
        radii[key] = radius

    coords = {survey.name: survey for survey in surveys}
    kernels = {name: CatalogKernel.from_xyz(coords[name].xyz) for name in names}
    dedupe = {}
    active = {}
    for name in names:
        dedupe[name], active[name] = dedupe_survey(
            coords[name], dedupe_radii[name], kernel=kernels[name]
        )

    required_pairs = (
        None
        if mode.requires_all_surveys()
        else {frozenset(pair) for pair in combinations(mode.required_surveys(), 2)}
    )
    pairs = {}
    for a, b in combinations(names, 2):
        pair = frozenset((a, b))
        if required_pairs is not None and pair not in required_pairs:
            continue
        pairs[pair] = build_pair_edges(
            coords[a],
            coords[b],
            active[a],
            active[b],
            radii[pair],
            kernel_a=kernels[a],
            kernel_b=kernels[b],
            workers=workers,
        )
    inputs = ResolveInputs(
        survey_names=names,
        active_row_index={name: np.flatnonzero(active[name]) for name in names},
        dedupe_disputed={name: dedupe[name].disputed_index for name in names},
        pairs=pairs,
        pair_radius_arcsec=radii,
        ids={name: np.asarray(coords[name].ids) for name in names},
        id_columns=columns,
    )
    result = mode.build(inputs)
    provenance = {
        "surveys": [
            {
                "name": name,
                "id_column": columns[name],
                "dedupe_radius_arcsec": dedupe_radii[name],
            }
            for name in names
        ],
        "pair_radius_arcsec": [
            {"surveys": [a, b], "radius_arcsec": radii[frozenset((a, b))]}
            for a, b in combinations(names, 2)
        ],
        "mode": mode_to_mapping(mode),
    }
    outcomes = {
        name: {
            "radius_arcsec": dedupe_radii[name],
            "n_rows": len(coords[name]),
            "n_dropped": dedupe[name].n_dropped,
            "n_disputed": dedupe[name].n_disputed,
        }
        for name in names
    }
    metadata = dict(result.table.schema.metadata or {})
    metadata[b"astro_crossmatch.resolved_config"] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    metadata[b"astro_crossmatch.dedupe"] = json.dumps(
        outcomes, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return ModeResult(result.table.replace_schema_metadata(metadata), result.summary)
