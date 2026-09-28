"""Hub-and-spoke matching with independent UID or spatial anchor links."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from astro_crossmatch.api import crossmatch
from astro_crossmatch.edges import survey_coords_from_arrays
from astro_crossmatch.kernel import float64_coordinates, resolve_workers
from astro_crossmatch.modes import SUBSET_JOIN_POLICIES, DegenerateCrossmatchConfig
from astro_crossmatch.uids import match_uids


def _radius(value: float, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"radius must be a finite number {bound}")
    return float(value)


@dataclass(frozen=True)
class UIDLink:
    """Match the anchor and counterpart by exact, unique UID equality."""


@dataclass(frozen=True)
class SpatialLink:
    """Match the full anchor and counterpart with an existing spatial policy."""

    radius_arcsec: float
    policy: str = "mutual_nearest"

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_arcsec", _radius(self.radius_arcsec))
        if not isinstance(self.policy, str) or self.policy not in SUBSET_JOIN_POLICIES:
            raise ValueError(f"unknown spatial policy: {self.policy!r}")


def _spatial_input(name: str, table: pa.Table):
    """Coordinates for spatial matching, in the catalog's row order."""
    if not {"ra", "dec"} <= set(table.column_names):
        raise ValueError(f"spatial catalog {name!r} requires ra and dec columns")
    for column in ("ra", "dec"):
        values = table[column]
        if not (pa.types.is_floating(values.type) or pa.types.is_integer(values.type)):
            raise ValueError(f"catalog {name!r}: {column} must be numeric")
        if values.null_count:
            raise ValueError(f"catalog {name!r}: {column} must be non-null")
    ra, dec = float64_coordinates(
        table["ra"].to_numpy(zero_copy_only=False),
        table["dec"].to_numpy(zero_copy_only=False),
        name=name,
    )
    for column, values in (("ra", ra), ("dec", dec)):
        if not np.isfinite(values).all():
            raise ValueError(f"catalog {name!r}: {column} must be finite")
    return survey_coords_from_arrays(name, ra, dec)


def match_hub_and_spoke(
    catalogs: Mapping[str, pa.Table],
    *,
    anchor: str,
    links: Mapping[str, UIDLink | SpatialLink],
    dedupe_radius_arcsec: Mapping[str, float],
    workers: int = 1,
) -> pa.Table:
    """Match every spoke to one anchor using UID or spatial links.

    All-UID, all-spatial, and mixed-link configurations use the same topology.
    Return the inner intersection in anchor input order; every link must
    succeed, with no spoke-to-spoke tests or groups lacking the anchor.

    UID links use each table's ``uid`` column when present, otherwise ``id``.
    Spatial participants need ``ra`` and ``dec`` in degrees. ``links`` names every non-anchor source;
    explicit dedupe radii name exactly the anchor and its spatial counterparts
    (or an empty mapping when there are no spatial links).

    Every link uses its complete input catalogs, even when another link has no
    matches. This preserves spatial ambiguity and deduplication decisions.
    Successful links are combined only by anchor ID, with inner intersection;
    counterparts need not match each other. This differs from
    the all-pairs contract of an N-survey spatial subset join.

    Output contains ``entity_id`` and an int64 ``<source>/row_index`` column per
    table: row ``i`` is the table's ``i``-th row. Use ``rows_to_ids`` to map rows
    to IDs. Anchored spatial policies may repeat counterpart rows. Keys remain in native
    Arrow buffers; only numeric coordinates and row positions use NumPy. Spatial
    dedupe keeps the lowest row of each duplicate group, so it follows the
    catalog's row order.
    """
    if not isinstance(catalogs, Mapping) or len(catalogs) < 2:
        raise ValueError("catalogs must map at least two sources to Arrow tables")
    names = tuple(catalogs)
    if any(not isinstance(name, str) or not name or "/" in name for name in names):
        raise ValueError("source names must be non-empty and must not contain '/'")
    if anchor not in catalogs:
        raise ValueError("anchor must name an input catalog")
    others = [name for name in names if name != anchor]
    if not isinstance(links, Mapping) or set(links) != set(others):
        raise ValueError("links must name every non-anchor catalog exactly once")
    if any(not isinstance(link, (UIDLink, SpatialLink)) for link in links.values()):
        raise TypeError("links must contain UIDLink or SpatialLink configurations")
    spatial = {name for name, link in links.items() if isinstance(link, SpatialLink)}
    if spatial:
        spatial.add(anchor)
    if (
        not isinstance(dedupe_radius_arcsec, Mapping)
        or set(dedupe_radius_arcsec) != spatial
    ):
        raise ValueError(
            "dedupe_radius_arcsec must name exactly the spatial participants"
        )
    dedupe_radii = {
        name: _radius(radius, allow_zero=True)
        for name, radius in dedupe_radius_arcsec.items()
    }
    if isinstance(workers, bool) or not isinstance(workers, Integral):
        raise TypeError("workers must be an integer >= 1 or -1")
    workers = resolve_workers(int(workers))

    uid_sources = {name for name, link in links.items() if isinstance(link, UIDLink)}
    if uid_sources:
        uid_sources.add(anchor)
    surveys = {}
    for name, table in catalogs.items():
        if not isinstance(table, pa.Table):
            raise TypeError(f"catalog {name!r} must be a pyarrow.Table")
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError(f"catalog {name!r} has duplicate column names")
        if name in uid_sources and not {"uid", "id"} & set(table.column_names):
            raise ValueError(f"catalog {name!r} needs a uid or id column for UID links")
        if name in spatial:
            surveys[name] = _spatial_input(name, table)

    rows = {anchor: pa.array(np.arange(catalogs[anchor].num_rows, dtype=np.int64))}
    link_provenance = {}
    for name in others:
        link = links[name]
        pair_names = (anchor, name)
        if isinstance(link, UIDLink):
            uid_columns = {
                source: "uid" if "uid" in catalogs[source].column_names else "id"
                for source in pair_names
            }
            pair = match_uids(
                {
                    source: catalogs[source][uid_columns[source]]
                    for source in pair_names
                },
                anchor=anchor,
            )
            pair_rows = {source: pair[f"{source}/row_index"] for source in pair_names}
            link_provenance[name] = {
                "method": "uid",
                "uid_columns": uid_columns,
                "resolved_config": json.loads(
                    pair.schema.metadata[b"astro_crossmatch.resolved_config"]
                ),
            }
        else:
            pair = crossmatch(
                [surveys[source] for source in pair_names],
                radius_arcsec=link.radius_arcsec,
                dedupe_radius_arcsec={
                    source: dedupe_radii[source] for source in pair_names
                },
                mode=DegenerateCrossmatchConfig(
                    surveys=list(pair_names), policy=link.policy
                ),
                workers=workers,
            )
            pair_rows = {source: pair[f"{source}/row_index"] for source in pair_names}
            link_provenance[name] = {
                "method": "sky",
                "radius_arcsec": link.radius_arcsec,
                "policy": link.policy,
                "resolved_config": json.loads(
                    pair.schema.metadata[b"astro_crossmatch.resolved_config"]
                ),
                "dedupe": json.loads(pair.schema.metadata[b"astro_crossmatch.dedupe"]),
            }
        link_provenance[name]["n_matches"] = pair.num_rows
        # Link decisions above always use full inputs. Only their completed
        # row maps are intersected with earlier links here.
        positions = pc.index_in(rows[anchor], value_set=pair_rows[anchor])
        keep = pc.is_valid(positions)
        for source, values in rows.items():
            rows[source] = pc.filter(values, keep)
        del values
        rows[name] = pc.take(pair_rows[name], pc.filter(positions, keep))
        del pair, pair_rows, positions, keep

    del surveys

    configuration = {
        "method": "hub_and_spoke",
        "anchor": anchor,
        "join": "inner",
        "topology": "anchor-pairs",
        "ordering": "anchor_input_order",
        "links": link_provenance,
        "dedupe_radius_arcsec": dedupe_radii,
        "surveys": [{"name": name} for name in names],
    }
    columns = {"entity_id": pa.array(np.arange(len(rows[anchor]), dtype=np.int64))}
    for name in names:
        columns[f"{name}/row_index"] = pc.cast(rows.pop(name), pa.int64())
    return pa.table(columns).replace_schema_metadata(
        {
            b"astro_crossmatch.resolved_config": json.dumps(
                configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        }
    )
