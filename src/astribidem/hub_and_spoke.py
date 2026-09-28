"""Hub-and-spoke matching with independent UID or spatial anchor links."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from astribidem._formats import index_metadata
from astribidem._radii import dedupe_radii, radius
from astribidem.api import resolve
from astribidem.candidate_edges import CandidateEdges
from astribidem.edges import build_pair_edges, dedupe_survey
from astribidem.kernel import CatalogKernel, resolve_workers
from astribidem.modes import SUBSET_JOIN_POLICIES, DegenerateCrossmatchConfig
from astribidem.uids import match_uids


@dataclass(frozen=True)
class UIDLink:
    """Match the anchor and counterpart by exact, unique UID equality."""


@dataclass(frozen=True)
class SpatialLink:
    """Match the full anchor and counterpart with an existing spatial policy."""

    radius_arcsec: float
    policy: str = "mutual_nearest"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "radius_arcsec", radius(self.radius_arcsec, name="radius_arcsec")
        )
        if not isinstance(self.policy, str) or self.policy not in SUBSET_JOIN_POLICIES:
            raise ValueError(f"unknown spatial policy: {self.policy!r}")


def _spatial_kernel(name: str, table: pa.Table) -> CatalogKernel:
    """Prepare one spatial participant in its original row order."""
    if not {"ra", "dec"} <= set(table.column_names):
        raise ValueError(f"spatial catalog {name!r} requires ra and dec columns")
    for column in ("ra", "dec"):
        values = table[column]
        if not (pa.types.is_floating(values.type) or pa.types.is_integer(values.type)):
            raise ValueError(f"catalog {name!r}: {column} must be numeric")
        if values.null_count:
            raise ValueError(f"catalog {name!r}: {column} must be non-null")
    return CatalogKernel(
        table["ra"].to_numpy(zero_copy_only=False),
        table["dec"].to_numpy(zero_copy_only=False),
        name=name,
    )


def match_hub_and_spoke(
    catalogs: Mapping[str, pa.Table],
    *,
    anchor: str,
    links: Mapping[str, UIDLink | SpatialLink],
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Match every spoke to one anchor using UID or spatial links.

    All-UID, all-spatial, and mixed-link configurations use the same topology.
    Return the inner intersection in anchor input order; every link must
    succeed, with no spoke-to-spoke tests or groups lacking the anchor.

    UID links use each table's ``uid`` column when present, otherwise ``id``.
    Spatial participants need ``ra`` and ``dec`` in degrees. ``links`` names
    every non-anchor source. The scalar ``dedupe_radius_arcsec`` applies to
    the anchor and its spatial counterparts; optional
    ``dedupe_radius_arcsec_overrides`` names exceptions among those participants.
    Use zero to disable spatial deduplication; UID uniqueness is unchanged.

    Every link uses its complete input catalogs, even when another link has no
    matches. This preserves spatial ambiguity and deduplication decisions.
    Successful links are combined only by anchor row, with inner intersection;
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
    radii = dedupe_radii(
        (name for name in names if name in spatial),
        dedupe_radius_arcsec,
        dedupe_radius_arcsec_overrides,
    )
    if isinstance(workers, bool) or not isinstance(workers, Integral):
        raise TypeError("workers must be an integer >= 1 or -1")
    workers = resolve_workers(int(workers))

    uid_sources = {name for name, link in links.items() if isinstance(link, UIDLink)}
    if uid_sources:
        uid_sources.add(anchor)
    for name, table in catalogs.items():
        if not isinstance(table, pa.Table):
            raise TypeError(f"catalog {name!r} must be a pyarrow.Table")
        if len(set(table.column_names)) != len(table.column_names):
            raise ValueError(f"catalog {name!r} has duplicate column names")
        if name in uid_sources and not {"uid", "id"} & set(table.column_names):
            raise ValueError(f"catalog {name!r} needs a uid or id column for UID links")

    # The full anchor is prepared once for this invocation. Every spatial
    # spoke reuses its tree and dedupe verdict, including after an empty link.
    if spatial:
        hub_kernel = _spatial_kernel(anchor, catalogs[anchor])
        hub_outcome, hub_active = dedupe_survey(anchor, radii[anchor], hub_kernel)

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
                    pair.schema.metadata[b"astribidem.resolved_config"]
                ),
            }
        else:
            spoke_kernel = _spatial_kernel(name, catalogs[name])
            spoke_outcome, spoke_active = dedupe_survey(name, radii[name], spoke_kernel)
            edge = build_pair_edges(
                anchor,
                name,
                hub_active,
                spoke_active,
                link.radius_arcsec,
                kernel_a=hub_kernel,
                kernel_b=spoke_kernel,
                workers=workers,
            )
            del spoke_kernel, spoke_active
            pair = resolve(
                CandidateEdges(
                    surveys=(hub_outcome, spoke_outcome),
                    pairs={frozenset(pair_names): edge},
                ),
                DegenerateCrossmatchConfig(
                    surveys=list(pair_names), policy=link.policy
                ),
            )
            del edge, spoke_outcome
            pair_rows = {source: pair[f"{source}/row_index"] for source in pair_names}
            link_provenance[name] = {
                "method": "sky",
                "radius_arcsec": link.radius_arcsec,
                "policy": link.policy,
                "resolved_config": json.loads(
                    pair.schema.metadata[b"astribidem.resolved_config"]
                ),
                "dedupe": json.loads(pair.schema.metadata[b"astribidem.dedupe"]),
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

    if spatial:
        del hub_kernel, hub_outcome, hub_active

    configuration = {
        "method": "hub_and_spoke",
        "anchor": anchor,
        "join": "inner",
        "topology": "anchor-pairs",
        "ordering": "anchor_input_order",
        "links": link_provenance,
        "dedupe_radius_arcsec": radii,
        "surveys": [{"name": name} for name in names],
    }
    columns = {"entity_id": pa.array(np.arange(len(rows[anchor]), dtype=np.int64))}
    for name in names:
        columns[f"{name}/row_index"] = pc.cast(rows.pop(name), pa.int64())
    return pa.table(columns).replace_schema_metadata(
        {
            **index_metadata(),
            b"astribidem.resolved_config": json.dumps(
                configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode(),
        }
    )
