"""Hub-and-spoke matching with independent UID or spatial anchor links."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import ArrayLike

from astribidem.api import resolve
from astribidem.candidate_edges import CandidateEdges
from astribidem.edges import build_pair_edges, dedupe_radius_by_survey, dedupe_survey
from astribidem.index_table import (
    DEDUPE_KEY,
    RESOLVED_CONFIG_KEY,
    check_survey_names,
    encode_metadata,
    row_column,
)
from astribidem.kernel import CatalogKernel, check_match_radius, resolve_workers
from astribidem.modes import DegenerateCrossmatchConfig, SubsetJoinPolicy
from astribidem.uids import match_uids


@dataclass(frozen=True)
class UIDLink:
    """Match the anchor and counterpart by exact, unique UID equality."""


@dataclass(frozen=True)
class SpatialLink:
    """Match the full anchor and counterpart with an existing spatial policy."""

    radius_arcsec: float
    policy: SubsetJoinPolicy = "mutual_nearest"

    def __post_init__(self) -> None:
        check_match_radius(self.radius_arcsec, "SpatialLink.radius_arcsec")


def match_hub_and_spoke(
    *,
    anchor: str,
    links: Mapping[str, UIDLink | SpatialLink],
    coordinates: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    uids: Mapping[str, Any] | None = None,
    dedupe_radius_arcsec: float,
    dedupe_radius_arcsec_overrides: Mapping[str, float] | None = None,
    workers: int = 1,
) -> pa.Table:
    """Match every catalogue in ``links`` to one anchor, by UID or by position.

    ``coordinates`` maps the anchor (when any link is spatial) and each
    spatially linked catalogue to ``(ra, dec)`` arrays in degrees, as for
    ``crossmatch``. ``uids`` maps the anchor (when any link is a UID link) and
    each UID-linked catalogue to unique, non-null identifiers, as for
    ``match_uids``. The scalar ``dedupe_radius_arcsec`` applies to the anchor
    and its spatial counterparts; ``dedupe_radius_arcsec_overrides`` names
    exceptions among them. Zero disables spatial dedupe.

    Every link uses its complete inputs, even when another link has no
    matches, so spatial ambiguity and dedupe decisions do not depend on the
    other links. Links are then combined by anchor row as an inner
    intersection; counterparts need not match each other, unlike the
    all-pairs contract of a subset join over several surveys.

    The result holds ``entity_id`` and an int64 ``<name>/row_index`` column
    for the anchor, then for each catalogue in ``links`` order, with rows in
    anchor input order. Anchored spatial policies may repeat counterpart rows.
    Spatial dedupe keeps the lowest row of each duplicate group, so it follows
    the input order.

    Raises:
        ValueError: ``coordinates`` or ``uids`` does not name exactly the
            catalogues its links need; or the anchor's coordinates and UIDs
            differ in length, which would misalign its rows.
    """
    coordinates = dict(coordinates or {})
    uids = dict(uids or {})
    if anchor in links:
        raise ValueError(f"links must not name the anchor {anchor!r}")
    names = (anchor, *links)
    check_survey_names(names)
    spatial = [name for name, link in links.items() if isinstance(link, SpatialLink)]
    uid_linked = [name for name, link in links.items() if isinstance(link, UIDLink)]
    for given, linked, argument in (
        (coordinates, spatial, "coordinates"),
        (uids, uid_linked, "uids"),
    ):
        needed = {anchor, *linked} if linked else set()
        if set(given) != needed:
            raise ValueError(
                f"{argument} must name exactly {sorted(needed)}, got {sorted(given)}"
            )
    if spatial and uid_linked and len(coordinates[anchor][0]) != len(uids[anchor]):
        raise ValueError(
            f"the anchor's coordinates and UIDs differ in length: "
            f"{len(coordinates[anchor][0])} and {len(uids[anchor])}"
        )
    spatial_names = [anchor, *spatial] if spatial else []
    radii = dedupe_radius_by_survey(
        spatial_names, dedupe_radius_arcsec, dedupe_radius_arcsec_overrides
    )
    workers = resolve_workers(workers)

    # The full anchor is prepared once for this invocation. Every spatial
    # spoke reuses its tree and dedupe verdict, including after an empty link.
    if spatial:
        hub_kernel = CatalogKernel(*coordinates[anchor], name=anchor)
        hub_outcome, hub_active = dedupe_survey(anchor, radii[anchor], hub_kernel)
        n_anchor = len(hub_kernel)
    else:
        n_anchor = len(uids[anchor])

    rows = {anchor: pa.array(np.arange(n_anchor, dtype=np.int64))}
    link_provenance = {}
    for name, link in links.items():
        pair_names = (anchor, name)
        if isinstance(link, UIDLink):
            pair = match_uids(
                {source: uids[source] for source in pair_names}, anchor=anchor
            )
            link_provenance[name] = {
                "method": "uid",
                "resolved_config": json.loads(
                    pair.schema.metadata[RESOLVED_CONFIG_KEY]
                ),
            }
        else:
            spoke_kernel = CatalogKernel(*coordinates[name], name=name)
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
            link_provenance[name] = {
                "method": "sky",
                "radius_arcsec": link.radius_arcsec,
                "policy": link.policy,
                "resolved_config": json.loads(
                    pair.schema.metadata[RESOLVED_CONFIG_KEY]
                ),
                "dedupe": json.loads(pair.schema.metadata[DEDUPE_KEY]),
            }
        pair_rows = {source: pair[row_column(source)] for source in pair_names}
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
        columns[row_column(name)] = pc.cast(rows.pop(name), pa.int64())
    return pa.table(columns).replace_schema_metadata(
        {RESOLVED_CONFIG_KEY: encode_metadata(configuration)}
    )
