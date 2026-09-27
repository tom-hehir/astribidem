"""Exact astronomical crossmatching with row-position entity indexes."""

from astro_crossmatch.api import crossmatch, resolve, rows_to_ids
from astro_crossmatch.audit import audit_edges
from astro_crossmatch.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astro_crossmatch.edge_files import (
    build_edges_to_directory,
    read_edges,
    write_edges,
)
from astro_crossmatch.edges import SurveyCoords, build_edges, survey_coords_from_arrays
from astro_crossmatch.hub_and_spoke import SpatialLink, UIDLink, match_hub_and_spoke
from astro_crossmatch.kernel import CatalogKernel
from astro_crossmatch.modes import (
    CrossmatchModeConfig,
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    ModeResult,
)
from astro_crossmatch.uids import match_uids

__all__ = [
    "CandidateEdges",
    "CatalogKernel",
    "CrossmatchModeConfig",
    "DedupeOutcome",
    "DegenerateCrossmatchConfig",
    "EntitySelectionConfig",
    "EntitywiseCrossmatchConfig",
    "ModeResult",
    "PairEdges",
    "SpatialLink",
    "SurveyCoords",
    "UIDLink",
    "audit_edges",
    "build_edges",
    "build_edges_to_directory",
    "crossmatch",
    "match_hub_and_spoke",
    "match_uids",
    "read_edges",
    "resolve",
    "rows_to_ids",
    "survey_coords_from_arrays",
    "write_edges",
]
