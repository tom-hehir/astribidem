"""Exact astronomical crossmatching with row-position entity indexes."""

from astribidem.api import crossmatch, resolve, rows_to_ids
from astribidem.audit import audit_edges
from astribidem.banded import (
    build_band,
    build_edges_by_band,
    prepare_band_build,
    sweep_boundaries,
    write_band_layout,
)
from astribidem.candidate_edges import (
    CandidateEdges,
    DedupeOutcome,
    PairEdges,
    combine_segments,
)
from astribidem.edge_files import (
    build_edges_to_directory,
    read_edges,
    read_segment,
    segment_names,
    write_edges,
    write_segment,
)
from astribidem.edges import SurveyCoords, build_edges, survey_coords_from_arrays
from astribidem.hub_and_spoke import SpatialLink, UIDLink, match_hub_and_spoke
from astribidem.index_files import resolve_to_file
from astribidem.kernel import CatalogKernel
from astribidem.modes import (
    CrossmatchModeConfig,
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    index_summary,
)
from astribidem.uids import match_uids

__all__ = [
    "CandidateEdges",
    "CatalogKernel",
    "CrossmatchModeConfig",
    "DedupeOutcome",
    "DegenerateCrossmatchConfig",
    "EntitySelectionConfig",
    "EntitywiseCrossmatchConfig",
    "PairEdges",
    "SpatialLink",
    "SurveyCoords",
    "UIDLink",
    "audit_edges",
    "build_band",
    "build_edges",
    "build_edges_by_band",
    "build_edges_to_directory",
    "combine_segments",
    "crossmatch",
    "index_summary",
    "match_hub_and_spoke",
    "match_uids",
    "prepare_band_build",
    "read_edges",
    "read_segment",
    "resolve",
    "resolve_to_file",
    "rows_to_ids",
    "segment_names",
    "survey_coords_from_arrays",
    "sweep_boundaries",
    "write_band_layout",
    "write_edges",
    "write_segment",
]
