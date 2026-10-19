"""Exact astronomical crossmatching with row-position entity indexes."""

from importlib.metadata import version

from astribidem.api import crossmatch, ids_to_rows, resolve, rows_to_ids
from astribidem.audit import audit_edges
from astribidem.candidate_edges import (
    CandidateEdges,
    DedupeOutcome,
    PairEdges,
)
from astribidem.edge_files import (
    build_edges_to_directory,
    read_edges,
    read_edges_metadata,
    read_segment,
    segment_names,
    write_edges,
)
from astribidem.edges import build_edges
from astribidem.hub_and_spoke import SpatialLink, UIDLink, match_hub_and_spoke
from astribidem.index_table import index_summary
from astribidem.kernel import CatalogKernel
from astribidem.modes import (
    CrossmatchModeConfig,
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    one_to_one_pairs,
)
from astribidem.out_of_core.banded import (
    as_chunks,
    build_band,
    build_edges_by_band,
    prepare_band_build,
    sweep_boundaries,
    write_band_layout,
)
from astribidem.out_of_core.index_files import resolve_to_file
from astribidem.uids import match_uids

__version__ = version("astribidem")

__all__ = [
    "__version__",
    "CandidateEdges",
    "CatalogKernel",
    "CrossmatchModeConfig",
    "DedupeOutcome",
    "DegenerateCrossmatchConfig",
    "EntitySelectionConfig",
    "EntitywiseCrossmatchConfig",
    "PairEdges",
    "SpatialLink",
    "UIDLink",
    "as_chunks",
    "audit_edges",
    "build_band",
    "build_edges",
    "build_edges_by_band",
    "build_edges_to_directory",
    "crossmatch",
    "ids_to_rows",
    "index_summary",
    "match_hub_and_spoke",
    "match_uids",
    "one_to_one_pairs",
    "prepare_band_build",
    "read_edges",
    "read_edges_metadata",
    "read_segment",
    "resolve",
    "resolve_to_file",
    "rows_to_ids",
    "segment_names",
    "sweep_boundaries",
    "write_band_layout",
    "write_edges",
]
