"""Exact astronomical crossmatching and stable-ID entity indexes."""

from astro_crossmatch.api import crossmatch
from astro_crossmatch.edges import SurveyCoords, survey_coords_from_arrays
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
    "CatalogKernel",
    "CrossmatchModeConfig",
    "DegenerateCrossmatchConfig",
    "EntitySelectionConfig",
    "EntitywiseCrossmatchConfig",
    "ModeResult",
    "SpatialLink",
    "SurveyCoords",
    "UIDLink",
    "crossmatch",
    "match_hub_and_spoke",
    "match_uids",
    "survey_coords_from_arrays",
]
