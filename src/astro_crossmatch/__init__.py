"""Exact astronomical crossmatching and stable-ID entity indexes."""

from astro_crossmatch.api import crossmatch
from astro_crossmatch.uids import match_uids
from astro_crossmatch.edges import SurveyCoords, survey_coords_from_arrays
from astro_crossmatch.kernel import CatalogKernel
from astro_crossmatch.modes import (
    CrossmatchModeConfig,
    DegenerateCrossmatchConfig,
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
    ModeResult,
)

__all__ = [
    "CatalogKernel",
    "CrossmatchModeConfig",
    "DegenerateCrossmatchConfig",
    "EntitySelectionConfig",
    "EntitywiseCrossmatchConfig",
    "ModeResult",
    "SurveyCoords",
    "crossmatch",
    "match_uids",
    "survey_coords_from_arrays",
]
