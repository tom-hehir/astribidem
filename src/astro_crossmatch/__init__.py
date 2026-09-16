"""Exact astronomical crossmatching and stable-ID entity indexes."""

from astro_crossmatch.api import crossmatch
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
    "survey_coords_from_arrays",
]
