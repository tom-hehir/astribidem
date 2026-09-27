"""Record AION-2 crossmatch outputs for ``tests/test_aion_parity.py``.

Run from an AION-2 checkout's environment, with this directory on the path:

    PYTHONPATH=tests/fixtures /path/to/AION-2/.venv/bin/python \
        tests/fixtures/generate_aion_parity.py <AION-2 commit>

The synthetic catalogs are dense enough to produce duplicates, chains and
contested components. Inputs and outputs are both stored, so the test does
not depend on NumPy's random stream.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from aion2.token_materialization.crossmatch.candidate_edges import build_edges
from aion2.token_materialization.crossmatch.candidate_edges.coordinates import (
    survey_coords_from_arrays,
)
from aion2.token_materialization.crossmatch.config import (
    CrossmatchConfig,
    MatchingConfig,
    RadiusOverride,
    SurveyConfig,
)
from aion2.token_materialization.crossmatch.modes.degenerate.config import (
    DegenerateCrossmatchConfig,
)
from aion2.token_materialization.crossmatch.modes.entitywise.config import (
    EntitySelectionConfig,
    EntitywiseCrossmatchConfig,
)
from aion2.token_materialization.crossmatch.resolve_inputs import inputs_from_result
from aion_parity_cases import (
    CASES,
    DEDUPE_RADIUS_ARCSEC,
    PAIR_RADIUS_OVERRIDE,
    RADIUS_ARCSEC,
    SURVEYS,
)

ARCSEC = 1 / 3600


def synthetic_catalogs(rng):
    n_objects, side = 4000, 0.0913
    ra0 = 150 + rng.uniform(0, side, n_objects)
    dec0 = 2 + rng.uniform(0, side, n_objects)

    def survey(fraction, jitter, duplicate_fraction, field, chain_step):
        keep = rng.random(n_objects) < fraction
        ra = ra0[keep] + rng.normal(0, jitter, keep.sum()) * ARCSEC
        dec = dec0[keep] + rng.normal(0, jitter, keep.sum()) * ARCSEC
        copy = rng.random(len(ra)) < duplicate_fraction
        ra = np.r_[ra, ra[copy] + rng.normal(0, 0.15, copy.sum()) * ARCSEC]
        dec = np.r_[dec, dec[copy] + rng.normal(0, 0.15, copy.sum()) * ARCSEC]
        ra = np.r_[ra, 150 + rng.uniform(0, side, field)]
        dec = np.r_[dec, 2 + rng.uniform(0, side, field)]
        if chain_step:
            # Three rows in a line, neighbours within the dedupe radius but the
            # ends outside it: dedupe disputes every member.
            start = rng.integers(len(ra), size=40)
            steps = np.arange(1, 3)[:, None] * chain_step * ARCSEC
            ra = np.r_[ra, (ra[start] + steps).ravel()]
            dec = np.r_[dec, np.tile(dec[start], 2)]
        order = rng.permutation(len(ra))
        return ra[order], dec[order]

    return {
        "a": survey(0.7, 0.2, 0.03, 700, chain_step=0.4),
        "b": survey(0.6, 0.3, 0.02, 1100, chain_step=0.3),
        "c": survey(0.5, 0.25, 0.04, 400, chain_step=None),
    }


def aion_mode(kind, arguments):
    if kind == "degenerate":
        return DegenerateCrossmatchConfig(
            tuple(arguments["surveys"]), arguments["policy"]
        )
    arguments = dict(arguments)
    if "priority" in arguments:
        arguments["priority"] = tuple(arguments["priority"])
    if "selection" in arguments:
        selection = arguments["selection"]
        arguments["selection"] = EntitySelectionConfig(
            min_surveys=selection["min_surveys"],
            must_include_surveys=tuple(selection["must_include_surveys"]),
        )
    return EntitywiseCrossmatchConfig(**arguments)


def main(aion_commit):
    catalogs = synthetic_catalogs(np.random.default_rng(20260927))
    config = CrossmatchConfig(
        surveys=tuple(
            SurveyConfig(name, "unused", name, DEDUPE_RADIUS_ARCSEC[name])
            for name in SURVEYS
        ),
        output_dir="unused",
        matching=MatchingConfig(
            RADIUS_ARCSEC, (RadiusOverride(*PAIR_RADIUS_OVERRIDE),)
        ),
    )
    coords = {
        name: survey_coords_from_arrays(name, *catalogs[name]) for name in SURVEYS
    }
    arrays = {"aion_commit": np.array(aion_commit)}
    for name, (ra, dec) in catalogs.items():
        arrays[f"input/{name}/ra"] = ra
        arrays[f"input/{name}/dec"] = dec
    for label, (kind, arguments) in CASES.items():
        mode = aion_mode(kind, arguments)
        pairs = (
            None
            if mode.requires_all_surveys()
            else {frozenset(p) for p in combinations(mode.required_surveys(), 2)}
        )
        edges = build_edges(config, coords=coords, pairs_to_build=pairs)
        table = mode.build(inputs_from_result(edges)).table
        for column in table.column_names:
            values = table[column]
            if column.endswith("/row_index"):
                values = values.fill_null(-1).to_numpy()
            elif column == "disputed_reason":
                values = np.array(values.fill_null("").to_pylist())
            else:
                values = values.to_numpy()
            arrays[f"{label}/{column}"] = values
    path = Path(__file__).with_name("aion_parity.npz")
    np.savez_compressed(path, **arrays)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
