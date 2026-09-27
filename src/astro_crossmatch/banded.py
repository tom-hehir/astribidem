"""Build candidate edges for catalogs larger than memory, band by band.

The build runs in three steps (see ``docs/design/banded-edge-builds.md``):

1. ``write_band_layout`` streams each survey's coordinates once and writes
   ``(row, ra, dec)`` grouped by declination band.
2. ``build_band`` loads one band plus a margin from its neighbours, saves the
   groups wholly inside the band as segment ``band-<k>``, and hands over the
   groups that reach the margin.
3. ``sweep_boundaries`` visits the band boundaries from south to north,
   completes the handed-over groups, saves them as segments
   ``boundary-<k>``, and writes ``metadata.json``.

``prepare_band_build`` records the settings that the band tasks and the sweep
read, so band tasks can run as separate processes. ``build_edges_by_band``
runs every step on one machine. The saved edges equal an in-memory
``build_edges`` of the same coordinates exactly.

Layout directory::

    layout.json                      band height, surveys and row counts
    <survey>/chunk-<i>.parquet       row, ra, dec; one row group per band present
    <survey>/index.parquet           chunk, row_group, band, n_rows
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.candidate_edges import DedupeOutcome
from astro_crossmatch.edge_files import write_metadata, write_segment
from astro_crossmatch.edges import EdgeSettings, edge_settings
from astro_crossmatch.graph import connected_components
from astro_crossmatch.kernel import float64_coordinates
from astro_crossmatch.regions import (
    MARGIN_ABOVE,
    MARGIN_BELOW,
    OWNED,
    Handover,
    RegionRows,
    build_region,
    combine_handovers,
    finish_handover,
)

_ARCSEC_PER_DEGREE = 3600.0
_LAYOUT_SCHEMA = pa.schema(
    [("row", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
)
_PLAN = "band-build.json"
_HANDOVERS = ".handovers"


# --- the regrouped layout -------------------------------------------------------


def band_of(dec: np.ndarray, band_height_deg: float) -> np.ndarray:
    """The declination band of each row: ``floor((dec + 90) / band_height)``."""
    return np.floor((np.asarray(dec) + 90.0) / band_height_deg).astype(np.int64)


def write_band_layout(
    directory: str | Path,
    surveys: Mapping[str, Iterable[tuple[np.ndarray, np.ndarray]]],
    *,
    band_height_deg: float,
) -> None:
    """Stream each survey's coordinates into a layout grouped by band.

    ``surveys`` maps each survey name, in survey order, to an iterable of
    ``(ra, dec)`` chunks in degrees; rows are numbered in the order they
    arrive. Each chunk is written as one file with one row group per band it
    contains. Coordinates become float64 first, as everywhere else.
    """
    band_height_deg = float(band_height_deg)
    if not np.isfinite(band_height_deg) or band_height_deg <= 0:
        raise ValueError("band_height_deg must be finite and > 0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, chunks in surveys.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError("survey names must be non-empty and must not contain '/'")
        survey_directory = directory / name
        survey_directory.mkdir(exist_ok=True)
        index = {"chunk": [], "row_group": [], "band": [], "n_rows": []}
        offset = 0
        for chunk_index, (ra, dec) in enumerate(chunks):
            ra, dec = float64_coordinates(ra, dec, name=name)
            if ra.shape != dec.shape or ra.ndim != 1:
                raise ValueError(
                    f"survey {name!r}: ra/dec chunks must be 1-D and aligned"
                )
            if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
                raise ValueError(f"survey {name!r}: coordinates must be finite")
            rows = np.arange(offset, offset + len(ra), dtype=np.int64)
            offset += len(ra)
            bands = band_of(dec, band_height_deg)
            order = np.argsort(bands, kind="stable")
            bands, rows, ra, dec = bands[order], rows[order], ra[order], dec[order]
            starts = (
                np.flatnonzero(np.r_[True, bands[1:] != bands[:-1]])
                if len(bands)
                else []
            )
            path = survey_directory / f"chunk-{chunk_index:05d}.parquet"
            with pq.ParquetWriter(path, _LAYOUT_SCHEMA) as writer:
                for group, start in enumerate(starts):
                    stop = starts[group + 1] if group + 1 < len(starts) else len(bands)
                    writer.write_table(
                        pa.table(
                            [rows[start:stop], ra[start:stop], dec[start:stop]],
                            schema=_LAYOUT_SCHEMA,
                        ),
                        row_group_size=stop - start,
                    )
                    index["chunk"].append(chunk_index)
                    index["row_group"].append(group)
                    index["band"].append(int(bands[start]))
                    index["n_rows"].append(stop - start)
        pq.write_table(
            pa.table(
                {
                    "chunk": pa.array(index["chunk"], pa.int32()),
                    "row_group": pa.array(index["row_group"], pa.int32()),
                    "band": pa.array(index["band"], pa.int64()),
                    "n_rows": pa.array(index["n_rows"], pa.int64()),
                }
            ),
            survey_directory / "index.parquet",
        )
        entries.append({"name": name, "n_rows": offset})
    layout = {"band_height_deg": band_height_deg, "surveys": entries}
    (directory / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")


def read_layout(directory: str | Path) -> dict:
    """The contents of a layout directory's ``layout.json``."""
    return json.loads((Path(directory) / "layout.json").read_text())


def _read_bands(directory: Path, survey: str, bands: Sequence[int]):
    """Rows, RA and Dec of one survey's rows in the given bands."""
    index = pq.read_table(directory / survey / "index.parquet")
    wanted = np.isin(index["band"].to_numpy(), list(bands))
    chunks = index["chunk"].to_numpy()[wanted]
    groups = index["row_group"].to_numpy()[wanted]
    tables = []
    for chunk in np.unique(chunks):
        path = directory / survey / f"chunk-{chunk:05d}.parquet"
        with pq.ParquetFile(path) as parquet:
            tables.append(parquet.read_row_groups(groups[chunks == chunk].tolist()))
    if not tables:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.astype(np.float64), empty.astype(np.float64)
    table = pa.concat_tables(tables)
    return (
        table["row"].to_numpy(),
        table["ra"].to_numpy(),
        table["dec"].to_numpy(),
    )


# --- planning -------------------------------------------------------------------


def prepare_band_build(
    layout_directory: str | Path,
    edges_directory: str | Path,
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> list[int]:
    """Validate the settings, record them, and return the bands to build.

    The band height must exceed the largest pair or dedupe radius: otherwise a
    pair between rows two bands apart could be missed by every band.
    """
    layout_directory = Path(layout_directory)
    edges_directory = Path(edges_directory)
    layout = read_layout(layout_directory)
    settings = edge_settings(
        {entry["name"]: entry["n_rows"] for entry in layout["surveys"]},
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        pair_radius_overrides=pair_radius_overrides,
        pairs=pairs,
    )
    band_height_arcsec = layout["band_height_deg"] * _ARCSEC_PER_DEGREE
    if band_height_arcsec <= settings.max_radius_arcsec * (1 + 1e-6):
        raise ValueError(
            f"band height {band_height_arcsec} arcsec must exceed the largest radius "
            f"{settings.max_radius_arcsec} arcsec"
        )
    bands = sorted(
        {
            int(band)
            for entry in layout["surveys"]
            for band in pq.read_table(
                layout_directory / entry["name"] / "index.parquet", columns=["band"]
            )["band"].to_numpy()
        }
    )
    edges_directory.mkdir(parents=True, exist_ok=True)
    plan = {
        "layout": str(layout_directory.resolve()),
        "band_height_deg": layout["band_height_deg"],
        "bands": bands,
        "surveys": list(settings.survey_names),
        "n_rows": settings.n_rows,
        "dedupe_radius_arcsec": settings.dedupe_radius_arcsec,
        "pair_radius_arcsec": [
            {"surveys": [a, b], "radius_arcsec": radius}
            for (a, b), radius in settings.pair_radius_arcsec.items()
        ],
    }
    (edges_directory / _PLAN).write_text(json.dumps(plan, indent=2) + "\n")
    return bands


def _read_plan(edges_directory: Path) -> tuple[dict, EdgeSettings]:
    plan = json.loads((edges_directory / _PLAN).read_text())
    settings = EdgeSettings(
        survey_names=tuple(plan["surveys"]),
        n_rows={name: int(n) for name, n in plan["n_rows"].items()},
        dedupe_radius_arcsec={
            name: float(r) for name, r in plan["dedupe_radius_arcsec"].items()
        },
        pair_radius_arcsec={
            tuple(entry["surveys"]): float(entry["radius_arcsec"])
            for entry in plan["pair_radius_arcsec"]
        },
    )
    return plan, settings


# --- band tasks -----------------------------------------------------------------


def _margin_deg(settings: EdgeSettings) -> float:
    # Slightly wider than the largest radius, so rounding in declination can
    # never exclude a true neighbour; a wider margin is always safe.
    return settings.max_radius_arcsec / _ARCSEC_PER_DEGREE * (1 + 1e-6) + 1e-12


def _band_region(plan: dict, settings: EdgeSettings, band: int) -> dict:
    layout = Path(plan["layout"])
    height = plan["band_height_deg"]
    lower = band * height - 90.0
    upper = lower + height
    margin = _margin_deg(settings)
    region = {}
    for name in settings.survey_names:
        rows, ra, dec = _read_bands(layout, name, [band - 1, band, band + 1])
        bands = band_of(dec, height)
        side = np.full(len(rows), 2, dtype=np.int8)
        side[bands == band] = OWNED
        side[(bands == band - 1) & (dec >= lower - margin)] = MARGIN_BELOW
        side[(bands == band + 1) & (dec <= upper + margin)] = MARGIN_ABOVE
        take = side != 2
        region[name] = RegionRows(
            rows=rows[take], ra=ra[take], dec=dec[take], side=side[take]
        )
    return region


def build_band(edges_directory: str | Path, band: int, *, workers: int = 1) -> None:
    """Build one band: save segment ``band-<k>`` and its handover.

    The band task reads the settings recorded by ``prepare_band_build`` and
    the band's rows from the layout, so tasks can run in separate processes.
    """
    edges_directory = Path(edges_directory)
    plan, settings = _read_plan(edges_directory)
    segment, handover = build_region(
        settings, _band_region(plan, settings, band), workers=workers
    )
    write_segment(edges_directory, f"band-{band}", segment)
    _write_handover(edges_directory / _HANDOVERS / f"band-{band}", settings, handover)


def _write_handover(directory: Path, settings: EdgeSettings, handover: Handover):
    directory.mkdir(parents=True, exist_ok=True)
    for name in settings.survey_names:
        pq.write_table(
            pa.table(
                {
                    "row": handover.rows[name],
                    "pending": handover.pending[name],
                    "piece": handover.piece[name],
                }
            ),
            directory / f"rows-{name}.parquet",
        )
        first, second = handover.self_pairs[name]
        pq.write_table(
            pa.table({"first": first, "second": second}),
            directory / f"self-{name}.parquet",
        )
    for (a, b), (row_a, row_b, separation) in handover.edges.items():
        pq.write_table(
            pa.table({"row_a": row_a, "row_b": row_b, "separation_arcsec": separation}),
            directory / f"edges-{a}__{b}.parquet",
        )
    pq.write_table(
        pa.table({"below": handover.touches[:, 0], "above": handover.touches[:, 1]}),
        directory / "touches.parquet",
    )


def _read_handover(directory: Path, settings: EdgeSettings) -> Handover:
    rows, pending, piece, self_pairs = {}, {}, {}, {}
    for name in settings.survey_names:
        table = pq.read_table(directory / f"rows-{name}.parquet")
        rows[name] = table["row"].to_numpy()
        pending[name] = table["pending"].to_numpy()
        piece[name] = table["piece"].to_numpy()
        pairs = pq.read_table(directory / f"self-{name}.parquet")
        self_pairs[name] = (pairs["first"].to_numpy(), pairs["second"].to_numpy())
    edges = {}
    for a, b in settings.pair_radius_arcsec:
        table = pq.read_table(directory / f"edges-{a}__{b}.parquet")
        edges[(a, b)] = (
            table["row_a"].to_numpy(),
            table["row_b"].to_numpy(),
            table["separation_arcsec"].to_numpy(),
        )
    touches = pq.read_table(directory / "touches.parquet")
    return Handover(
        rows=rows,
        pending=pending,
        piece=piece,
        touches=np.column_stack(
            [touches["below"].to_numpy(), touches["above"].to_numpy()]
        ).astype(bool)
        if touches.num_rows
        else np.zeros((0, 2), dtype=bool),
        self_pairs=self_pairs,
        edges=edges,
    )


# --- the boundary sweep ---------------------------------------------------------


def _row_pieces(handover: Handover, name: str, rows: np.ndarray) -> np.ndarray:
    """The piece of each of ``rows`` in ``handover``, or -1 when absent."""
    order = np.argsort(handover.rows[name], kind="stable")
    sorted_rows = handover.rows[name][order]
    if len(sorted_rows) == 0:
        return np.full(len(rows), -1, dtype=np.int64)
    index = np.clip(np.searchsorted(sorted_rows, rows), 0, len(sorted_rows) - 1)
    found = sorted_rows[index] == rows
    return np.where(found, handover.piece[name][order][index], -1)


def _piece_groups(settings: EdgeSettings, handover: Handover) -> np.ndarray:
    """Join pieces that share a pair into groups; return each piece's group."""
    first, second = [], []
    for name in settings.survey_names:
        a, b = handover.self_pairs[name]
        first.append(_row_pieces(handover, name, a))
        second.append(_row_pieces(handover, name, b))
    for (a, b), (row_a, row_b, _) in handover.edges.items():
        first.append(_row_pieces(handover, a, row_a))
        second.append(_row_pieces(handover, b, row_b))
    first = np.concatenate(first) if first else np.empty(0, dtype=np.int64)
    second = np.concatenate(second) if second else np.empty(0, dtype=np.int64)
    both = (first >= 0) & (second >= 0)
    return connected_components(handover.n_pieces, first[both], second[both])


def _select_pieces(
    settings: EdgeSettings, handover: Handover, keep: np.ndarray
) -> Handover:
    """The part of ``handover`` belonging to the pieces where ``keep`` is true."""
    rows, pending, piece, self_pairs, edges = {}, {}, {}, {}, {}
    renumber = np.cumsum(keep) - 1
    for name in settings.survey_names:
        mask = keep[handover.piece[name]] if handover.n_pieces else np.zeros(0, bool)
        rows[name] = handover.rows[name][mask]
        pending[name] = handover.pending[name][mask]
        piece[name] = renumber[handover.piece[name][mask]]
        a, b = handover.self_pairs[name]
        emitter = _row_pieces(handover, name, np.minimum(a, b))
        chosen = (emitter >= 0) & keep[np.maximum(emitter, 0)]
        self_pairs[name] = (a[chosen], b[chosen])
    for (a, b), (row_a, row_b, separation) in handover.edges.items():
        emitter = _row_pieces(handover, a, row_a)
        chosen = (emitter >= 0) & keep[np.maximum(emitter, 0)]
        edges[(a, b)] = (row_a[chosen], row_b[chosen], separation[chosen])
    return Handover(
        rows=rows,
        pending=pending,
        piece=piece,
        touches=handover.touches[keep],
        self_pairs=self_pairs,
        edges=edges,
    )


def sweep_boundaries(
    edges_directory: str | Path, *, max_carried_rows: int | None = None
) -> None:
    """Complete handed-over groups boundary by boundary; write ``metadata.json``.

    Boundary ``k`` lies between bands ``k`` and ``k + 1``. The sweep takes the
    pieces that reach each boundary, joins them into groups, saves the groups
    that reach no later boundary as segment ``boundary-<k>``, and carries the
    rest forward. ``max_carried_rows`` bounds how many rows may be carried; a
    larger carry means groups span many bands, which happens only when the
    radius is too large for the source density.
    """
    edges_directory = Path(edges_directory)
    plan, settings = _read_plan(edges_directory)
    handover_root = edges_directory / _HANDOVERS
    # Each piece from band k enters at the lowest boundary it reaches and can
    # finish once the highest boundary it reaches has been swept.
    arrivals: dict[int, list[tuple[Handover, np.ndarray]]] = {}
    for band in plan["bands"]:
        handover = _read_handover(handover_root / f"band-{band}", settings)
        if handover.n_pieces == 0:
            continue
        below, above = handover.touches[:, 0], handover.touches[:, 1]
        enter = np.where(below, band - 1, band)
        leave = np.where(above, band, band - 1)
        for boundary in np.unique(enter):
            keep = enter == boundary
            arrivals.setdefault(int(boundary), []).append(
                (_select_pieces(settings, handover, keep), leave[keep])
            )
    open_pieces: Handover | None = None
    open_leave = np.empty(0, dtype=np.int64)
    boundary_segments = []
    for boundary in sorted(arrivals):
        parts = [(open_pieces, open_leave)] if open_pieces is not None else []
        parts += arrivals[boundary]
        current = combine_handovers([handover for handover, _ in parts])
        leave = np.concatenate([leave for _, leave in parts])
        groups = _piece_groups(settings, current)
        group_leave = np.full(groups.max(initial=-1) + 1, -(10**18), dtype=np.int64)
        np.maximum.at(group_leave, groups, leave)
        finished = group_leave[groups] <= boundary
        if finished.any():
            name = f"boundary-{boundary}"
            segment = finish_handover(
                settings, _select_pieces(settings, current, finished)
            )
            write_segment(edges_directory, name, segment)
            boundary_segments.append(name)
        if (~finished).any():
            open_pieces = _select_pieces(settings, current, ~finished)
            open_leave = leave[~finished]
            carried = sum(len(open_pieces.rows[n]) for n in settings.survey_names)
            if max_carried_rows is not None and carried > max_carried_rows:
                raise RuntimeError(
                    f"the boundary sweep carries {carried} rows past boundary "
                    f"{boundary}, more than max_carried_rows={max_carried_rows}: "
                    "groups span many bands, so the radius is too large for the "
                    "source density"
                )
        else:
            open_pieces, open_leave = None, np.empty(0, dtype=np.int64)
    if open_pieces is not None:
        raise RuntimeError("handed-over groups remained open after the last boundary")
    surveys = [
        DedupeOutcome(
            name=name,
            n_rows=settings.n_rows[name],
            dedupe_radius_arcsec=settings.dedupe_radius_arcsec[name],
            dropped_rows=np.empty(0, dtype=np.int64),
            kept_rows=np.empty(0, dtype=np.int64),
            disputed_rows=np.empty(0, dtype=np.int64),
        )
        for name in settings.survey_names
    ]
    write_metadata(
        edges_directory,
        surveys,
        [(a, b, r) for (a, b), r in settings.pair_radius_arcsec.items()],
        [f"band-{band}" for band in plan["bands"]] + boundary_segments,
        extra={"band_height_deg": plan["band_height_deg"]},
    )
    shutil.rmtree(handover_root)
    (edges_directory / _PLAN).unlink()


# --- one-machine orchestration --------------------------------------------------


def _build_band_task(arguments):
    edges_directory, band, workers = arguments
    build_band(edges_directory, band, workers=workers)


def build_edges_by_band(
    layout_directory: str | Path,
    edges_directory: str | Path,
    *,
    radius_arcsec: float,
    dedupe_radius_arcsec: Mapping[str, float],
    pair_radius_overrides: Mapping[tuple[str, str] | frozenset[str], float]
    | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    workers: int = 1,
    band_processes: int = 1,
    max_carried_rows: int | None = None,
) -> None:
    """Build candidate edges from a band layout on one machine.

    ``band_processes`` runs band tasks in that many processes; ``workers``
    sets the KD-tree threads inside each task. The result can be loaded with
    ``read_segment`` / ``read_edges`` like any saved edges.
    """
    bands = prepare_band_build(
        layout_directory,
        edges_directory,
        radius_arcsec=radius_arcsec,
        dedupe_radius_arcsec=dedupe_radius_arcsec,
        pair_radius_overrides=pair_radius_overrides,
        pairs=pairs,
    )
    tasks = [(str(edges_directory), band, workers) for band in bands]
    if band_processes > 1:
        with ProcessPoolExecutor(band_processes) as pool:
            list(pool.map(_build_band_task, tasks))
    else:
        for task in tasks:
            _build_band_task(task)
    sweep_boundaries(edges_directory, max_carried_rows=max_carried_rows)


def as_chunks(
    ra: np.ndarray, dec: np.ndarray, chunk_rows: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Split in-memory coordinates into ``(ra, dec)`` chunks for the layout."""
    for start in range(0, max(len(ra), 1), chunk_rows):
        yield ra[start : start + chunk_rows], dec[start : start + chunk_rows]
