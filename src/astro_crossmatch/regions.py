"""Build candidate edges one sky region at a time.

A region supplies, for every survey, the rows it owns plus margin rows: rows
owned by neighbouring regions that lie within the largest radius of the
region's boundary. ``build_region`` finds every pair among those rows and
splits the region's owned rows in two:

- rows whose connected group lies wholly inside the region, which form a
  finished segment; and
- rows whose group reaches the margin, which are handed over, together with
  the pairs needed to finish them, to be completed once neighbouring regions
  have run.

A group here is a connected piece of the graph whose edges are the
cross-survey pairs and the within-survey dedupe pairs of groups that are not
yet decided. Every row and every pair is saved or handed over by exactly one
region: a region hands over its own rows, dedupe pairs whose lower row it
owns, and cross-survey pairs whose first row it owns. ``finish_handover``
turns a set of handed-over pieces that contains every piece of its groups
into a finished segment. The result equals an in-memory build of the same
rows exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from astro_crossmatch.candidate_edges import CandidateEdges, DedupeOutcome, PairEdges
from astro_crossmatch.edges import EdgeSettings, dedupe_from_pairs
from astro_crossmatch.graph import connected_components
from astro_crossmatch.kernel import CatalogKernel, radec_to_xyz, resolve_workers

MARGIN_BELOW = -1
OWNED = 0
MARGIN_ABOVE = 1

_EMPTY_ROWS = np.empty(0, dtype=np.int64)
_EMPTY_SEPARATIONS = np.empty(0, dtype=np.float64)


@dataclass(frozen=True)
class RegionRows:
    """One survey's rows loaded for a region.

    ``rows`` are global row numbers; ``side`` is ``OWNED`` for rows the region
    owns and ``MARGIN_BELOW`` / ``MARGIN_ABOVE`` for margin rows owned by the
    neighbouring region below or above.
    """

    rows: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    side: np.ndarray


@dataclass(frozen=True)
class Handover:
    """Pieces of groups that a region could not finish, in global rows.

    ``rows[survey]`` are the region's own rows in those pieces, with
    ``pending[survey]`` marking rows whose dedupe verdict is still open and
    ``piece[survey]`` the piece each row belongs to. ``touches[p]`` records
    whether piece ``p`` reaches the boundary below and the boundary above.
    ``self_pairs[survey]`` are the undecided dedupe pairs whose lower row the
    region owns; ``edges[(a, b)]`` are the cross-survey pairs, as ``(row_a,
    row_b, separation_arcsec)``, whose ``a`` row the region owns.
    """

    rows: dict[str, np.ndarray]
    pending: dict[str, np.ndarray]
    piece: dict[str, np.ndarray]
    touches: np.ndarray
    self_pairs: dict[str, tuple[np.ndarray, np.ndarray]]
    edges: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]

    @property
    def n_pieces(self) -> int:
        return len(self.touches)


def _outcome_fields(settings: EdgeSettings, name: str) -> dict:
    return {
        "name": name,
        "n_rows": settings.n_rows[name],
        "dedupe_radius_arcsec": settings.dedupe_radius_arcsec[name],
    }


def build_region(
    settings: EdgeSettings, region: Mapping[str, RegionRows], *, workers: int = 1
) -> tuple[CandidateEdges, Handover]:
    """Find a region's pairs; return its finished segment and its handover.

    ``region`` maps every survey to its owned and margin rows. The margin must
    cover every row within ``settings.max_radius_arcsec`` of the region's
    boundary, and margin rows below must be at least that far from margin rows
    above.
    """
    workers = resolve_workers(workers)
    names = settings.survey_names
    loaded = {name: region[name] for name in names}
    sizes = {name: len(loaded[name].rows) for name in names}
    offsets = dict(zip(names, np.cumsum([0, *[sizes[n] for n in names]])[:-1]))
    n_nodes = sum(sizes.values())
    kernels = {
        name: CatalogKernel.from_xyz(
            radec_to_xyz(loaded[name].ra, loaded[name].dec, name=name)
        )
        for name in names
    }
    owned = {name: np.asarray(loaded[name].side) == OWNED for name in names}

    # Dedupe within each survey. A dedupe group is decided here only when the
    # region owns all of it; otherwise its rows stay pending.
    pending, inactive, verdicts, pending_pairs = {}, {}, {}, {}
    for name in names:
        n = sizes[name]
        radius = settings.dedupe_radius_arcsec[name]
        if radius > 0 and n:
            first, second, _ = kernels[name].self_pairs(radius)
        else:
            first = second = _EMPTY_ROWS
        labels = connected_components(n, first, second)
        foreign = np.bincount(
            labels, weights=~owned[name], minlength=labels.max(initial=-1) + 1
        )
        pending[name] = foreign[labels] > 0 if n else np.zeros(0, dtype=bool)
        decided_pair = ~pending[name][first]
        decided = np.flatnonzero(~pending[name])
        local = np.full(n, -1, dtype=np.int64)
        local[decided] = np.arange(len(decided))
        rows = loaded[name].rows
        verdicts[name] = dedupe_from_pairs(
            rows[decided], local[first[decided_pair]], local[second[decided_pair]]
        )
        dropped, _, disputed = verdicts[name]
        inactive[name] = np.isin(rows, np.concatenate([dropped, disputed]))
        pending_pairs[name] = (first[~decided_pair], second[~decided_pair])

    # Cross-survey pairs, leaving out rows whose dedupe verdict removed them.
    cross = {}
    for (a, b), radius in settings.pair_radius_arcsec.items():
        i, j, separation = kernels[a].all_pairs(kernels[b], radius, workers=workers)
        keep = ~inactive[a][i] & ~inactive[b][j]
        cross[(a, b)] = (i[keep], j[keep], separation[keep])

    # Connected pieces over cross-survey pairs and undecided dedupe pairs. A
    # piece is finished exactly when the region owns every row in it: a
    # pending row is joined by its undecided dedupe pairs to the margin row
    # that left its dedupe group undecided.
    first_parts, second_parts = [], []
    for (a, b), (i, j, _) in cross.items():
        first_parts.append(i + offsets[a])
        second_parts.append(j + offsets[b])
    for name, (i, j) in pending_pairs.items():
        first_parts.append(i + offsets[name])
        second_parts.append(j + offsets[name])
    labels = connected_components(
        n_nodes,
        np.concatenate(first_parts) if first_parts else _EMPTY_ROWS,
        np.concatenate(second_parts) if second_parts else _EMPTY_ROWS,
    )
    unfinished_node = (
        np.concatenate([~owned[name] for name in names])
        if n_nodes
        else np.zeros(0, dtype=bool)
    )
    n_labels = labels.max(initial=-1) + 1
    unfinished = np.bincount(labels, weights=unfinished_node, minlength=n_labels) > 0
    deferred = {
        name: unfinished[labels[offsets[name] : offsets[name] + sizes[name]]]
        for name in names
    }
    below = (
        np.bincount(
            labels,
            weights=np.concatenate([loaded[n].side == MARGIN_BELOW for n in names]),
            minlength=n_labels,
        )
        > 0
    )
    above = (
        np.bincount(
            labels,
            weights=np.concatenate([loaded[n].side == MARGIN_ABOVE for n in names]),
            minlength=n_labels,
        )
        > 0
    )

    segment = _finished_segment(settings, loaded, owned, deferred, verdicts, cross)
    handover = _handover(
        settings,
        loaded,
        owned,
        deferred,
        pending,
        pending_pairs,
        cross,
        labels,
        offsets,
        below,
        above,
    )
    return segment, handover


def _finished_segment(settings, loaded, owned, deferred, verdicts, cross):
    names = settings.survey_names
    finished = {name: owned[name] & ~deferred[name] for name in names}
    surveys = []
    for name in names:
        dropped, kept, disputed = verdicts[name]
        surveys.append(
            DedupeOutcome(
                **_outcome_fields(settings, name),
                dropped_rows=dropped,
                kept_rows=kept,
                disputed_rows=disputed,
                rows=np.sort(loaded[name].rows[finished[name]]),
            )
        )
    pairs = {}
    for (a, b), (i, j, separation) in cross.items():
        keep = finished[a][i] & finished[b][j]
        pairs[frozenset((a, b))] = _pair(
            a,
            b,
            settings.pair_radius_arcsec[(a, b)],
            loaded[a].rows[i[keep]],
            loaded[b].rows[j[keep]],
            separation[keep],
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)


def _handover(
    settings,
    loaded,
    owned,
    deferred,
    pending,
    pending_pairs,
    cross,
    labels,
    offsets,
    below,
    above,
):
    names = settings.survey_names
    handed = {name: owned[name] & deferred[name] for name in names}
    node_labels = {
        name: labels[offsets[name] : offsets[name] + len(loaded[name].rows)]
        for name in names
    }
    used = np.unique(np.concatenate([node_labels[n][handed[n]] for n in names]))
    piece_of_label = np.full(len(below), -1, dtype=np.int64)
    piece_of_label[used] = np.arange(len(used))
    rows, pending_out, piece = {}, {}, {}
    for name in names:
        order = np.argsort(loaded[name].rows[handed[name]], kind="stable")
        rows[name] = loaded[name].rows[handed[name]][order]
        pending_out[name] = pending[name][handed[name]][order]
        piece[name] = piece_of_label[node_labels[name][handed[name]]][order]
    self_pairs = {}
    for name, (i, j) in pending_pairs.items():
        global_rows = loaded[name].rows
        lower = np.where(global_rows[i] < global_rows[j], i, j)
        keep = owned[name][lower]
        self_pairs[name] = (global_rows[i[keep]], global_rows[j[keep]])
    edges = {}
    for (a, b), (i, j, separation) in cross.items():
        keep = handed[a][i]
        edges[(a, b)] = (
            loaded[a].rows[i[keep]],
            loaded[b].rows[j[keep]],
            separation[keep],
        )
    return Handover(
        rows=rows,
        pending=pending_out,
        piece=piece,
        touches=np.column_stack([below[used], above[used]])
        if len(used)
        else np.zeros((0, 2), dtype=bool),
        self_pairs=self_pairs,
        edges=edges,
    )


def _pair(a, b, radius, row_a, row_b, separation) -> PairEdges:
    order = np.lexsort((row_b, row_a))
    return PairEdges(
        survey_a=a,
        survey_b=b,
        radius_arcsec=radius,
        row_a=np.asarray(row_a, dtype=np.int64)[order],
        row_b=np.asarray(row_b, dtype=np.int64)[order],
        separation_arcsec=np.asarray(separation, dtype=np.float64)[order],
    )


def combine_handovers(handovers: Sequence[Handover]) -> Handover:
    """Concatenate handovers, keeping every piece distinct."""
    names = list(handovers[0].rows)
    offsets = np.cumsum([0, *[h.n_pieces for h in handovers]])[:-1]
    return Handover(
        rows={n: np.concatenate([h.rows[n] for h in handovers]) for n in names},
        pending={n: np.concatenate([h.pending[n] for h in handovers]) for n in names},
        piece={
            n: np.concatenate([h.piece[n] + o for h, o in zip(handovers, offsets)])
            for n in names
        },
        touches=np.concatenate([h.touches for h in handovers]),
        self_pairs={
            n: tuple(
                np.concatenate([h.self_pairs[n][k] for h in handovers]) for k in (0, 1)
            )
            for n in names
        },
        edges={
            key: tuple(
                np.concatenate([h.edges[key][k] for h in handovers]) for k in (0, 1, 2)
            )
            for key in handovers[0].edges
        },
    )


def finish_handover(settings: EdgeSettings, handover: Handover) -> CandidateEdges:
    """Finish handed-over pieces into one segment.

    ``handover`` must contain every piece of each of its groups. Dedupe groups
    of pending rows are decided from their pairs, and cross-survey pairs are
    kept when both rows are handed over and active.
    """
    names = settings.survey_names
    surveys, active = [], {}
    for name in names:
        rows = handover.rows[name]
        order = np.argsort(rows, kind="stable")
        rows, pending = rows[order], handover.pending[name][order]
        pending_rows = rows[pending]
        first, second = handover.self_pairs[name]
        i = _positions(pending_rows, first)
        j = _positions(pending_rows, second)
        keep = (i >= 0) & (j >= 0)
        dropped, kept, disputed = dedupe_from_pairs(pending_rows, i[keep], j[keep])
        inactive = np.concatenate([dropped, disputed])
        active[name] = np.setdiff1d(rows, inactive, assume_unique=True)
        surveys.append(
            DedupeOutcome(
                **_outcome_fields(settings, name),
                dropped_rows=dropped,
                kept_rows=kept,
                disputed_rows=disputed,
                rows=rows,
            )
        )
    pairs = {}
    for (a, b), radius in settings.pair_radius_arcsec.items():
        row_a, row_b, separation = handover.edges[(a, b)]
        keep = (_positions(active[a], row_a) >= 0) & (_positions(active[b], row_b) >= 0)
        pairs[frozenset((a, b))] = _pair(
            a, b, radius, row_a[keep], row_b[keep], separation[keep]
        )
    return CandidateEdges(surveys=tuple(surveys), pairs=pairs)


def _positions(sorted_rows: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Index of each of ``rows`` in ``sorted_rows``, or -1 when absent."""
    if len(sorted_rows) == 0:
        return np.full(len(rows), -1, dtype=np.int64)
    index = np.searchsorted(sorted_rows, rows)
    clipped = np.clip(index, 0, len(sorted_rows) - 1)
    return np.where(sorted_rows[clipped] == rows, clipped, -1)
