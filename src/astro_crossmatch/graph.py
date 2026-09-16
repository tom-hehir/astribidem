"""Graph utilities over candidate edges: components and clique verdicts.

These are the canonical primitives the design reduces to (see
``docs/crossmatching.md``): connected components via ``scipy.sparse.csgraph``
and clique tests by edge counting. A component with ``k`` nodes is a clique
iff it has exactly ``k(k-1)/2`` edges — and because edges exist precisely
when a pair is within its radius, the edge count *is* the
all-pairs-within-radius test. Verification, not search: no clique is ever
searched for, so none of the NP-hard clique machinery is involved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def connected_components(
    n_nodes: int, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Label each node with its connected component.

    ``first``/``second`` are aligned arrays of node indices, one edge per
    entry. Returns an ``(n_nodes,)`` int array of component labels; isolated
    nodes get their own singleton components.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    if n_nodes == 0:
        return np.empty(0, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    data = np.ones(len(first), dtype=np.int8)
    adjacency = coo_matrix((data, (first, second)), shape=(n_nodes, n_nodes))
    _, labels = _cc(adjacency, directed=False)
    return labels.astype(np.int64)


def component_sizes_and_edge_counts(
    labels: np.ndarray, first: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-component node counts and edge counts.

    ``first`` is one endpoint per edge; both endpoints share a label by
    construction, so one endpoint suffices to attribute the edge.
    """
    n_components = int(labels.max()) + 1 if len(labels) else 0
    sizes = np.bincount(labels, minlength=n_components)
    edge_counts = np.bincount(labels[first], minlength=n_components)
    return sizes, edge_counts


def clique_mask(sizes: np.ndarray, edge_counts: np.ndarray) -> np.ndarray:
    """True for components whose edge count equals ``k(k-1)/2``."""
    return edge_counts == sizes * (sizes - 1) // 2


@dataclass(frozen=True)
class GlobalGraph:
    """The joint graph over every survey's active rows.

    Nodes are (survey, row) pairs numbered densely: survey ``i``'s rows occupy
    ``[offsets[i], offsets[i] + n_i)`` in survey order, in ascending
    original-position order. Everything downstream (the entity builder) reads
    components and verdicts from here.
    """

    survey_names: tuple[str, ...]
    offsets: np.ndarray
    node_row_index: np.ndarray
    node_survey: np.ndarray
    first: np.ndarray
    second: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray
    edge_counts: np.ndarray
    survey_bits: np.ndarray
    n_surveys_in: np.ndarray
    edge_sep: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_row_index)

    def clique(self) -> np.ndarray:
        return clique_mask(self.sizes, self.edge_counts)

    def clean(self) -> np.ndarray:
        """Legal-group verdict per component: clique and one row per survey."""
        return self.clique() & (self.sizes == self.n_surveys_in)


def build_global_graph(
    survey_names: tuple[str, ...],
    active_row_index: dict[str, np.ndarray],
    pair_edges: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> GlobalGraph:
    """Assemble the joint graph from per-pair edges in original positions.

    ``pair_edges`` entries are ``(survey_a, survey_b, row_a, row_b,
    sep_arcsec)``. ``active_row_index`` arrays must be sorted ascending (they
    are, by construction: positions of a boolean mask). The design caps at 63
    surveys (``survey_bits`` is a single int64 bitmask per component).
    """
    offsets_list: list[int] = []
    total = 0
    for name in survey_names:
        offsets_list.append(total)
        total += len(active_row_index[name])
    offsets = np.array(offsets_list, dtype=np.int64)

    node_row_index = (
        np.concatenate([active_row_index[name] for name in survey_names])
        if total
        else np.empty(0, dtype=np.int64)
    )
    node_survey = np.empty(total, dtype=np.int64)
    for i, name in enumerate(survey_names):
        start = offsets_list[i]
        node_survey[start : start + len(active_row_index[name])] = i

    survey_pos = {name: i for i, name in enumerate(survey_names)}
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    sep_parts: list[np.ndarray] = []
    for survey_a, survey_b, row_a, row_b, sep in pair_edges:
        rows_a = active_row_index[survey_a]
        rows_b = active_row_index[survey_b]
        first_parts.append(
            np.searchsorted(rows_a, row_a) + offsets[survey_pos[survey_a]]
        )
        second_parts.append(
            np.searchsorted(rows_b, row_b) + offsets[survey_pos[survey_b]]
        )
        sep_parts.append(np.asarray(sep, dtype=np.float64))
    first = (
        np.concatenate(first_parts).astype(np.int64)
        if first_parts
        else np.empty(0, dtype=np.int64)
    )
    second = (
        np.concatenate(second_parts).astype(np.int64)
        if second_parts
        else np.empty(0, dtype=np.int64)
    )
    edge_sep = (
        np.concatenate(sep_parts).astype(np.float64)
        if sep_parts
        else np.empty(0, dtype=np.float64)
    )

    labels = connected_components(total, first, second)
    sizes, edge_counts = component_sizes_and_edge_counts(labels, first)

    n_components = len(sizes)
    survey_bits = np.zeros(n_components, dtype=np.int64)
    if total:
        np.bitwise_or.at(survey_bits, labels, np.int64(1) << node_survey)
    n_surveys_in = np.zeros(n_components, dtype=np.int64)
    for i in range(len(survey_names)):
        n_surveys_in += (survey_bits >> i) & 1

    return GlobalGraph(
        survey_names=tuple(survey_names),
        offsets=offsets,
        node_row_index=node_row_index,
        node_survey=node_survey,
        first=first,
        second=second,
        labels=labels,
        sizes=sizes,
        edge_counts=edge_counts,
        survey_bits=survey_bits,
        n_surveys_in=n_surveys_in,
        edge_sep=edge_sep,
    )
