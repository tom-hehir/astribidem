"""Mode builders: one invocation emits exactly one index product.

Two kinds exist. The **degenerate** crossmatch (`DegenerateCrossmatchConfig`)
is, for a declared list of surveys, an independent inner join under a
symmetric policy. "Degenerate" is the physics sense of *non-unique*: a
catalog row may hold multiple group memberships across products (it appears
in the {A,B}, {A,C}, and {A,B,C} products at once), and this mode does not
resolve that. The **entitywise** crossmatch (`EntitywiseCrossmatchConfig`)
lifts the degeneracy — the exactly-once outer-join over all configured
surveys, with absent rows represented by nulls and disputed rows handled by
the configured disposition.

Index columns carry **row positions** (``<survey>/row_index``): row ``i`` of a
survey is the ``i``-th coordinate the caller passed in, and a null marks an
absent survey. ``astro_crossmatch.rows_to_ids`` maps rows to caller IDs.

Pass a mode instance to ``astro_crossmatch.crossmatch``. Mode configuration
is recorded in the resulting Arrow schema metadata.
"""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pyarrow as pa

from astro_crossmatch.candidate_edges import CandidateEdges
from astro_crossmatch.graph import build_global_graph

SUBSET_JOIN_POLICIES = (
    "mutual_nearest",
    "mutual_unique",
    "anchored_nearest",
    "anchored_unique",
)
ENTITY_RESOLVERS = ("refuse", "sequential", "split")
ENTITY_DISPOSITIONS = ("singleton", "drop")
ENTITY_DEFAULT_SIZE_CAP = 30

_MISSING = -1  # Internal NumPy position marker; never serialized.
_REASON_CLEAN = None
_REASON_AMBIGUOUS = "ambiguous_component"
_REASON_DEDUPE = "dedupe_disputed"


def _json_bytes(value: Any) -> bytes:
    """Encode structured Arrow metadata as compact, deterministic JSON."""
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True)
class ModeResult:
    """One built index plus its summary."""

    table: pa.Table
    summary: dict[str, Any]


class CrossmatchModeConfig(ABC):
    """A mode selection; instantiated from the ``mode:`` class_path block."""

    @abstractmethod
    def required_surveys(self) -> tuple[str, ...]: ...

    @abstractmethod
    def requires_all_surveys(self) -> bool:
        """Entity semantics depend on every configured survey; joins do not."""

    @abstractmethod
    def build(self, edges: CandidateEdges) -> ModeResult: ...


def mode_to_mapping(mode: CrossmatchModeConfig) -> dict[str, Any]:
    """Serialize a mode as its jsonargparse class_path/init_args mapping."""
    init_args = json.loads(json.dumps(dataclasses.asdict(mode), default=list))
    return {
        "class_path": f"{type(mode).__module__}.{type(mode).__qualname__}",
        "init_args": init_args,
    }


def _row_column(survey: str) -> str:
    """The output column carrying a survey's row positions."""
    return f"{survey}/row_index"


def _row_array(positions: np.ndarray, missing: np.ndarray | None = None) -> pa.Array:
    """Row positions -> int64 Arrow array; ``missing`` marks null slots."""
    return pa.array(positions, pa.int64(), mask=missing)


# --- one-to-one policy reductions over a pair's edges -----------------------


def _sorted_lookup(
    keys: np.ndarray, values: np.ndarray, queries: np.ndarray
) -> np.ndarray:
    """Map queries through a sorted (keys -> values) table; misses -> -1."""
    if len(keys) == 0 or len(queries) == 0:
        return np.full(len(queries), -1, dtype=np.int64)
    pos = np.searchsorted(keys, queries)
    pos_clipped = np.clip(pos, 0, len(keys) - 1)
    hit = keys[pos_clipped] == queries
    out = np.full(len(queries), -1, dtype=np.int64)
    out[hit] = values[pos_clipped[hit]]
    return out


def _nearest_per_row(
    row_x: np.ndarray, row_y: np.ndarray, sep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct x-row: its nearest y-row and the separation (x sorted)."""
    order = np.lexsort((sep, row_x))
    rx, ry, rs = row_x[order], row_y[order], sep[order]
    firsts = np.ones(len(rx), dtype=bool)
    firsts[1:] = rx[1:] != rx[:-1]
    return rx[firsts], ry[firsts], rs[firsts]


def one_to_one_pairs(
    row_a: np.ndarray,
    row_b: np.ndarray,
    sep_arcsec: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a pair's candidate edges under a policy.

    Returns matched ``(row_a, row_b, sep_arcsec)``, sorted by ``row_a``. For
    the mutual policies the result is one-to-one; the anchored legacy policies
    are one-directional — a b-row may be claimed by several a-rows.
    """
    row_a = np.asarray(row_a, dtype=np.int64)
    row_b = np.asarray(row_b, dtype=np.int64)
    sep_arcsec = np.asarray(sep_arcsec, dtype=np.float64)

    if policy == "anchored_nearest":
        return _nearest_per_row(row_a, row_b, sep_arcsec)

    if policy == "anchored_unique":
        # Anchor rows with exactly one in-radius candidate; no reverse check,
        # so (like anchored_nearest) a b-row may be claimed by several anchors.
        ua, counts_a = np.unique(row_a, return_counts=True)
        keep = np.isin(row_a, ua[counts_a == 1])
        order = np.argsort(row_a[keep])
        return row_a[keep][order], row_b[keep][order], sep_arcsec[keep][order]

    if policy == "mutual_nearest":
        ka, va, sa = _nearest_per_row(row_a, row_b, sep_arcsec)
        kb, vb, _ = _nearest_per_row(row_b, row_a, sep_arcsec)
        mutual = _sorted_lookup(kb, vb, va) == ka
        return ka[mutual], va[mutual], sa[mutual]

    if policy == "mutual_unique":
        ua, counts_a = np.unique(row_a, return_counts=True)
        ub, counts_b = np.unique(row_b, return_counts=True)
        keep = np.isin(row_a, ua[counts_a == 1]) & np.isin(row_b, ub[counts_b == 1])
        order = np.argsort(row_a[keep])
        return row_a[keep][order], row_b[keep][order], sep_arcsec[keep][order]

    raise ValueError(
        f"unknown subset-join policy {policy!r}; supported: "
        f"{list(SUBSET_JOIN_POLICIES)}"
    )


# --- subset join -------------------------------------------------------------


@dataclass(frozen=True)
class DegenerateCrossmatchConfig(CrossmatchModeConfig):
    """One declared-subset inner join under a pairwise policy.

    Named for the physics sense of *degenerate* = non-unique: a catalog row
    may hold multiple, non-unique group memberships across products, and this
    mode does not resolve that (it reasons only within the declared subset,
    per its policy, and permits the same row to recur across products). The
    entitywise mode lifts this degeneracy into an exactly-once assignment.
    """

    surveys: list[str] = field(default_factory=list)
    policy: str = "mutual_nearest"

    def __post_init__(self) -> None:
        object.__setattr__(self, "surveys", [str(s) for s in self.surveys])
        if len(self.surveys) < 2:
            raise ValueError(
                f"subset join needs at least two surveys, got {self.surveys!r}"
            )
        if len(set(self.surveys)) != len(self.surveys):
            raise ValueError(f"subset join lists a survey twice: {self.surveys!r}")
        if self.policy not in SUBSET_JOIN_POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; supported: "
                f"{list(SUBSET_JOIN_POLICIES)}"
            )
        if self.policy.startswith("anchored_") and len(self.surveys) != 2:
            raise ValueError(
                f"{self.policy} is a two-survey legacy policy; got "
                f"{len(self.surveys)} surveys. Use a mutual policy for larger "
                "subsets."
            )

    def required_surveys(self) -> tuple[str, ...]:
        return tuple(self.surveys)

    def requires_all_surveys(self) -> bool:
        return False

    def build(self, edges: CandidateEdges) -> ModeResult:
        matched: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for a, b in combinations(self.surveys, 2):
            row_a, row_b, sep = edges.oriented_edges(a, b)
            matched[(a, b)] = one_to_one_pairs(row_a, row_b, sep, self.policy)

        anchor = self.surveys[0]
        rows: dict[str, np.ndarray] = {}
        base_a, base_b, _ = matched[(anchor, self.surveys[1])]
        rows[anchor] = base_a
        rows[self.surveys[1]] = base_b

        # Attach each further survey through its pair with the anchor, then
        # demand every remaining pairwise relation agrees (all-pairs within
        # the subset).
        for name in self.surveys[2:]:
            ka, va, _ = matched[(anchor, name)]
            partner = _sorted_lookup(ka, va, rows[anchor])
            keep = partner >= 0
            rows = {s: r[keep] for s, r in rows.items()}
            rows[name] = partner[keep]
        for i, a in enumerate(self.surveys[1:], start=1):
            for b in self.surveys[i + 1 :]:
                ka, va, _ = matched[(a, b)]
                keep = _sorted_lookup(ka, va, rows[a]) == rows[b]
                rows = {s: r[keep] for s, r in rows.items()}

        columns: dict[str, pa.Array] = {
            _row_column(name): _row_array(rows[name]) for name in self.surveys
        }
        for a, b in combinations(self.surveys, 2):
            ka, va, sep = matched[(a, b)]
            # Look the group's a-row up in the pair's matched table to recover
            # the separation column (anchored policies can repeat b-rows, so
            # key on the a side, which is unique for every policy).
            pos = np.searchsorted(ka, rows[a])
            columns[f"{a}__{b}/separation_arcsec"] = pa.array(
                sep[np.clip(pos, 0, max(len(sep) - 1, 0))].astype(np.float32)
                if len(sep)
                else np.zeros(0, dtype=np.float32),
                pa.float32(),
            )

        n_groups = len(rows[anchor])
        metadata = {
            b"astro_crossmatch.match_index.mode": b"degenerate",
            b"astro_crossmatch.match_index.surveys": _json_bytes(list(self.surveys)),
            b"astro_crossmatch.match_index.match_policy": self.policy.encode(),
            b"astro_crossmatch.match_index.pair_radius_arcsec": _json_bytes(
                edges.radius_metadata(combinations(self.surveys, 2))
            ),
        }
        table = pa.table(columns).replace_schema_metadata(metadata)
        summary = {
            "mode": "degenerate",
            "surveys": list(self.surveys),
            "policy": self.policy,
            "n_groups": n_groups,
        }
        return ModeResult(table=table, summary=summary)


# --- ambiguous-component resolvers (R3a / R3b) -------------------------------
#
# Both operate on ONE ambiguous component at a time and return a partition of
# its nodes into legal cliques (each: <=1 row per survey, all pairs within
# radius). They run only on the ambiguous residual, so a per-component Python
# loop is fine. `neighbors[u]` is {v: sep_arcsec} restricted to the component
# (edges exist exactly when a pair is within radius; same-survey pairs never
# have edges). Node ids are globally unique, giving a deterministic tie-break.


def _resolve_split_component(
    nodes: list[int],
    node_survey: np.ndarray,
    neighbors: dict[int, dict[int, float]],
) -> list[list[int]]:
    """R3b: greedy complete-linkage agglomeration under the group laws.

    Merge the two groups whose farthest cross-pair separation is smallest,
    vetoing any merge that would repeat a survey or leave a non-edge cross-pair
    (which keeps every group an internal clique). Ties break on the merged
    node-id set. Order-free.
    """
    groups: list[list[int]] = [[n] for n in nodes]
    surveys: list[set[int]] = [{int(node_survey[n])} for n in nodes]

    def complete_linkage(gi: int, gj: int) -> float | None:
        if surveys[gi] & surveys[gj]:
            return None
        worst = 0.0
        for u in groups[gi]:
            for v in groups[gj]:
                w = neighbors.get(u, {}).get(v)
                if w is None:
                    return None
                worst = max(worst, w)
        return worst

    while True:
        best: tuple[float, tuple[int, ...], int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                dist = complete_linkage(i, j)
                if dist is None:
                    continue
                key = tuple(sorted(groups[i] + groups[j]))
                cand = (dist, key, i, j)
                if best is None or cand < best:
                    best = cand
        if best is None:
            break
        _, _, i, j = best
        groups[i] = groups[i] + groups[j]
        surveys[i] = surveys[i] | surveys[j]
        del groups[j]
        del surveys[j]
    return groups


def _resolve_sequential_component(
    nodes: list[int],
    node_survey: np.ndarray,
    neighbors: dict[int, dict[int, float]],
    priority_rank: dict[int, int],
) -> list[list[int]]:
    """R3a: sequential priority with the all-members attach law.

    Surveys are processed in declared priority order. Each survey's rows
    greedily attach to the nearest existing entity they are within radius of
    *every* current member of (else found a new entity). Within a survey step,
    candidate (row, entity) pairs are ordered by row-to-farthest-member
    separation; a claimed row or entity drops out. Contested cases are decided
    by the declared order.
    """
    by_survey: dict[int, list[int]] = {}
    for n in nodes:
        by_survey.setdefault(int(node_survey[n]), []).append(n)

    groups: list[list[int]] = []
    for survey in sorted(by_survey, key=lambda s: priority_rank[s]):
        rows = by_survey[survey]
        if not groups:
            groups = [[r] for r in rows]
            continue
        candidates: list[tuple[float, int, int]] = []
        for r in rows:
            for gi, group in enumerate(groups):
                seps = [neighbors.get(r, {}).get(m) for m in group]
                if any(s is None for s in seps):
                    continue  # not within radius of every member
                candidates.append((max(seps), r, gi))
        candidates.sort()
        claimed_rows: set[int] = set()
        claimed_groups: set[int] = set()
        for _, r, gi in candidates:
            if r in claimed_rows or gi in claimed_groups:
                continue
            groups[gi].append(r)
            claimed_rows.add(r)
            claimed_groups.add(gi)
        for r in rows:
            if r not in claimed_rows:
                groups.append([r])
    return groups


# --- entity index ------------------------------------------------------------


@dataclass(frozen=True)
class EntitySelectionConfig:
    """Membership predicate for a materialized entity-index view.

    The predicate is applied only after the full configured survey graph has
    been resolved.  ``must_include_surveys`` is an all-of requirement; surveys
    not named there remain optional and are retained when present.
    """

    min_surveys: int = 1
    must_include_surveys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.min_surveys < 1:
            raise ValueError(
                f"selection min_surveys must be >= 1, got {self.min_surveys}"
            )
        names = [str(name) for name in self.must_include_surveys]
        if any(not name for name in names):
            raise ValueError("selection must_include_surveys may not contain ''")
        if len(set(names)) != len(names):
            raise ValueError(
                "selection must_include_surveys contains duplicate survey names"
            )
        # This field denotes a set. Canonicalizing it makes policy identity and
        # serialized metadata independent of declaration order.
        object.__setattr__(self, "must_include_surveys", sorted(names))

    def required_positions(self, names: tuple[str, ...]) -> tuple[int, ...]:
        if self.min_surveys > len(names):
            raise ValueError(
                f"selection min_surveys={self.min_surveys} exceeds the "
                f"{len(names)} configured surveys"
            )
        index = {name: i for i, name in enumerate(names)}
        missing = [name for name in self.must_include_surveys if name not in index]
        if missing:
            raise ValueError(
                f"selection names unknown surveys {missing}; configured surveys: "
                f"{list(names)}"
            )
        return tuple(index[name] for name in self.must_include_surveys)


@dataclass(frozen=True)
class EntitywiseCrossmatchConfig(CrossmatchModeConfig):
    """The exactly-once outer-join entity index over all configured surveys.

    ``resolver`` decides the ambiguous components (clean components always
    become entities): ``refuse`` (the conservative default) dissolves them to
    flagged singletons; ``sequential`` (R3a) resolves by a declared survey
    ``priority``; ``split`` (R3b) resolves by best-scoring constrained split.
    R3a/R3b emit resolved groups as ordinary (unflagged) entities and dissolve
    only components larger than ``size_cap``. ``disputed`` (``singleton`` /
    ``drop``) governs the flagged rows — dissolved-component members and
    dedupe-disputed rows. An optional ``selection`` materializes only entities
    with the requested survey membership after resolution; omitted surveys
    remain optional and still participate in dispute detection.
    """

    resolver: str = "refuse"
    disputed: str = "singleton"
    priority: list[str] | None = None
    size_cap: int = ENTITY_DEFAULT_SIZE_CAP
    selection: EntitySelectionConfig | None = None

    def __post_init__(self) -> None:
        if self.resolver not in ENTITY_RESOLVERS:
            raise ValueError(
                f"unknown resolver {self.resolver!r}; supported: "
                f"{list(ENTITY_RESOLVERS)}"
            )
        if self.disputed not in ENTITY_DISPOSITIONS:
            raise ValueError(
                f"unknown disputed disposition {self.disputed!r}; supported: "
                f"{list(ENTITY_DISPOSITIONS)}"
            )
        if self.size_cap < 1:
            raise ValueError(f"size_cap must be >= 1, got {self.size_cap}")
        if self.resolver == "sequential" and not self.priority:
            raise ValueError(
                "resolver 'sequential' (R3a) requires 'priority': an ordering "
                "of the configured surveys that decides contested attachments"
            )
        if self.priority is not None and self.resolver != "sequential":
            raise ValueError(
                f"'priority' applies only to the 'sequential' resolver, not "
                f"{self.resolver!r}"
            )

    def required_surveys(self) -> tuple[str, ...]:
        return ()

    def requires_all_surveys(self) -> bool:
        return True

    def _priority_rank(self, names: tuple[str, ...]) -> dict[int, int]:
        priority = self.priority or []
        if set(priority) != set(names) or len(priority) != len(names):
            raise ValueError(
                f"resolver 'sequential' priority {list(priority)} must be a "
                f"permutation of the configured surveys {list(names)}"
            )
        index = {name: i for i, name in enumerate(names)}
        return {index[name]: rank for rank, name in enumerate(priority)}

    def build(self, edges: CandidateEdges) -> ModeResult:
        names = edges.survey_names
        n_surveys = len(names)
        selection = self.selection or EntitySelectionConfig()
        required_positions = selection.required_positions(names)
        graph = build_global_graph(
            names,
            {survey.name: survey.active_rows() for survey in edges.surveys},
            [
                (p.survey_a, p.survey_b, p.row_a, p.row_b, p.separation_arcsec)
                for p in edges.pairs.values()
            ],
        )
        clean = graph.clean()
        clean_node = clean[graph.labels]
        n_components = len(graph.sizes)

        required_bits = sum(1 << i for i in required_positions)
        selected_clean = clean & (graph.n_surveys_in >= selection.min_surveys)
        if required_bits:
            selected_clean &= (graph.survey_bits & required_bits) == required_bits
        selected_clean_node = selected_clean[graph.labels]

        priority_rank = (
            self._priority_rank(names) if self.resolver == "sequential" else {}
        )

        # Clean components: one entity per component (vectorized).
        n_clean_components = int(clean.sum())
        n_selected_clean = int(selected_clean.sum())
        clean_pos = np.cumsum(selected_clean) - 1
        clean_cols: dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            slot = np.full(n_selected_clean, _MISSING, dtype=np.int64)
            node_mask = selected_clean_node & (graph.node_survey == i)
            slot[clean_pos[graph.labels[node_mask]]] = graph.node_row_index[node_mask]
            clean_cols[name] = slot

        # Ambiguous components + dedupe-disputed rows: the residual. Under the
        # default ``refuse`` resolver every ambiguous component dissolves into
        # flagged singletons — pure bookkeeping, one output row per node, so
        # the whole block is emitted with the same vectorized scatter as the
        # clean columns above. Only ``sequential`` / ``split`` decide anything
        # per component, so only they loop — and only over components small
        # enough to resolve; their oversized components dissolve inline to
        # keep row order by component label.
        amb_component = ~clean & (graph.sizes > 1)
        amb_node = ~clean_node
        order = np.argsort(graph.labels, kind="stable")
        sorted_labels = graph.labels[order]

        def singleton_block(nodes: np.ndarray) -> np.ndarray:
            block = np.full((len(nodes), n_surveys), _MISSING, dtype=np.int64)
            block[np.arange(len(nodes)), graph.node_survey[nodes]] = (
                graph.node_row_index[nodes]
            )
            return block

        # Reasons travel as int8 codes so the drop/selection filters below
        # stay vectorized; codes map back to the nullable reason strings only
        # when the column is built.
        reason_values = np.array(
            [_REASON_CLEAN, _REASON_AMBIGUOUS, _REASON_DEDUPE], dtype=object
        )
        _CLEAN, _AMBIGUOUS, _DEDUPE = np.int8(0), np.int8(1), np.int8(2)

        residual_blocks: list[np.ndarray] = []
        reason_blocks: list[np.ndarray] = []
        n_resolved_groups = 0
        if self.resolver == "refuse":
            # Ambiguous-component nodes ordered by (component label, node) —
            # the same order the per-component loop would emit.
            dissolved = order[amb_component[sorted_labels]]
            residual_blocks.append(singleton_block(dissolved))
            reason_blocks.append(np.full(len(dissolved), _AMBIGUOUS, dtype=np.int8))
        else:
            comp_start = np.searchsorted(sorted_labels, np.arange(n_components))
            comp_end = np.searchsorted(
                sorted_labels, np.arange(n_components), side="right"
            )

            edge_mask = amb_node[graph.first] & amb_node[graph.second]
            neighbors: dict[int, dict[int, float]] = {}
            for u, v, w in zip(
                graph.first[edge_mask].tolist(),
                graph.second[edge_mask].tolist(),
                graph.edge_sep[edge_mask].tolist(),
            ):
                neighbors.setdefault(u, {})[v] = w
                neighbors.setdefault(v, {})[u] = w

            def record(group_nodes: list[int]) -> np.ndarray:
                rec = np.full(n_surveys, _MISSING, dtype=np.int64)
                for node in group_nodes:
                    rec[graph.node_survey[node]] = graph.node_row_index[node]
                return rec

            rows: list[np.ndarray] = []
            reasons: list[np.int8] = []
            for label in np.flatnonzero(amb_component):
                nodes = order[comp_start[label] : comp_end[label]].tolist()
                if len(nodes) > self.size_cap:
                    for node in nodes:
                        rows.append(record([node]))
                        reasons.append(_AMBIGUOUS)
                    continue
                if self.resolver == "sequential":
                    groups = _resolve_sequential_component(
                        nodes, graph.node_survey, neighbors, priority_rank
                    )
                else:
                    groups = _resolve_split_component(
                        nodes, graph.node_survey, neighbors
                    )
                for group in groups:
                    rows.append(record(group))
                    reasons.append(_CLEAN)
                    n_resolved_groups += 1
            if rows:
                residual_blocks.append(np.stack(rows))
                reason_blocks.append(np.array(reasons, dtype=np.int8))

        for i, name in enumerate(names):
            disputed_rows = edges.survey(name).disputed_rows
            block = np.full((len(disputed_rows), n_surveys), _MISSING, dtype=np.int64)
            block[:, i] = disputed_rows
            residual_blocks.append(block)
            reason_blocks.append(np.full(len(disputed_rows), _DEDUPE, dtype=np.int8))

        residual = (
            np.concatenate(residual_blocks)
            if residual_blocks
            else np.empty((0, n_surveys), dtype=np.int64)
        )
        reason_codes = (
            np.concatenate(reason_blocks)
            if reason_blocks
            else np.empty(0, dtype=np.int8)
        )

        if self.disputed == "drop":
            keep = reason_codes == _CLEAN
            residual = residual[keep]
            reason_codes = reason_codes[keep]

        n_entities_before_selection = n_clean_components + len(residual)
        if self.selection is not None:
            # The membership predicate, vectorized over the residual block.
            present = residual != _MISSING
            keep = present.sum(axis=1) >= selection.min_surveys
            for pos in required_positions:
                keep &= present[:, pos]
            residual = residual[keep]
            reason_codes = reason_codes[keep]

        arrays: dict[str, pa.Array] = {}
        for i, name in enumerate(names):
            positions = np.concatenate([clean_cols[name], residual[:, i]])
            arrays[_row_column(name)] = _row_array(
                positions, missing=positions == _MISSING
            )
        arrays["disputed_reason"] = pa.array(
            [_REASON_CLEAN] * n_selected_clean + list(reason_values[reason_codes]),
            pa.string(),
        )

        n_dedupe_disputed = sum(survey.n_disputed for survey in edges.surveys)
        radius_entries = [
            {
                "surveys": [a, b],
                "radius_arcsec": edges.pairs[frozenset((a, b))].radius_arcsec,
            }
            for a, b in combinations(names, 2)
        ]
        metadata = {
            b"astro_crossmatch.entity_index.surveys": _json_bytes(list(names)),
            b"astro_crossmatch.entity_index.resolver": self.resolver.encode(),
            b"astro_crossmatch.entity_index.disputed": self.disputed.encode(),
            b"astro_crossmatch.entity_index.size_cap": str(self.size_cap).encode(),
            b"astro_crossmatch.entity_index.pair_radius_arcsec": _json_bytes(
                radius_entries
            ),
        }
        if self.priority is not None:
            metadata[b"astro_crossmatch.entity_index.priority"] = _json_bytes(
                list(self.priority)
            )
        if self.selection is not None:
            metadata[b"astro_crossmatch.entity_index.selection.min_surveys"] = str(
                self.selection.min_surveys
            ).encode()
            metadata[
                b"astro_crossmatch.entity_index.selection.must_include_surveys"
            ] = _json_bytes(list(self.selection.must_include_surveys))
        table = pa.table(arrays).replace_schema_metadata(metadata)
        summary = {
            "mode": "entitywise",
            "resolver": self.resolver,
            "disputed": self.disputed,
            "n_entities": table.num_rows,
            "n_clean_components": n_clean_components,
            "n_clean_multi_survey": int((clean & (graph.sizes > 1)).sum()),
            "n_ambiguous_components": int(amb_component.sum()),
            "n_resolved_groups": n_resolved_groups,
            "rows_in_ambiguous_components": int(amb_node.sum()),
            "n_dedupe_disputed_rows": n_dedupe_disputed,
        }
        if self.selection is not None:
            summary.update(
                {
                    "selection": {
                        "min_surveys": self.selection.min_surveys,
                        "must_include_surveys": list(
                            self.selection.must_include_surveys
                        ),
                    },
                    "n_entities_before_selection": n_entities_before_selection,
                    "n_selected_clean_components": n_selected_clean,
                }
            )
        return ModeResult(table=table, summary=summary)
