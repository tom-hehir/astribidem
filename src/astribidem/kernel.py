"""Exact spatial pair-generation kernel: scipy cKDTree over unit vectors.

Coordinates become float64 Cartesian unit vectors, angular radii become chord
lengths, and every query is exact — no approximate structures, since a missed
pair is a silent duplicate entity. Adopted 2026-08-13 (AION-2 issue #188,
transplanted here per astral-projections issue #8) after a measured
head-to-head against the smatch C HEALPix engine: the dual-tree join is ~10x
faster on the production-density benchmark, chord separations avoid smatch's
~0.05" ``1 - cos`` cancellation floor, and no GPL dependency is needed.

The kernel contract:

- **Build once, match many.** One ``CatalogKernel`` per survey serves its
  dedupe self-match and every pair it participates in. Trees are built with
  ``compact_nodes=True, balanced_tree=True`` (data referenced, not copied;
  ~24 B/row overhead) — kernel policy, not configuration.
- **Boundary is inclusive.** Pairs at exactly the radius are included
  (separation <= r) by every emitting method. scipy's
  ``query(distance_upper_bound=...)`` is exclusive at the bound, so the
  k-truncated path bumps it by one ulp to match.
- **Deterministic output.** Raw dual-tree output follows traversal order, so
  every emitting method lexsorts by ``(i, j)`` before returning.
- **Exact separations.** float64 chord lengths converted through
  ``2 asin(d / 2)``; there is no cosine-distance path.

Parallelism: the dual-tree join and ``query_pairs`` are single-threaded C
routines with no ``workers`` option, but they release the GIL, so ``workers``
on ``all_pairs`` splits the query side into chunks — a throwaway tree per
chunk, joined against the resident right tree, one call per thread. Chunking
exists only to manufacture parallel calls (serially it costs ~2.5x); the
chunk count is capped by a minimum chunk size so small inputs stay one-shot.
Because chunks partition ``i`` in order and each chunk is lexsorted, the
concatenated output is bit-identical to the one-shot join.
``all_pairs_chunks`` yields the same chunks in order with at most ``workers``
in flight, so a consumer that writes each chunk out holds only
``workers * chunk_rows`` query rows' worth of pairs at a time.

Deliberately absent: a minimum-radius "too close" veto. The old matching
layer's ``min_radius_arcsec`` veto was its only guard against duplicates and
blends; here per-survey dedupe and the entitywise dispute machinery own that
concern, and the edge substrate must stay complete — pre-filtering edges can
convert a disputed component into a false clean clique.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree

DEG_TO_RAD = math.pi / 180.0
_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def float64_coordinates(
    ra, dec, *, name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec as float64 arrays: the first step for any received coordinates.

    Floating-point inputs less precise than float64 trigger a ``UserWarning``
    (naming the survey when ``name`` is given), because their rounding already
    limits positional accuracy and casting cannot restore it. Values that are
    already float64 pass through unchanged and without a warning.
    """
    converted = []
    for column, values in (("ra", ra), ("dec", dec)):
        values = np.asarray(values)
        if values.dtype.kind == "f" and values.dtype.itemsize < 8:
            where = f"survey {name!r}: " if name is not None else ""
            warnings.warn(
                f"{where}{column} is {8 * values.dtype.itemsize}-bit floating "
                "point, less precise than float64 (float32 RA is spaced up to "
                "0.11 arcsec apart near 360 deg). Matching computes in float64 "
                "but cannot restore the lost precision; pass float64 "
                "coordinates to avoid this.",
                UserWarning,
                stacklevel=3,
            )
        converted.append(values.astype(np.float64, copy=False))
    ra, dec = converted
    where = f"survey {name!r}: " if name is not None else ""
    if not (np.isfinite(ra).all() and np.isfinite(dec).all()):
        raise ValueError(f"{where}coordinates must be finite")
    if np.any((dec < -90.0) | (dec > 90.0)):
        raise ValueError(f"{where}declination must be between -90 and 90 degrees")
    return ra, dec


def radec_to_xyz(
    ra: np.ndarray, dec: np.ndarray, *, name: str | None = None
) -> np.ndarray:
    """Unit vectors on the sphere for degree-valued ``ra`` / ``dec``."""
    ra, dec = float64_coordinates(ra, dec, name=name)
    ra_rad = ra * DEG_TO_RAD
    dec_rad = dec * DEG_TO_RAD
    r_xy = np.cos(dec_rad)
    return np.stack(
        [r_xy * np.cos(ra_rad), r_xy * np.sin(ra_rad), np.sin(dec_rad)],
        axis=-1,
    )


def xyz_to_radec(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degree-valued ``(ra, dec)`` for ``(n, 3)`` unit vectors."""
    xyz = np.asarray(xyz, dtype=np.float64)
    ra = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    return ra, dec


def arcsec_to_chord(radius_arcsec: float) -> float:
    """Great-circle arcseconds to the unit-sphere chord length."""
    return 2.0 * math.sin(radius_arcsec * _ARCSEC_TO_RAD / 2.0)


def chord_to_arcsec(chord: np.ndarray) -> np.ndarray:
    """Unit-sphere chord lengths to great-circle arcseconds, exact float64."""
    half = np.clip(np.asarray(chord, dtype=np.float64) / 2.0, 0.0, 1.0)
    return 2.0 * np.arcsin(half) / _ARCSEC_TO_RAD


def _validate_radius(radius_arcsec: float) -> float:
    if not math.isfinite(radius_arcsec) or radius_arcsec < 0:
        raise ValueError(f"radius_arcsec must be finite and >= 0, got {radius_arcsec}")
    return arcsec_to_chord(radius_arcsec)


#: Chunks smaller than this are not worth their per-chunk tree build; the
#: chunk count is capped so no chunk falls below it (oversplitting makes the
#: chunked join slower than one-shot even fully threaded).
MIN_CHUNK_ROWS = 250_000


def resolve_workers(workers: int) -> int:
    """Normalize a worker count: ``-1`` means all cores, otherwise >= 1."""
    if workers == -1:
        return os.cpu_count() or 1
    if workers < 1:
        raise ValueError(f"workers must be >= 1 or -1 (all cores), got {workers}")
    return workers


def _chunk_bounds(n: int, workers: int, chunk_rows: int | None) -> np.ndarray:
    """Chunk boundaries for the query side: ~one chunk per worker, floored."""
    if chunk_rows is None:
        chunk_rows = max(MIN_CHUNK_ROWS, -(-n // max(workers, 1)))
    elif chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    n_chunks = max(1, -(-n // chunk_rows))
    return np.linspace(0, n, n_chunks + 1).astype(np.int64)


class CatalogKernel:
    """One survey's match structure: unit vectors plus their KD-tree."""

    def __init__(
        self, ra: np.ndarray, dec: np.ndarray, *, name: str | None = None
    ) -> None:
        ra_shape, dec_shape = np.shape(ra), np.shape(dec)
        if len(ra_shape) != 1 or len(dec_shape) != 1 or ra_shape != dec_shape:
            raise ValueError("ra/dec must be aligned 1-D arrays")
        self._init_from_xyz(radec_to_xyz(ra, dec, name=name))

    @classmethod
    def from_xyz(cls, xyz: np.ndarray) -> CatalogKernel:
        """Build from ``(n, 3)`` unit vectors already on the sphere."""
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (n, 3), got {xyz.shape}")
        kernel = cls.__new__(cls)
        kernel._init_from_xyz(xyz)
        return kernel

    def _init_from_xyz(self, xyz: np.ndarray) -> None:
        self.xyz = xyz
        self.tree = cKDTree(self.xyz, compact_nodes=True, balanced_tree=True)

    def __len__(self) -> int:
        return len(self.xyz)

    def self_pairs(
        self, radius_arcsec: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unique in-radius pairs ``i < j`` with identity structurally excluded.

        The dedupe primitive. Exactly-coincident rows (separation 0) are
        included, as dedupe requires.
        """
        chord = _validate_radius(radius_arcsec)
        pairs = self.tree.query_pairs(chord, output_type="ndarray")
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        sep = chord_to_arcsec(np.linalg.norm(self.xyz[i] - self.xyz[j], axis=1))
        return i, j, sep

    def all_pairs(
        self,
        other: CatalogKernel,
        radius_arcsec: float,
        k: int | None = None,
        *,
        workers: int = 1,
        chunk_rows: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """In-radius cross pairs ``(i, j, sep_arcsec)``; ``i`` indexes ``self``.

        ``k=None`` (the default) emits the complete all-pairs set via the
        dual-tree join — the load-bearing edge substrate. An explicit ``k``
        truncates to each ``i``-row's k nearest in-radius neighbours (k=1 for
        nearest policies, k=2 for uniqueness tests); truncation is only sound
        when aligned to a single known degenerate policy within the same run,
        so any persisted or shared edge product must never be built from it.

        ``workers`` (``-1`` = all cores, ``1`` = serial one-shot) parallelizes
        the ``k=None`` join by chunking this kernel's rows; output is
        bit-identical to the serial call. ``chunk_rows`` overrides the
        automatic per-worker chunk sizing (mainly for tests).
        """
        chord = _validate_radius(radius_arcsec)
        if k is None:
            parts = list(
                self._all_pairs_chunk_arrays(other, chord, workers, chunk_rows)
            )
            if len(parts) == 1:
                return parts[0]
            return tuple(  # type: ignore[return-value]
                np.concatenate([part[axis] for part in parts]) for axis in range(3)
            )

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        # query's distance_upper_bound is exclusive at the bound; one ulp up
        # makes the boundary inclusive (<= r) like the all-pairs path.
        dist, idx = other.tree.query(
            self.xyz,
            k=k,
            distance_upper_bound=np.nextafter(chord, np.inf),
            workers=resolve_workers(workers),
        )
        dist = dist.reshape(len(self.xyz), k)
        idx = idx.reshape(len(self.xyz), k)
        found = np.isfinite(dist)
        i = np.broadcast_to(
            np.arange(len(self.xyz), dtype=np.int64)[:, None], idx.shape
        )[found]
        j = idx[found].astype(np.int64)
        sep = chord_to_arcsec(dist[found])
        order = np.lexsort((j, i))
        return i[order], j[order], sep[order]

    def all_pairs_chunks(
        self,
        other: CatalogKernel,
        radius_arcsec: float,
        *,
        workers: int = 1,
        chunk_rows: int | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The complete ``all_pairs`` join as an ordered stream of chunks.

        Each chunk covers a contiguous slice of this kernel's rows, and the
        chunks concatenate to exactly ``all_pairs(other, radius_arcsec)``. At
        most ``workers`` chunks are computed or buffered at once.
        """
        chord = _validate_radius(radius_arcsec)
        yield from self._all_pairs_chunk_arrays(other, chord, workers, chunk_rows)

    def _all_pairs_chunk_arrays(
        self,
        other: CatalogKernel,
        chord: float,
        workers: int,
        chunk_rows: int | None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        workers = resolve_workers(workers)
        bounds = _chunk_bounds(len(self.xyz), workers, chunk_rows)

        def join(lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if lo == 0 and hi == len(self.xyz):
                sub_tree = self.tree  # single chunk: reuse the built tree
            else:
                sub_tree = cKDTree(
                    self.xyz[lo:hi], compact_nodes=True, balanced_tree=True
                )
            m = sub_tree.sparse_distance_matrix(
                other.tree, max_distance=chord, output_type="ndarray"
            )
            m = m[np.lexsort((m["j"], m["i"]))]
            return (
                m["i"].astype(np.int64) + lo,
                m["j"].astype(np.int64),
                chord_to_arcsec(m["v"]),
            )

        spans = list(pairwise(bounds))
        if workers == 1 or len(spans) == 1:
            for lo, hi in spans:
                yield join(lo, hi)
            return

        with ThreadPoolExecutor(workers) as pool:
            pending = deque(pool.submit(join, lo, hi) for lo, hi in spans[:workers])
            for lo, hi in spans[workers:]:
                result = pending.popleft().result()
                pending.append(pool.submit(join, lo, hi))
                yield result
            while pending:
                yield pending.popleft().result()
