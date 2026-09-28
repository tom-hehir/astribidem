"""Resolve saved edges segment by segment into one index file.

``resolve_to_file`` resolves each segment of a saved edge directory on its own,
optionally in parallel processes, and writes the index as one Parquet file.
Segments never share a connected group of rows, so the entities are exactly
those of resolving every segment together. With ``sort=True`` (the default)
the per-segment indexes, each already in entity order, are merged into the
global entity order: first present survey, then that survey's row. The result
then equals ``resolve(read_edges(directory), mode).table``, metadata included.

Memory stays near ``batch_rows`` index rows however many segments there are.
Each segment's index is saved as a temporary Arrow IPC file of
``batch_rows / n_segments``-row batches, and the merge reads the files
through memory maps, one batch at a time; the output is written in row groups
of ``batch_rows``. A band segment's rows are scattered over the whole row
range, so reading a full batch from every segment would hold nearly the whole
index at once, and Parquet readers would hold decoding buffers for every
open segment.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.api import resolve
from astro_crossmatch.edge_files import read_segment, segment_names
from astro_crossmatch.modes import CrossmatchModeConfig

_ROW_BITS = 40  # row numbers below 2**40; survey positions fill the bits above


def _resolve_segment(arguments) -> None:
    edges_directory, name, mode, path, batch_rows = arguments
    table = resolve(read_segment(edges_directory, name), mode).table
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table, max_chunksize=batch_rows)


def _batches(reader: pa.ipc.RecordBatchFileReader):
    return (reader.get_batch(i) for i in range(reader.num_record_batches))


def _open_segment(path: Path) -> pa.ipc.RecordBatchFileReader:
    return pa.ipc.open_file(pa.memory_map(str(path)))


class _BatchWriter:
    """Collect tables and write them in row groups of ``batch_rows``."""

    def __init__(self, writer: pq.ParquetWriter, batch_rows: int):
        self.writer, self.batch_rows = writer, batch_rows
        self.tables: list[pa.Table] = []
        self.n_rows = 0

    def add(self, table: pa.Table) -> None:
        self.tables.append(table)
        self.n_rows += table.num_rows
        if self.n_rows >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if self.n_rows:
            self.writer.write_table(pa.concat_tables(self.tables))
        self.tables, self.n_rows = [], 0


def _is_count(key: str) -> bool:
    """Per-segment counts that add up; ``n_rows`` is each survey's total."""
    return key != "n_rows" and key.startswith(("n_", "rows_in_"))


def _merged(values: list, key: str | None = None):
    """Add counts across segments; every other value must agree."""
    first = values[0]
    if isinstance(first, dict):
        return {k: _merged([v[k] for v in values], k) for k in first}
    if key is not None and _is_count(key):
        return sum(values)
    if any(value != first for value in values):
        raise ValueError(f"segments disagree on {key!r}: {first!r}")
    return first


def _merged_metadata(metadatas: list[dict[bytes, bytes]]) -> dict[bytes, bytes]:
    merged = dict(metadatas[0])
    for key in (b"astro_crossmatch.dedupe", b"astro_crossmatch.summary"):
        combined = _merged([json.loads(metadata[key]) for metadata in metadatas])
        merged[key] = json.dumps(
            combined, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    return merged


def _entity_keys(batch: pa.RecordBatch | pa.Table, row_columns: list[str]):
    """One sortable integer per entity: first present survey, then its row."""
    keys = np.full(batch.num_rows, -1, dtype=np.int64)
    for position, column in enumerate(row_columns):
        rows = batch[column]
        present = ~np.asarray(rows.is_null())
        if hasattr(rows, "combine_chunks"):
            rows = rows.combine_chunks()
        values = rows.fill_null(0).to_numpy()
        take = present & (keys < 0)
        keys[take] = (position << _ROW_BITS) + values[take]
    return keys


def _merge_sorted(paths: list[Path], output: _BatchWriter):
    """K-way merge of per-segment indexes that are each sorted by entity key."""
    readers = [_open_segment(path) for path in paths]
    schema = readers[0].schema
    row_columns = [name for name in schema.names if name.endswith("/row_index")]
    streams = [_batches(reader) for reader in readers]
    buffers: list[pa.Table | None] = [None] * len(streams)
    keys: list[np.ndarray | None] = [None] * len(streams)

    def refill(i):
        for batch in streams[i]:
            if batch.num_rows:
                buffers[i] = pa.Table.from_batches([batch])
                keys[i] = _entity_keys(buffers[i], row_columns)
                return
        buffers[i], keys[i] = None, None

    for i in range(len(streams)):
        refill(i)
    while any(buffer is not None for buffer in buffers):
        active = [i for i, buffer in enumerate(buffers) if buffer is not None]
        # Every buffered key up to the smallest buffer tail is final: no
        # segment can later produce a smaller key.
        threshold = min(keys[i][-1] for i in active)
        tables, table_keys = [], []
        for i in active:
            count = int(np.searchsorted(keys[i], threshold, side="right"))
            if count:
                tables.append(buffers[i].slice(0, count))
                table_keys.append(keys[i][:count])
                buffers[i] = buffers[i].slice(count)
                keys[i] = keys[i][count:]
            if buffers[i].num_rows == 0:
                refill(i)
        chunk = pa.concat_tables(tables)
        order = np.argsort(np.concatenate(table_keys), kind="stable")
        output.add(chunk.take(order))


def resolve_to_file(
    edges_directory: str | Path,
    mode: CrossmatchModeConfig,
    index_path: str | Path,
    *,
    processes: int = 1,
    sort: bool = True,
    batch_rows: int = 1_048_576,
) -> None:
    """Resolve every segment of saved edges and write one index file.

    ``processes`` resolves segments in parallel. With ``sort=False`` the index
    lists each segment's entities in segment order instead of merging them.
    ``batch_rows`` sets how many index rows the merge holds at once, across all
    segments, and the output's row-group size.
    """
    edges_directory = Path(edges_directory)
    index_path = Path(index_path)
    names = segment_names(edges_directory)
    segment_rows = max(1, batch_rows // len(names))
    work = index_path.with_name(f".{index_path.name}.segments")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        paths = [work / f"{name}.arrow" for name in names]
        tasks = [
            (str(edges_directory), name, mode, str(path), segment_rows)
            for name, path in zip(names, paths)
        ]
        if processes > 1:
            with ProcessPoolExecutor(processes) as pool:
                list(pool.map(_resolve_segment, tasks))
        else:
            for task in tasks:
                _resolve_segment(task)
        metadatas = [_open_segment(path).schema.metadata for path in paths]
        schema = _open_segment(paths[0]).schema.with_metadata(
            _merged_metadata(metadatas)
        )
        with pq.ParquetWriter(index_path, schema) as writer:
            output = _BatchWriter(writer, batch_rows)
            if sort:
                _merge_sorted(paths, output)
            else:
                for path in paths:
                    for batch in _batches(_open_segment(path)):
                        output.add(pa.Table.from_batches([batch]))
            output.flush()
    finally:
        shutil.rmtree(work, ignore_errors=True)
