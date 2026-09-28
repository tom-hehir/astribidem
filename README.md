# astribidem

Astronomical catalogue matching by sky position or shared identifiers.
Results are Arrow tables of matched row positions, which you can map to your
own catalogue IDs.

## Match coordinate arrays

Pass each catalogue as a pair of `(ra, dec)` arrays in degrees, in the same
celestial frame and epoch.

```python
from astribidem import DegenerateCrossmatchConfig, crossmatch, rows_to_ids

sources = {
    "a": ([10.0, 20.0], [0.0, 0.0]),
    "b": ([10.0001, 30.0], [0.0, 0.0]),
}

# Match pairs that are each other's nearest neighbours within one arcsecond.
rows = crossmatch(
    sources,
    radius_arcsec=1.0,
    dedupe_radius_arcsec=0.0,  # Disable within-catalogue deduplication.
    mode=DegenerateCrossmatchConfig(surveys=["a", "b"], policy="mutual_nearest"),
)

# Contents of `rows`: the matched input row positions and their separation.
# Separations are rounded here for display.
# a/row_index | b/row_index | a__b/separation_arcsec
#           0 |           0 |                   0.36

# Replace row positions with your own IDs, supplied in the original row order.
index = rows_to_ids(rows, {"a": [101, 102], "b": [201, 202]})

# Contents of `index`: the same match, now labelled with catalogue IDs.
# a/id | b/id | a__b/separation_arcsec
#  101 |  201 |                   0.36
```

See the [matching guide](https://github.com/tom-hehir/astribidem/blob/main/docs/matching-guide.md)
for other policies, multiple catalogues and per-catalogue radius settings.

## Match shared identifiers

When catalogues share an identifier system, match their unique IDs directly:

```python
from astribidem import match_uids

uids = {
    "images": ["object-B", "object-A"],
    "spectra": ["object-A", "object-C"],
}
matches = match_uids(uids, join="outer")  # Keep every identifier from either catalogue.

# Contents of `matches`: input row positions; null means no match in that catalogue.
# entity_id | images/row_index | spectra/row_index
#         0 |                0 |              null
#         1 |                1 |                 0
#         2 |             null |                 1
```

Use `join="inner"` to keep only shared identifiers. See
[UID matching](https://github.com/tom-hehir/astribidem/blob/main/docs/uid-matching.md)
for join options and mapping the results to IDs.

## Save an index

Results are ordinary Arrow tables, so they can be saved as Parquet:

```python
import pyarrow.parquet as pq

pq.write_table(index, "index.parquet")
index = pq.read_table("index.parquet")
```

See the [documentation](https://github.com/tom-hehir/astribidem/blob/main/docs/README.md)
for mixed UID/coordinate matching, larger-than-memory workflows, API details
and development instructions.
