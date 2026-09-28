# Matching shared identifiers

Use `match_uids` when identifiers refer to the same objects across catalogues.
It compares keys exactly, without coordinates or a matching radius. Equal
numbers from unrelated catalogue ID systems do not establish shared identity.

## Join catalogues

```python
from astribidem import match_uids, rows_to_ids

uids = {
    "images": ["object-B", "object-A"],
    "spectra": ["object-A", "object-C"],
}
rows = match_uids(uids, join="outer")

# Contents of `rows`: original row positions; null means absent membership.
# entity_id | images/row_index | spectra/row_index
#         0 |                0 |              null
#         1 |                1 |                 0
#         2 |             null |                 1

index = rows_to_ids(rows, {"images": [101, 102], "spectra": [201, 202]})

# Contents of `index`: row positions replaced with catalogue IDs.
# entity_id | images/id | spectra/id
#         0 |       101 |       null
#         1 |       102 |        201
#         2 |      null |        202
```

| Join | Included identifiers |
| --- | --- |
| `"inner"` (default) | Present in every catalogue. |
| `"left"` | Every identifier in the anchor catalogue. |
| `"outer"` | Every identifier in any catalogue. |

The first mapping entry is the default anchor. Set `anchor="spectra"` to choose
another. Results follow anchor input order; outer joins append unseen keys in
remaining catalogue/input order. Columns follow the input mapping order.

## Inputs and results

Keys must be unique and non-null within each catalogue. All catalogues must
use integer keys or all must use string keys. Integers compare exactly across
signedness and widths, including uint64; strings are case-sensitive. Numeric
strings and integers are different key systems and cannot be mixed.

The result is an Arrow table with a consecutive `entity_id` and an int64
`<catalogue>/row_index` column for each input. `entity_id` labels this result;
it is not a persistent source identifier. Null means that catalogue has no row
for the entity. `rows_to_ids` preserves the supplied ID types and nulls; pass
one ID per original input row. To recover the shared identifiers themselves,
use `rows_to_ids(rows, uids)`.

Matching runs in memory. For a combination of identifier and coordinate links,
see [hub-and-spoke matching](hub-and-spoke-matching.md). Saved tables and ordering
are covered in [API and compatibility](api-and-compatibility.md).
