# Hub-and-spoke development history

Historical implementation, migration and validation records. See the
[usage guide](../docs/hub-and-spoke-matching.md) for supported behavior.

The independent-link composition follows AstroBench's mixed MMU workflow. Unlike
its first-link ordering, this API returns rows in anchor input order. HF scanning, persistence and payload materialization remain
responsibilities of `astribidem-hf`.

## Renaming from the original interface

`match_hub_and_spoke` replaces `match_mixed`, and `hub_and_spoke.py` replaces
`mixed.py`; there are no compatibility aliases. Update imports and calls.
Matching behavior is unchanged. The resolved-config provenance method is now
`hub_and_spoke`, so saved workflow identities using the old method are
different.

`astribidem-hf` now exposes `match_catalog_hub_and_spoke` and the
`hub_and_spoke` CLI method, with its dependency pinned to a core commit
providing the renamed API. Its
[migration guide](https://github.com/tom-hehir/astribidem-hf/blob/5de543ac020e8a89854c2b97a1e0911772b362d4/docs/cli.md#migrating-the-original-anchor-link-interface)
covers the import and configuration changes and the new work directory needed
for workflows saved under `mixed`.

## Initial implementation validation

Before this rename, the original anchor-link implementation passed the complete
core suite (299 tests), including 39 new composition regressions. Those 39 tests
and 84 UID tests also passed on minimum-supported PyArrow 15.0.0. The downstream
HF suite passed against that core (228 tests), and all 42 new HF
composition/index tests passed on PyArrow 15.0.0. These historical checks used
local fixtures; they do not establish survey-scale performance or validation of
subsequent changes.

## Related design notes

The [grouped UID/spatial proposal](design/grouped-uid-spatial-matching.md)
describes a deferred possible extension: UID grouping, coordinate-source
priority, and richer spatial resolution between representative objects without a
universal anchor. Its stricter coordinate requirement and partial membership
handling do not apply to this API. It will only be implemented when a concrete
use case needs it. Broader hub-and-spoke retention modes likewise remain future
work; this API currently supports inner results only.
