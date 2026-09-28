# Development notes

Design proposals, implementation decisions, experiments and historical records.
These notes may describe older interfaces or ideas that were never implemented;
use the [documentation](../docs/README.md) for supported behavior.

## Designs

- [Banded edge builds](design/banded-edge-builds.md): design and alternatives for
  the implemented larger-than-memory spatial workflow.
- [Grouped UID/spatial matching](design/grouped-uid-spatial-matching.md): deferred
  proposal for matching representative objects across UID groups.

## Implementation and experiments

- [UID implementation decisions](uid-matching-implementation.md): algorithm and
  dependency choices, including comparisons with other join implementations.
- [Acero evaluation](uid-acero-evaluation.md): experimental Arrow join comparison.
- [Benchmarks](../benchmarks/README.md): reproduction commands and recorded timings.

## History

- [Development provenance](provenance.md): origins and extraction of the code.
- [Hub-and-spoke history](hub-and-spoke-history.md): original interface, migration
  and implementation validation.
