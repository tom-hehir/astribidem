"""The ``astribidem`` command line: audit saved edges and summarise indexes.

``astribidem audit EDGES`` prints one audit of a saved edge directory,
optionally after narrowing its match radius. ``astribidem audit-sweep EDGES
--radius R ...`` audits one saved build at each of several radii.
``astribidem summary INDEX`` prints the settings and summary an index file
records, reading only its Parquet footer. ``--save`` writes an audit to a JSON
file; the terminal always gets formatted tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from astribidem.audit.report import SIZE_HISTOGRAM_CAP, EdgeAudit, audit_edges
from astribidem.audit.sweep import restrict_radius
from astribidem.candidate_edges import CandidateEdges
from astribidem.edge_files import read_edges
from astribidem.index_table import (
    DEDUPE_KEY,
    RESOLVED_CONFIG_KEY,
    SUMMARY_KEY,
    pair_name,
)

# --- formatting -----------------------------------------------------------------


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "-"
    if isinstance(value, list):
        return "+".join(_cell(item) for item in value)
    return str(value)


def _percent(fraction: float) -> str:
    return f"{fraction:.1%}"


def _arcsec(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Aligned columns: the first left-aligned, the rest right-aligned."""
    cells = [list(headers)] + [[_cell(value) for value in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]

    def line(row: list[str]) -> str:
        parts = [row[0].ljust(widths[0])]
        parts += [
            cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)
        ]
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(cells[0]), rule, *(line(row) for row in cells[1:])])


def _section(title: str, body: str) -> str:
    return f"{title}\n{body}"


def format_audit(audit: EdgeAudit) -> str:
    """One audit as titled tables."""
    surveys = table(
        [
            "survey",
            "rows",
            "dedupe radius",
            "active",
            "dropped",
            "disputed",
            "duplicate groups",
            "largest group",
        ],
        [
            [
                s.name,
                s.n_rows,
                s.dedupe_radius_arcsec,
                s.n_active,
                s.n_dropped,
                s.n_disputed,
                s.n_duplicate_groups,
                s.largest_duplicate_group,
            ]
            for s in audit.surveys
        ],
    )
    pairs = table(
        ["pair", "radius", "edges", "sep p5", "p25", "p50", "p75", "p95"],
        [
            [
                pair_name(p.survey_a, p.survey_b),
                p.radius_arcsec,
                p.n_edges,
                *(
                    _arcsec(getattr(p.separation_arcsec, name, None))
                    for name in ("p5", "p25", "p50", "p75", "p95")
                ),
            ]
            for p in audit.pairs
        ],
    )
    matched = table(
        ["pair", "survey", "matched", "multi-candidate rows"],
        [
            row
            for p in audit.pairs
            for row in (
                [
                    pair_name(p.survey_a, p.survey_b),
                    p.survey_a,
                    _percent(p.matched_fraction_a),
                    p.multi_candidate_rows_a,
                ],
                [
                    "",
                    p.survey_b,
                    _percent(p.matched_fraction_b),
                    p.multi_candidate_rows_b,
                ],
            )
        ],
    )
    census = audit.components
    components = table(
        ["", "count"],
        [
            ["active rows", census.n_rows],
            ["components", census.n_components],
            ["largest component", census.max_component_size],
            ["clean, two or more surveys", census.n_clean_multi_survey],
            ["ambiguous", census.n_ambiguous],
            ["rows in ambiguous components", census.rows_in_ambiguous_components],
        ],
    )
    sizes = table(
        ["size", *map(str, range(1, SIZE_HISTOGRAM_CAP)), f">={SIZE_HISTOGRAM_CAP}"],
        [["components", *census.size_counts]],
    )
    combinations_table = table(
        ["surveys", "components", "clean"],
        [
            ["+".join(c.surveys), c.n_components, c.n_clean]
            for c in census.by_survey_combination
        ],
    )
    sections = [
        f"Every survey pair built: {_cell(audit.all_pairs_built)}",
        _section("Surveys (radii in arcsec)", surveys),
        _section("Pairs (radii and separations in arcsec)", pairs),
        _section("Rows with candidates", matched),
        _section("Components", components),
        _section("Component sizes", sizes),
        _section(
            "Components of two or more rows, by survey combination", combinations_table
        ),
    ]
    return "\n\n".join(sections)


def format_sweep(radii: Sequence[float], audits: Sequence[EdgeAudit]) -> str:
    """Audits at several radii as tables with one row per radius."""
    pair_rows = []
    matched_rows = []
    for index in range(len(audits[0].pairs)):
        shown = set()
        for audit in audits:
            p = audit.pairs[index]
            # A pair held at an override radius is the same at every step.
            if p.radius_arcsec in shown:
                continue
            shown.add(p.radius_arcsec)
            name = pair_name(p.survey_a, p.survey_b)
            median = p.separation_arcsec.p50 if p.separation_arcsec else None
            pair_rows.append([name, p.radius_arcsec, p.n_edges, _arcsec(median)])
            for survey, fraction, multi in (
                (p.survey_a, p.matched_fraction_a, p.multi_candidate_rows_a),
                (p.survey_b, p.matched_fraction_b, p.multi_candidate_rows_b),
            ):
                matched_rows.append(
                    [name, survey, p.radius_arcsec, _percent(fraction), multi]
                )
    component_rows = [
        [
            radius,
            a.components.n_components,
            a.components.n_clean_multi_survey,
            a.components.n_ambiguous,
            a.components.rows_in_ambiguous_components,
            a.components.max_component_size,
        ]
        for radius, a in zip(radii, audits, strict=True)
    ]
    sections = [
        _section(
            "Pairs (radii and separations in arcsec)",
            table(["pair", "radius", "edges", "median sep"], pair_rows),
        ),
        _section(
            "Rows with candidates",
            table(
                ["pair", "survey", "radius", "matched", "multi-candidate rows"],
                matched_rows,
            ),
        ),
        _section(
            "Components",
            table(
                [
                    "radius",
                    "components",
                    "clean, 2+ surveys",
                    "ambiguous",
                    "rows in ambiguous",
                    "largest",
                ],
                component_rows,
            ),
        ),
    ]
    return "\n\n".join(sections)


_NAME_KEYS = ("name", "survey", "surveys")


def _is_record_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
        and all(
            not isinstance(v, dict)
            and not (isinstance(v, list) and any(isinstance(x, dict | list) for x in v))
            for item in value
            for v in item.values()
        )
    )


def _render(value: Any, indent: int = 0) -> list[str]:
    """Nested metadata as indented key/value lines, with lists of records as tables."""
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        width = max((len(str(key)) for key in value), default=0)
        for key, item in value.items():
            if isinstance(item, dict | list) and item:
                lines.append(f"{pad}{key}")
                lines.extend(_render(item, indent + 2))
            else:
                lines.append(f"{pad}{str(key).ljust(width)}  {_cell(item)}")
        return lines
    if _is_record_list(value):
        keys = dict.fromkeys(key for item in value for key in item)
        # Recorded JSON sorts its keys; lead with the column that names the row.
        headers = sorted(keys, key=lambda key: key not in _NAME_KEYS)
        rows = [[item.get(key) for key in headers] for item in value]
        return [pad + line for line in table(headers, rows).splitlines()]
    if isinstance(value, list):
        if all(not isinstance(item, dict | list) for item in value):
            return [pad + ", ".join(_cell(item) for item in value)]
        return [line for item in value for line in _render(item, indent)]
    return [pad + _cell(value)]


def format_index_summary(path: str | Path) -> str:
    """The row count and the recorded settings, dedupe outcomes and summary."""
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(path)
    recorded = metadata.schema.to_arrow_schema().metadata or {}
    sections = [
        f"Rows: {metadata.num_rows:,}",
        _section(
            "Columns",
            "\n".join(metadata.schema.to_arrow_schema().names),
        ),
    ]
    for key, title in (
        (SUMMARY_KEY, "Summary"),
        (RESOLVED_CONFIG_KEY, "Settings"),
        (DEDUPE_KEY, "Dedupe"),
    ):
        if key in recorded:
            value = json.loads(recorded[key])
            if key == DEDUPE_KEY:
                value = [{"survey": name, **entry} for name, entry in value.items()]
            sections.append(_section(title, "\n".join(_render(value))))
    if len(sections) == 2:
        sections.append("No astribidem settings are recorded in this file.")
    return "\n\n".join(sections)


# --- commands -------------------------------------------------------------------


def _radius_override(text: str) -> tuple[tuple[str, str], float]:
    pair, sep, radius = text.partition("=")
    surveys = tuple(pair.split("+"))
    if not sep or len(surveys) != 2:
        raise argparse.ArgumentTypeError(f"expected SURVEY+SURVEY=RADIUS, got {text!r}")
    try:
        return surveys, float(radius)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"radius must be a number of arcseconds, got {radius!r}"
        ) from None


def _restricted(
    edges: CandidateEdges,
    radius: float | None,
    overrides: list[tuple[tuple[str, str], float]],
) -> CandidateEdges:
    """The edges at ``radius`` and ``overrides``; other pairs keep their built radius."""
    if radius is None and not overrides:
        return edges
    radii = {
        key: pair.radius_arcsec if radius is None else radius
        for key, pair in edges.pairs.items()
    }
    for surveys, value in overrides:
        radii[frozenset(surveys)] = value
    # Every built pair is named, so the default only meets restrict_radius's checks.
    default = radius if radius is not None else max(radii.values(), default=1.0)
    return restrict_radius(edges, default, radii)


def _save(path: str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def _audit(args: argparse.Namespace) -> str:
    edges = _restricted(read_edges(args.edges), args.radius, args.radius_override)
    audit = audit_edges(edges)
    if args.save:
        _save(args.save, dataclasses.asdict(audit))
    return f"Candidate edges: {args.edges}\n\n{format_audit(audit)}"


def _audit_sweep(args: argparse.Namespace) -> str:
    edges = read_edges(args.edges)
    radii = sorted(args.radius)
    audits = [
        audit_edges(_restricted(edges, radius, args.radius_override))
        for radius in radii
    ]
    if args.save:
        _save(args.save, [dataclasses.asdict(audit) for audit in audits])
    return f"Candidate edges: {args.edges}\n\n{format_sweep(radii, audits)}"


def _summary(args: argparse.Namespace) -> str:
    return f"Index: {args.index}\n\n{format_index_summary(args.index)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astribidem",
        description="Audit saved candidate edges and summarise index files.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    override_help = (
        "hold one pair at its own radius in arcsec, as SURVEY+SURVEY=RADIUS; "
        "repeat for more pairs"
    )

    audit = commands.add_parser(
        "audit",
        help="audit one saved edge directory",
        description="Audit a saved edge directory, optionally at a smaller radius.",
    )
    audit.add_argument("edges", help="directory written by write_edges or a build")
    audit.add_argument(
        "--radius",
        type=float,
        help="match radius in arcsec for every pair (default: the built radii)",
    )
    audit.add_argument(
        "--radius-override",
        type=_radius_override,
        action="append",
        default=[],
        metavar="A+B=RADIUS",
        help=override_help,
    )
    audit.add_argument("--save", metavar="PATH", help="also write the audit as JSON")
    audit.set_defaults(run=_audit)

    sweep = commands.add_parser(
        "audit-sweep",
        help="audit one saved edge directory at several radii",
        description=(
            "Audit a saved edge directory at each of several match radii, each "
            "no larger than the built radius, without searching again."
        ),
    )
    sweep.add_argument("edges", help="directory written by write_edges or a build")
    sweep.add_argument(
        "--radius",
        type=float,
        nargs="+",
        required=True,
        help="match radii in arcsec, each applied to every pair",
    )
    sweep.add_argument(
        "--radius-override",
        type=_radius_override,
        action="append",
        default=[],
        metavar="A+B=RADIUS",
        help=override_help + "; the pair keeps that radius at every step",
    )
    sweep.add_argument(
        "--save", metavar="PATH", help="also write the audits as a JSON list"
    )
    sweep.set_defaults(run=_audit_sweep)

    summary = commands.add_parser(
        "summary",
        help="print the settings and summary an index file records",
        description=(
            "Print an index file's row count, columns, and recorded settings, "
            "dedupe outcomes and summary. Reads only the Parquet footer."
        ),
    )
    summary.add_argument("index", help="Parquet index file")
    summary.set_defaults(run=_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = args.run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"astribidem {args.command}: error: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
