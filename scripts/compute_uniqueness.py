import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from coverage.numbits import numbits_to_nums


def main() -> None:
    conn = sqlite3.connect(".coverage")
    conn.row_factory = sqlite3.Row

    context_rows = conn.execute(
        "select id, context from context where context like 'test/%'"
    ).fetchall()

    line_sets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    arc_sets: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    for row in context_rows:
        context_id = row["id"]
        context_name = row["context"]

        for rec in conn.execute(
            """
            select f.path, lb.numbits
            from line_bits lb
            join file f on f.id = lb.file_id
            where lb.context_id = ? and f.path like '%/deckard/score/%'
            """,
            (context_id,),
        ):
            for line_number in numbits_to_nums(rec["numbits"]):
                line_sets[context_name].add((rec["path"], line_number))

        for rec in conn.execute(
            """
            select f.path, a.fromno, a.tono
            from arc a
            join file f on f.id = a.file_id
            where a.context_id = ? and f.path like '%/deckard/score/%'
            """,
            (context_id,),
        ):
            arc_sets[context_name].add((rec["path"], rec["fromno"], rec["tono"]))
            if rec["fromno"] > 0:
                line_sets[context_name].add((rec["path"], rec["fromno"]))
            if rec["tono"] > 0:
                line_sets[context_name].add((rec["path"], rec["tono"]))

    line_owners: dict[tuple[str, int], set[str]] = defaultdict(set)
    arc_owners: dict[tuple[str, int, int], set[str]] = defaultdict(set)

    for context_name, items in line_sets.items():
        for item in items:
            line_owners[item].add(context_name)

    for context_name, items in arc_sets.items():
        for item in items:
            arc_owners[item].add(context_name)

    rows = []
    for context_name in sorted(set(line_sets) | set(arc_sets)):
        unique_lines = sum(
            1 for item in line_sets[context_name] if len(line_owners[item]) == 1
        )
        unique_arcs = sum(
            1 for item in arc_sets[context_name] if len(arc_owners[item]) == 1
        )
        rows.append(
            {
                "context": context_name,
                "lines": len(line_sets[context_name]),
                "unique_lines": unique_lines,
                "arcs": len(arc_sets[context_name]),
                "unique_arcs": unique_arcs,
                "unique_total": unique_lines + unique_arcs,
            }
        )

    result = {
        "context_count": len(rows),
        "covered_line_items": len(line_owners),
        "covered_arc_items": len(arc_owners),
        "top_by_unique_total": sorted(
            rows, key=lambda row: (-row["unique_total"], row["context"])
        )[:15],
        "bottom_by_unique_total": sorted(
            rows, key=lambda row: (row["unique_total"], row["context"])
        )[:15],
        "zero_unique_contexts": [
            row["context"] for row in rows if row["unique_total"] == 0
        ],
    }

    Path("build/quality_controller/uniqueness_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()