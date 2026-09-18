from __future__ import annotations

import csv
import json
import os


TARGETS = {
    ("IPC", "167"),
    ("IPC", "210"),
    ("IPC", "294A"),
    ("IPC", "341"),
    ("IPC", "342"),
    ("IPC", "343"),
    ("IPC", "344"),
    ("IPC", "345"),
    ("IPC", "346"),
    ("IPC", "347"),
    ("BNS", "72"),
}

RAW_FILES = {
    "IPC": os.path.join("data", "raw", "IPC_dataset.csv"),
    "BNS": os.path.join("data", "raw", "BNS_dataset.csv"),
}

UNIFIED_FILE = os.path.join(
    "data",
    "processed",
    "unified_sections.json",
)


def norm(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).lower()


def find_section(rows, section):
    section_norm = norm(section)

    for row in rows:
        for key, value in row.items():
            if norm(value) == section_norm:
                return row

    return None


def get_text(row):
    if row is None:
        return ""

    # Prefer obvious text/body columns.
    preferred = [
        "text",
        "Text",
        "section_text",
        "Section Text",
        "description",
        "Description",
        "content",
        "Content",
        "provision",
        "Provision",
    ]

    for key in preferred:
        if key in row and row[key]:
            return str(row[key]).strip()

    # Fallback: longest textual field.
    values = [
        str(v).strip()
        for v in row.values()
        if v is not None and str(v).strip()
    ]

    return max(values, key=len) if values else ""


def get_heading(row):
    if row is None:
        return ""

    preferred = [
        "heading",
        "Heading",
        "title",
        "Title",
        "section_heading",
        "Section Heading",
    ]

    for key in preferred:
        if key in row and row[key]:
            return str(row[key]).strip()

    return ""


def load_raw(law):
    path = RAW_FILES[law]

    with open(
        path,
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def load_unified():
    with open(
        UNIFIED_FILE,
        encoding="utf-8",
    ) as f:
        return json.load(f)["sections"]


def find_unified(sections, law, section):
    for record in sections:
        if (
            record.get("law") == law
            and str(record.get("section")) == section
        ):
            return record

    return None


def main():

    print("=" * 100)
    print("RAW CSV vs UNIFIED CORPUS — DEFINITIVE INTEGRITY CHECK")
    print("=" * 100)

    unified = load_unified()

    raw_cache = {}

    for law in ["IPC", "BNS"]:
        raw_cache[law] = load_raw(law)

    problems = 0

    for law, section in sorted(
        TARGETS,
        key=lambda x: (x[0], x[1]),
    ):

        print()
        print("=" * 100)
        print(f"{law} {section}")
        print("=" * 100)

        unified_record = find_unified(
            unified,
            law,
            section,
        )

        raw_record = find_section(
            raw_cache[law],
            section,
        )

        if unified_record is None:
            print("UNIFIED: NOT FOUND")
            problems += 1
            continue

        if raw_record is None:
            print("RAW CSV: NOT FOUND")
            problems += 1
            continue

        unified_text = str(
            unified_record.get("text", "")
        ).strip()

        raw_text = get_text(raw_record)

        unified_heading = str(
            unified_record.get("heading", "")
        ).strip()

        raw_heading = get_heading(raw_record)

        print()
        print("UNIFIED HEADING:")
        print(unified_heading)

        print()
        print("RAW HEADING:")
        print(raw_heading)

        print()
        print("UNIFIED TEXT:")
        print(unified_text[:700])

        print()
        print("RAW TEXT:")
        print(raw_text[:700])

        text_same = norm(unified_text) == norm(raw_text)

        heading_same = (
            not raw_heading
            or norm(unified_heading) == norm(raw_heading)
        )

        print()
        print("-" * 100)

        if text_same:
            print("TEXT STATUS:      MATCH")
        else:
            print("TEXT STATUS:      *** MISMATCH ***")
            problems += 1

        if heading_same:
            print("HEADING STATUS:   MATCH")
        else:
            print("HEADING STATUS:   *** MISMATCH ***")

    print()
    print("=" * 100)

    if problems == 0:
        print("RESULT: ALL TARGETS MATCH RAW CSV")
        print()
        print(
            "The corruption is NOT between raw CSV and unified_sections.json."
        )
        print(
            "Next suspect: Phase-1 construction/indexing logic."
        )
    else:
        print(
            f"RESULT: {problems} mismatch/not-found condition(s) detected."
        )
        print()
        print(
            "If RAW is correct and UNIFIED is wrong, fix the Phase-1"
        )
        print(
            "CSV -> unified_sections construction."
        )
        print()
        print(
            "If RAW is also wrong, the source CSV itself is corrupted."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()