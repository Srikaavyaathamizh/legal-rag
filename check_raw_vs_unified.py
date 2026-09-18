import csv
import json

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

FILES = {
    "IPC": "data/raw/IPC_dataset.csv",
    "BNS": "data/raw/BNS_dataset.csv",
}


def load_unified():
    with open(
        "data/processed/unified_sections.json",
        encoding="utf-8"
    ) as f:
        return json.load(f)["sections"]


def find_unified(sections, law, section):
    for record in sections:
        if (
            record.get("law") == law
            and str(record.get("section")).strip() == section
        ):
            return record

    return None


def find_raw(law, section):
    path = FILES[law]

    with open(
        path,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        print("RAW CSV COLUMNS:")
        print(reader.fieldnames)

        for row in reader:

            # Check every column for the section number.
            for value in row.values():

                if value is None:
                    continue

                if str(value).strip() == section:
                    return row

    return None


def print_record(label, record):

    print()
    print(label)
    print("-" * 80)

    if record is None:
        print("NOT FOUND")
        return

    for key, value in record.items():

        if value is None:
            value = ""

        text = str(value)

        if len(text) > 700:
            text = text[:700] + "..."

        print(f"{key}: {text}")


def main():

    sections = load_unified()

    print("=" * 100)
    print("RAW CSV vs UNIFIED CORPUS CHECK")
    print("=" * 100)

    for law, section in sorted(TARGETS):

        print()
        print("=" * 100)
        print(f"{law} {section}")
        print("=" * 100)

        unified = find_unified(
            sections,
            law,
            section
        )

        print_record(
            "UNIFIED SECTIONS.JSON",
            unified
        )

        raw = find_raw(
            law,
            section
        )

        print_record(
            f"RAW {law}_dataset.csv",
            raw
        )

    print()
    print("=" * 100)
    print("CHECK COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()