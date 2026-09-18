import sys
import os
import json
from datetime import date

# Add src folder to Python path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "src")
)

from temporal_filter import (
    resolve_applicable_law,
    get_candidate_sections,
    get_current_equivalent,
    build_lookup_index,
)


# Load unified sections JSON using UTF-8
data_file = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "processed",
    "unified_sections.json"
)

with open(data_file, encoding="utf-8") as f:
    data = json.load(f)


sections = data["sections"]
lookup = build_lookup_index(sections)


print("=" * 70)
print("TEST 1 — Pre-cutover date -> should resolve IPC")

r = resolve_applicable_law(date(2023, 12, 10))
print(r)


print("=" * 70)
print("TEST 2 — Post-cutover date -> should resolve BNS")

r = resolve_applicable_law(date(2024, 8, 1))
print(r)


print("=" * 70)
print("TEST 3 — Exact boundary date (2024-07-01) -> should resolve BNS, no gap/overlap bug")

r = resolve_applicable_law(date(2024, 7, 1))
print(r)


print("=" * 70)
print("TEST 4 — Day before boundary (2024-06-30) -> should resolve IPC")

r = resolve_applicable_law(date(2024, 6, 30))
print(r)


print("=" * 70)
print("TEST 5 — No date provided -> should be AMBIGUOUS with guidance")

r = resolve_applicable_law(None)
print(r)


print("=" * 70)
print("TEST 6 — Future date -> resolves BNS but with a warning")

r = resolve_applicable_law(date(2030, 1, 1))
print(r)


print("=" * 70)
print("TEST 7 — Candidate filtering for IPC date: count + repealed_in_ipc warning surfaced")

candidates, res = get_candidate_sections(
    sections,
    date(2023, 5, 1)
)

print("law:", res.law, "| candidate count:", len(candidates))
print("warnings:", res.warnings)


print("=" * 70)
print("TEST 8 — Candidate filtering for BNS date: count + no spurious warnings")

candidates, res = get_candidate_sections(
    sections,
    date(2025, 1, 1)
)

print("law:", res.law, "| candidate count:", len(candidates))
print("warnings:", res.warnings)


print("=" * 70)
print("TEST 9 — get_current_equivalent: clean 1:1 mapping (IPC 304A)")

ipc_304a = lookup[("IPC", "304A")]

print(
    get_current_equivalent(
        ipc_304a,
        lookup
    )
)


print("=" * 70)
print("TEST 10 — get_current_equivalent: no equivalent at all (IPC 15, repealed_in_ipc + REPEALED)")

ipc_15 = lookup[("IPC", "15")]

print(
    get_current_equivalent(
        ipc_15,
        lookup
    )
)


print("=" * 70)
print("TEST 11 — get_current_equivalent: many-to-one consolidation target")

# Find an IPC section whose mapped BNS section is one of the heavily reused ones
consolidated_example = next(
    s
    for s in sections
    if s["law"] == "IPC"
    and s.get("mapped_sections") == ["127"]
)

print(
    "IPC section used:",
    consolidated_example["section"],
    consolidated_example["heading"]
)

print(
    get_current_equivalent(
        consolidated_example,
        lookup
    )
)


print("=" * 70)
print("ALL TEMPORAL FILTER TESTS COMPLETED")
print("=" * 70)