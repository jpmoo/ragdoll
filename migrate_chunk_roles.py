#!/usr/bin/env python3
"""
Migrate existing chunk_role values in all RAG group databases to the new role set.

Mapping:
  definition                 -> description
  framework explanation       -> description
  diagnostic guidance         -> application
  action/strategy             -> application
  example/application         -> application
  implications/consequences  -> implication

Run from project root (so ragdoll_ingest is importable). Uses RAGDOLL env/config for DATA_DIR.
"""

from __future__ import annotations

import sys

from ragdoll_ingest.storage import _connect, _list_sync_groups, init_db

# (old_role, new_role) in the order we want to apply (no conflicts)
ROLE_MIGRATION = [
    ("definition", "description"),
    ("framework explanation", "description"),
    ("diagnostic guidance", "application"),
    ("action/strategy", "application"),
    ("example/application", "application"),
    ("implications/consequences", "implication"),
]


def migrate_group(group: str) -> dict[str, int]:
    """Update chunk_role in one group's DB. Returns dict of old_role -> count updated."""
    conn = _connect(group)
    try:
        init_db(conn)
        # Check if chunk_role column exists
        try:
            conn.execute("SELECT chunk_role FROM chunks LIMIT 1")
        except Exception:
            return {}
        counts: dict[str, int] = {}
        for old_role, new_role in ROLE_MIGRATION:
            cur = conn.execute(
                "UPDATE chunks SET chunk_role = ? WHERE chunk_role = ?",
                (new_role, old_role),
            )
            n = cur.rowcount
            if n:
                counts[old_role] = n
        conn.commit()
        return counts
    finally:
        conn.close()


def main() -> int:
    groups = _list_sync_groups()
    if not groups:
        print("No RAG groups found (no ragdoll.db under DATA_DIR).")
        return 0
    total_updated = 0
    for group in sorted(groups):
        counts = migrate_group(group)
        if counts:
            print(f"Group {group}:")
            for old_role, new_role in ROLE_MIGRATION:
                if old_role in counts:
                    n = counts[old_role]
                    total_updated += n
                    print(f"  {old_role!r} -> {new_role!r}: {n} chunks")
    if total_updated:
        print(f"Total chunks updated: {total_updated}")
    else:
        print("No chunk_role values needed migration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
