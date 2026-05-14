"""CLI tool for managing RAGDoll collections and sources."""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from . import config
from .chunk_csv import CHUNK_CSV_HEADERS
from .embedder import build_text_to_embed, embed
from .memory import MEMORY_GROUP
from .storage import (
    _connect,
    _list_sync_groups,
    add_chunks,
    delete_source_by_id,
    get_source_by_id,
    init_db,
    list_sources,
    set_source_external_url,
    unmark_processed,
)


REQUIRED_CSV_FIELDS = frozenset({"source_path", "source_type", "chunk_index", "text"})


def _strip_csv_row(d: dict[str, str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = (k or "").strip()
        if not key:
            continue
        if isinstance(v, str):
            out[key] = v.strip()
        elif v is None:
            out[key] = ""
        else:
            out[key] = str(v)
    return out


def _parse_int(val: str | None, _field: str) -> int | None:
    s = (val or "").strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_key_signals(raw: str | None) -> list[str] | str | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return s


def ensure_collection_db(group: str) -> None:
    """Create group directory, sources/, and SQLite schema if new."""
    gp = config.get_group_paths(group)
    gp.group_dir.mkdir(parents=True, exist_ok=True)
    gp.sources_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect(group)
    try:
        init_db(conn)
        conn.commit()
    finally:
        conn.close()


def cmd_import_csv(args: argparse.Namespace) -> int:
    """Import chunks from CSV (same columns as Review export)."""
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    name = (args.collection or "").strip()
    if not name:
        name = input("Collection name (created if missing): ").strip()
    if not name:
        print("Error: collection name is required.", file=sys.stderr)
        return 1

    group = config._sanitize_group(name)
    if group == MEMORY_GROUP:
        print("Error: do not import into the 'memory' collection; use MCP write_memory.", file=sys.stderr)
        return 1

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print("Error: CSV has no header row.", file=sys.stderr)
                return 1
            fields = {(h or "").strip() for h in reader.fieldnames if h}
            missing = REQUIRED_CSV_FIELDS - fields
            if missing:
                print(f"Error: CSV missing required column(s): {', '.join(sorted(missing))}", file=sys.stderr)
                print(f"Expected header includes: {', '.join(CHUNK_CSV_HEADERS)}", file=sys.stderr)
                return 1
            rows = [_strip_csv_row(dict(r)) for r in reader]
    except OSError as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    by_source: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        sp = (r.get("source_path") or "").strip()
        st = (r.get("source_type") or "").strip() or ".txt"
        if not sp:
            print("Warning: skipping row with empty source_path.", file=sys.stderr)
            continue
        by_source[(sp, st)].append(r)

    if not by_source:
        print("Error: no importable rows (need source_path and chunk rows).", file=sys.stderr)
        return 1

    existed_before = group in _list_sync_groups()
    ensure_collection_db(group)
    gp = config.get_group_paths(group)
    if not existed_before:
        print(f"Created collection '{group}' at {gp.group_dir}")
    else:
        print(f"Using existing collection '{group}'.")

    conn = _connect(group)
    try:
        init_db(conn)
        total_chunks = 0
        for (source_path, source_type), src_rows in sorted(by_source.items(), key=lambda x: x[0][0]):

            def row_chunk_index(row: dict[str, str]) -> int:
                v = _parse_int(row.get("chunk_index", ""), "chunk_index")
                return v if v is not None else 0

            src_rows.sort(key=row_chunk_index)

            doc_summary = ""
            for r in src_rows:
                if (r.get("doc_summary") or "").strip():
                    doc_summary = r["doc_summary"].strip()
                    break
            canonical = ""
            for r in src_rows:
                if (r.get("canonical_url") or "").strip():
                    canonical = r["canonical_url"].strip()
                    break

            existing = conn.execute(
                "SELECT id FROM sources WHERE source_path = ?", (source_path,)
            ).fetchone()
            if existing:
                if args.replace_sources:
                    delete_source_by_id(conn, int(existing["id"]))
                    print(f"  Replaced existing source: {source_path!r}")
                else:
                    print(
                        f"  Skipping existing source {source_path!r} "
                        "(use --replace-sources to delete and re-import).",
                        file=sys.stderr,
                    )
                    continue

            valid_rows = [r for r in src_rows if (r.get("text") or "").strip()]
            if not valid_rows:
                print(f"  Skipping {source_path!r}: no rows with non-empty text.", file=sys.stderr)
                continue

            embed_inputs = [
                build_text_to_embed(
                    doc_summary or None,
                    ((r.get("primary_question_answered") or "").strip() or None),
                    (r.get("text") or "").strip(),
                )
                for r in valid_rows
            ]

            batch_size = 100
            all_embs: list[list[float]] = []
            for i in range(0, len(embed_inputs), batch_size):
                batch = embed_inputs[i : i + batch_size]
                all_embs.extend(embed(batch, group=group))

            bodies: list[dict] = []
            for r, emb in zip(valid_rows, all_embs, strict=True):
                text = (r.get("text") or "").strip()
                pqa = (r.get("primary_question_answered") or "").strip() or None
                ks = _parse_key_signals(r.get("key_signals", ""))
                page_v = _parse_int(r.get("page", ""), "page")
                art = (r.get("artifact_type") or "text").strip() or "text"
                apath = (r.get("artifact_path") or "").strip() or None
                role = (r.get("chunk_role") or "").strip() or None
                concept = (r.get("concept") or "").strip() or None
                dctx = (r.get("decision_context") or "").strip() or None
                chunk: dict = {
                    "text": text,
                    "embedding": emb,
                    "artifact_type": art,
                    "artifact_path": apath,
                    "page": page_v,
                    "concept": concept,
                    "decision_context": dctx,
                    "primary_question_answered": pqa,
                    "chunk_role": role,
                }
                if isinstance(ks, list):
                    chunk["key_signals"] = ks
                elif ks:
                    chunk["key_signals"] = ks
                bodies.append(chunk)

            add_chunks(
                conn,
                source_path,
                source_type,
                bodies,
                doc_summary=doc_summary or None,
            )
            sid_row = conn.execute(
                "SELECT id FROM sources WHERE source_path = ?", (source_path,)
            ).fetchone()
            if sid_row and canonical:
                set_source_external_url(conn, int(sid_row["id"]), canonical)
            conn.commit()
            total_chunks += len(bodies)
            print(f"  Imported {len(bodies)} chunk(s) for {source_path!r}")

        print(f"Done. {total_chunks} chunk(s) written for this import.")
        return 0
    finally:
        conn.close()


def cmd_collections(args: argparse.Namespace) -> int:
    """List all collections."""
    collections = _list_sync_groups()
    if not collections:
        print("No collections found.")
        return 0
    
    print(f"Found {len(collections)} collection(s):")
    for coll in sorted(collections):
        print(f"  - {coll}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all sources in a collection."""
    group = args.collection
    collections = _list_sync_groups()
    
    if group not in collections:
        print(f"Error: Collection '{group}' not found.", file=sys.stderr)
        print(f"Available collections: {', '.join(sorted(collections))}", file=sys.stderr)
        return 1
    
    conn = _connect(group)
    try:
        sources = list_sources(conn)
        if not sources:
            print(f"No sources found in collection '{group}'.")
            return 0
        
        print(f"Found {len(sources)} source(s) in collection '{group}':")
        print(f"{'ID':<6} {'Filename':<60} {'Chunks':<10}")
        print("-" * 80)
        total_chunks = 0
        for source_id, source_path, count, _ in sources:
            # Extract just the filename from the path
            from pathlib import Path
            filename = Path(source_path).name
            # Truncate long filenames for display
            display_name = filename if len(filename) <= 58 else filename[:55] + "..."
            print(f"{source_id:<6} {display_name:<60} {count:<10}")
            total_chunks += count
        print("-" * 80)
        print(f"Total: {total_chunks} chunk{'s' if total_chunks != 1 else ''} across {len(sources)} source(s)")
        return 0
    finally:
        conn.close()


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete all chunks for a source by ID (with confirmation)."""
    group = args.collection
    try:
        source_id = int(args.source_id)
    except ValueError:
        print(f"Error: Source ID must be a number, got '{args.source_id}'.", file=sys.stderr)
        print(f"Use 'ragdoll list {group}' to see source IDs.", file=sys.stderr)
        return 1
    
    collections = _list_sync_groups()
    
    if group not in collections:
        print(f"Error: Collection '{group}' not found.", file=sys.stderr)
        print(f"Available collections: {', '.join(sorted(collections))}", file=sys.stderr)
        return 1
    
    conn = _connect(group)
    try:
        init_db(conn)
        # Check if source exists
        source_info = get_source_by_id(conn, source_id)
        if not source_info:
            print(f"Error: Source ID {source_id} not found in collection '{group}'.", file=sys.stderr)
            print(f"Use 'ragdoll list {group}' to see available source IDs.", file=sys.stderr)
            return 1
        
        source_path, source_type = source_info
        
        # Get chunk count
        count = conn.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)).fetchone()[0]
        
        if count == 0:
            print(f"Source ID {source_id} ({source_path}) has no chunks.", file=sys.stderr)
            return 1
        
        # Confirmation prompt
        if not args.yes:
            print(f"Warning: This will delete {count} chunk{'s' if count != 1 else ''} from source ID {source_id}:")
            print(f"  Path: {source_path}")
            print(f"  Type: {source_type}")
            response = input("Are you sure? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                print("Cancelled.")
                return 0
        
        # Delete chunks from database
        deleted = delete_source_by_id(conn, source_id)
        conn.commit()
        
        if deleted > 0:
            # Unmark from processed list so the file can be re-ingested if put back in ingest
            source_filename = Path(source_path).name
            unmark_processed(source_filename, group)
            # Move source file to deleted folder
            gp = config.get_group_paths(group)
            source_file = Path(source_path)
            deleted_dir = gp.group_dir / "deleted"
            deleted_dir.mkdir(parents=True, exist_ok=True)
            
            if source_file.exists() and source_file.is_file():
                deleted_file = deleted_dir / source_file.name
                # Handle filename conflicts by appending a number
                counter = 1
                while deleted_file.exists():
                    stem = source_file.stem
                    suffix = source_file.suffix
                    deleted_file = deleted_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(source_file), str(deleted_file))
                print(f"Deleted {deleted} chunk{'s' if deleted != 1 else ''} from source ID {source_id} ({source_path}) in collection '{group}'.")
                print(f"Moved source file to: {deleted_file}")
            else:
                print(f"Deleted {deleted} chunk{'s' if deleted != 1 else ''} from source ID {source_id} ({source_path}) in collection '{group}'.")
                print(f"Note: Source file not found at {source_path}, may have been already moved or deleted.")
            print("Removed from processed list; put the file back in the ingest folder to re-ingest (restart ingest service if it was running).")
            return 0
        else:
            print(f"No chunks found for source ID {source_id} in collection '{group}'.", file=sys.stderr)
            return 1
    finally:
        conn.close()


def cmd_reprocess(args: argparse.Namespace) -> int:
    """Unmark a file so it will be re-ingested when put back in the ingest folder."""
    group = args.collection
    match = args.path_or_filename.strip()
    if not match:
        print("Error: path or filename is required.", file=sys.stderr)
        return 1
    collections = _list_sync_groups()
    if group not in collections:
        print(f"Error: Collection '{group}' not found.", file=sys.stderr)
        print(f"Available collections: {', '.join(sorted(collections))}", file=sys.stderr)
        return 1
    removed = unmark_processed(match, group)
    if removed > 0:
        print(f"Unmarked {removed} processed record(s) for '{match}' in collection '{group}'.")
        print("Restart the ingest service so it reloads the processed list, then the file in the ingest folder will be re-ingested:")
        print("  sudo systemctl restart ragdoll-ingest")
        return 0
    print(f"No processed record found for '{match}' in collection '{group}'.", file=sys.stderr)
    print("Use the full ingest path or just the filename (e.g. 'Issue Briefing - Key PLC Protocols.pdf').", file=sys.stderr)
    return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGDoll CLI: Manage collections and sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # collections command
    subparsers.add_parser(
        "collections",
        help="List all collections",
        description="List all available RAG collections (groups)."
    )
    
    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List sources in a collection",
        description="List all unique sources (documents) in a collection with chunk counts."
    )
    list_parser.add_argument(
        "collection",
        help="Collection name to list sources from"
    )
    
    # delete command
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete all chunks for a source by ID",
        description="Delete all chunks associated with a specific source ID in a collection. The source file will be moved to the collection's 'deleted/' folder. Requires confirmation unless --yes is used. Use 'ragdoll list <collection>' to see source IDs."
    )
    delete_parser.add_argument(
        "collection",
        help="Collection name"
    )
    delete_parser.add_argument(
        "source_id",
        help="Source ID to delete chunks for (use 'ragdoll list <collection>' to see IDs)"
    )
    delete_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # reprocess command
    reprocess_parser = subparsers.add_parser(
        "reprocess",
        help="Unmark a file so it will be re-ingested",
        description="Remove a file from the processed list so the watcher will ingest it again when it appears in the ingest folder. Use after deleting chunks and putting the file back. Pass full path or just the filename."
    )
    reprocess_parser.add_argument(
        "collection",
        help="Collection name (e.g. edleadership)"
    )
    reprocess_parser.add_argument(
        "path_or_filename",
        help="Full ingest path or filename (e.g. 'Issue Briefing - Key PLC Protocols.pdf')"
    )

    import_csv_parser = subparsers.add_parser(
        "import-csv",
        help="Import chunks from CSV (Review export / Claude handoff format)",
        description=(
            "Create the collection if it does not exist, then embed and insert chunks. "
            "CSV must include columns: source_path, source_type, chunk_index, text "
            f"(full header: {', '.join(CHUNK_CSV_HEADERS)}). "
            "Rows are grouped by (source_path, source_type). Chunk order follows chunk_index "
            "(re-numbered 0..N-1 in the DB). Skips sources that already exist unless "
            "--replace-sources is set."
        ),
    )
    import_csv_parser.add_argument(
        "csv_path",
        help="Path to the CSV file",
    )
    import_csv_parser.add_argument(
        "-c",
        "--collection",
        metavar="NAME",
        help="Collection name (sanitized like ingest subfolders). If omitted, you are prompted.",
    )
    import_csv_parser.add_argument(
        "--replace-sources",
        action="store_true",
        help="If a source_path already exists, delete its chunks and re-import.",
    )

    args = parser.parse_args()
    
    # Route to command handler
    if args.command == "collections":
        return cmd_collections(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "delete":
        return cmd_delete(args)
    elif args.command == "reprocess":
        return cmd_reprocess(args)
    elif args.command == "import-csv":
        return cmd_import_csv(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
