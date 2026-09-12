"""CLI tool for managing RAGDoll collections and sources."""

import argparse
import shutil
import sys
from pathlib import Path

from . import config
from .chunk_csv import CHUNK_CSV_HEADERS
from .csv_import import parse_csv_bytes, run_csv_import
from .insights import (
    INSIGHTS_GROUP,
    ORIGINS,
    InsightError,
    get_insight,
    list_insights,
    migrate_memory_collection,
    restore_insight,
    retire_insight,
)
from .query_log import get_query, recent_queries
from .stew import get_run, list_runs, read_reflection, run_stew
from .storage import (
    _connect,
    _list_sync_groups,
    delete_source_by_id,
    get_source_by_id,
    init_db,
    list_sources,
    unmark_processed,
)


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

    try:
        data = csv_path.read_bytes()
        rows = parse_csv_bytes(data)
    except OSError as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        summary = run_csv_import(name, rows, replace_sources=args.replace_sources)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for line in summary.messages:
        print(line)
    return 0


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
        print(f"{'ID':<6} {'Title / filename':<60} {'Chunks':<10}")
        print("-" * 80)
        total_chunks = 0
        for source_id, source_path, count, _summary, _eu, display_title in sources:
            basename = Path(source_path).name
            dt = (display_title or "").strip()
            filename = dt or basename
            display_cell = filename if len(filename) <= 58 else filename[:55] + "..."
            print(f"{source_id:<6} {display_cell:<60} {count:<10}")
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


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _print_insight(i: dict) -> None:
    flags = f"{i['status']}, origin {i['origin']}, version {i['version']}" + (", pinned" if i["pinned"] else "")
    print(f"Insight {i['id']} ({flags})")
    if i["status"] != "active":
        superseded = f" (superseded by {i['superseded_by']})" if i["superseded_by"] else ""
        print(f"  Reason: {i['status_reason']}{superseded}")
    for label, key in (
        ("Topic", "topic"), ("Question", "question"), ("Statement", "statement"),
        ("Rationale", "rationale"), ("Open questions", "open_questions"),
    ):
        if i.get(key):
            print(f"\n{label}:\n  {i[key]}")
    print()
    if i["tags"]:
        print(f"Tags: {', '.join(i['tags'])}")
    if i["confidence"] is not None:
        print(f"Confidence: {i['confidence']}")
    run = f"  Run: {i['run_id']}" if i["run_id"] else ""
    print(f"Created: {i['created_at']}  Updated: {i['updated_at']}{run}")
    if i["lineage"]:
        print(f"\nLineage ({len(i['lineage'])}):")
        for entry in i["lineage"]:
            if entry["ref_source_path"]:
                ref = f"{entry['ref_group']}: {Path(entry['ref_source_path']).name}#{entry['ref_chunk_index']}"
            elif entry["ref_query_id"]:
                ref = f"query {entry['ref_query_id']}"
            else:
                ref = f"insight {entry['ref_insight_id']}"
            print(f"  - {entry['kind']} ({entry['relation'] or 'related'}): {ref}")
    print(f"\nRevisions ({len(i['revisions'])}):")
    for r in i["revisions"]:
        print(f"  {r['ts']}  {r['actor']:<9} {r['action']:<9} {r['reason'] or ''}")


def cmd_insights(args: argparse.Namespace) -> int:
    """List, show, retire, or restore insights; migrate the legacy memory collection."""
    try:
        if args.insights_command == "list":
            rows = list_insights(
                status=None if args.status == "all" else args.status, origin=args.origin, limit=args.limit
            )
            if not rows:
                print("No insights found.")
                return 0
            print(f"{'ID':<6} {'Status':<11} {'Origin':<9} {'Updated (UTC)':<20} Statement")
            print("-" * 100)
            for i in rows:
                print(f"{i['id']:<6} {i['status']:<11} {i['origin']:<9} {i['updated_at']:<20} {_truncate(i['statement'], 52)}")
            return 0
        if args.insights_command == "show":
            _print_insight(get_insight(args.insight_id))
            return 0
        if args.insights_command == "retire":
            retire_insight(args.insight_id, actor="user", reason=args.reason)
            print(f"Retired insight {args.insight_id}. It is no longer searchable; 'ragdoll insights restore {args.insight_id}' undoes this.")
            return 0
        if args.insights_command == "restore":
            restore_insight(args.insight_id, actor="user", reason=args.reason)
            print(f"Restored insight {args.insight_id}.")
            return 0
        if args.insights_command == "stew":
            summary = run_stew(
                write=args.write, model=args.model, since=args.since,
                max_clusters=args.max_clusters, progress=lambda m: print(f"  {m}"),
            )
            mode = "created" if args.write else "would create"
            print(f"\nRun {summary['run_id']} ({'writing' if args.write else 'dry run'}, model {summary['model']})")
            print(f"  {summary['n_queries']} queries since {summary['since']}; {len(summary['clusters'])} cluster(s) stewed")
            print(f"  {mode}: {summary['n_created']}   reinforced: {summary['n_reinforced']}   rejected: {summary['n_rejected']}")
            if summary["gaps"]:
                print(f"  {len(summary['gaps'])} question(s) found nothing (listed in the reflection)")
            print(f"  Reflection: {summary['reflection_path']}")
            if not args.write:
                print("  Nothing was written. Re-run with --write to create these insights.")
            return 0
        if args.insights_command == "runs":
            runs = list_runs(args.limit)
            if not runs:
                print("No stew runs yet.")
                return 0
            print(f"{'Run':<24} {'Mode':<8} {'Queries':<8} {'New':<5} {'Reinf':<6} {'Rej':<5} Model")
            print("-" * 100)
            for r in runs:
                mode = "dry run" if r["dry_run"] else "write"
                print(f"{r['run_id']:<24} {mode:<8} {r['n_queries']:<8} {r['n_created']:<5} {r['n_reinforced']:<6} {r['n_rejected']:<5} {r['model']}")
            return 0
        if args.insights_command == "run":
            run = get_run(args.run_id)
            if not run:
                print(f"Error: Run {args.run_id} not found.", file=sys.stderr)
                return 1
            print(f"Run {run['run_id']} ({'dry run' if run['dry_run'] else 'write'}, model {run['model']}, {run['status']})")
            print(f"  {run['started_at']} to {run['finished_at'] or '(unfinished)'}; {run['n_queries']} queries since {run['since']}")
            for c in run["clusters"]:
                print(f"\n[{c['label']}]  {c['n_queries']} questions, {c['n_chunks']} passages, {c['outcome']}")
                if c["notes"]:
                    print(f"  Notes: {c['notes']}")
                for cand in c["candidates"]:
                    target = f" -> insight {cand['insight_id']}" if cand["insight_id"] else ""
                    print(f"  - [{cand['decision']}{target}] {cand['statement']}")
                    if cand["decision_reason"]:
                        print(f"      {cand['decision_reason']}")
                    sources = cand["trace"].get("supporting_sources") or []
                    if sources:
                        print(f"      sources: {', '.join(sources)}")
                    if cand["trace"].get("weak_spots"):
                        print(f"      weak spots: {cand['trace']['weak_spots']}")
            print(f"\nReflection: {run['reflection_path']}")
            return 0
        if args.insights_command == "reflection":
            text = read_reflection(args.run_id)
            if text is None:
                print(f"Error: No reflection for run {args.run_id}.", file=sys.stderr)
                return 1
            print(text)
            return 0
        if args.insights_command == "migrate-memory":
            r = migrate_memory_collection(archive=not args.no_archive, dry_run=args.dry_run)
            if r["found"] == 0:
                print("No memory collection found; nothing to migrate.")
                return 0
            verb = "Would migrate" if args.dry_run else "Migrated"
            print(f"{verb} {r['migrated']} of {r['found']} memories into insights.")
            if r["skipped_existing"]:
                print(f"Skipped {r['skipped_existing']} already migrated.")
            if r["skipped_empty"]:
                print(f"Skipped {r['skipped_empty']} with no content.")
            if r["archived_to"]:
                print(f"Moved the memory collection to {r['archived_to']}")
            if r["archive_error"]:
                print(f"Error: the memory collection was not archived. {r['archive_error']}", file=sys.stderr)
                print(
                    "Re-run as the user that owns the data directory (e.g. sudo -u <service user>); "
                    "memories already migrated are skipped.",
                    file=sys.stderr,
                )
                return 1
            return 0
    except InsightError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 1


def cmd_queries(args: argparse.Namespace) -> int:
    """List recently logged queries, or show one with its hits."""
    if args.query_id is not None:
        q = get_query(args.query_id)
        if not q:
            print(f"Error: Query {args.query_id} not found.", file=sys.stderr)
            return 1
        print(f"Query {q['id']}  {q['ts']} UTC  via {q['transport']}")
        print(f"  Prompt:      {q['prompt']}")
        print(f"  Expanded:    {q['expanded_query']}")
        print(f"  Collections: {', '.join(q['collections'])}  (threshold {q['threshold']}, {q['result_count']} results)")
        for h in q["hits"]:
            # Insight paths ("insight/12") read better whole; document paths by filename
            name = h["source_path"] if h["group_name"] == INSIGHTS_GROUP else Path(h["source_path"]).name
            print(f"  {h['rank']:>3}. {h['similarity']:.3f}  {h['group_name']}  {name}#{h['chunk_index']}")
        return 0
    rows = recent_queries(args.limit)
    if not rows:
        print("No queries logged.")
        return 0
    print(f"{'ID':<6} {'When (UTC)':<20} {'Via':<5} {'Hits':<6} {'Top':<6} Prompt")
    print("-" * 100)
    for q in rows:
        top = f"{q['top_similarity']:.3f}" if q["top_similarity"] is not None else "-"
        print(f"{q['id']:<6} {q['ts']:<20} {q['transport']:<5} {q['result_count']:<6} {top:<6} {_truncate(q['prompt'], 55)}")
    return 0


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
            "CSV must include columns: chunk_index, text (full header may match export: "
            f"{', '.join(CHUNK_CSV_HEADERS)}). "
            "Rows are grouped per document using source_path if set, else canonical_url, "
            "else import:source_key:{id}. source_type defaults to .txt if blank. "
            "Chunk order follows chunk_index (stored as 0..N-1). Skips sources that already "
            "exist unless --replace-sources is set."
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

    # insights command
    insights_parser = subparsers.add_parser(
        "insights",
        help="List, show, retire, or restore insights",
        description="Manage the insights collection. Retired insights are kept (with the reason) but no longer searched.",
    )
    insights_sub = insights_parser.add_subparsers(dest="insights_command", required=True)
    insights_list = insights_sub.add_parser("list", help="List insights, newest-updated first")
    insights_list.add_argument("--status", choices=["active", "retired", "superseded", "all"], default="active")
    insights_list.add_argument("--origin", choices=list(ORIGINS))
    insights_list.add_argument("-n", "--limit", type=int, default=50)
    insights_show = insights_sub.add_parser("show", help="Show an insight with its lineage and revisions")
    insights_show.add_argument("insight_id", type=int)
    insights_retire = insights_sub.add_parser("retire", help="Stop an insight from being searched (kept, not deleted)")
    insights_retire.add_argument("insight_id", type=int)
    insights_retire.add_argument("--reason", required=True, help="Why it's being retired; kept so it isn't regenerated")
    insights_restore = insights_sub.add_parser("restore", help="Make a retired or superseded insight searchable again")
    insights_restore.add_argument("insight_id", type=int)
    insights_restore.add_argument("--reason")
    insights_stew = insights_sub.add_parser(
        "stew",
        help="Synthesize insights from logged queries (dry run unless --write)",
        description=(
            "Cluster recently logged queries, send each cluster's questions and the passages they returned to the "
            "insight model, and record what it proposes. Writes a reflection either way."
        ),
    )
    insights_stew.add_argument("--write", action="store_true", help="Create the insights instead of only recording what would be created")
    insights_stew.add_argument("--model", help=f"Override RAGDOLL_INSIGHT_MODEL (currently {config.INSIGHT_MODEL})")
    insights_stew.add_argument("--since", metavar="TS", help="Read queries from this UTC timestamp (default: the last run)")
    insights_stew.add_argument("--max-clusters", type=int, dest="max_clusters", help="Stew at most this many clusters")
    insights_runs = insights_sub.add_parser("runs", help="List stew runs")
    insights_runs.add_argument("-n", "--limit", type=int, default=20)
    insights_run = insights_sub.add_parser("run", help="Show one stew run: clusters, candidates, and decisions")
    insights_run.add_argument("run_id")
    insights_reflection = insights_sub.add_parser("reflection", help="Print a run's reflection")
    insights_reflection.add_argument("run_id")

    insights_migrate = insights_sub.add_parser(
        "migrate-memory",
        help="Copy the legacy memory collection into insights, then archive it",
        description="Each memory becomes an 'agent' insight. Safe to re-run; already-migrated memories are skipped.",
    )
    insights_migrate.add_argument("--dry-run", action="store_true", help="Report what would be migrated without writing")
    insights_migrate.add_argument("--no-archive", action="store_true", help="Leave the memory collection in place")

    # queries command
    queries_parser = subparsers.add_parser(
        "queries",
        help="Show logged queries",
        description="List recent API/MCP queries from the query log, or show one query with the chunks it returned.",
    )
    queries_parser.add_argument("query_id", nargs="?", type=int, help="Query ID to show in detail")
    queries_parser.add_argument("-n", "--limit", type=int, default=20)

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
    elif args.command == "insights":
        return cmd_insights(args)
    elif args.command == "queries":
        return cmd_queries(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
