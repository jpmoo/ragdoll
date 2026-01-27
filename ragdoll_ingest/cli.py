"""CLI tool for managing RAGDoll collections and sources."""

import argparse
import sys
from pathlib import Path

from . import config
from .storage import _connect, _list_sync_groups, delete_source_by_id, get_source_by_id, init_db, list_sources


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
        for source_id, source_path, count in sources:
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
        
        # Delete
        deleted = delete_source_by_id(conn, source_id)
        conn.commit()
        
        if deleted > 0:
            print(f"Deleted {deleted} chunk{'s' if deleted != 1 else ''} from source ID {source_id} ({source_path}) in collection '{group}'.")
            return 0
        else:
            print(f"No chunks found for source ID {source_id} in collection '{group}'.", file=sys.stderr)
            return 1
    finally:
        conn.close()


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
        description="Delete all chunks associated with a specific source ID in a collection. Requires confirmation unless --yes is used. Use 'ragdoll list <collection>' to see source IDs."
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
    
    args = parser.parse_args()
    
    # Route to command handler
    if args.command == "collections":
        return cmd_collections(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "delete":
        return cmd_delete(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
