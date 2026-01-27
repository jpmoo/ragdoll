"""CLI tool for managing RAGDoll collections and sources."""

import argparse
import sys
from pathlib import Path

from . import config
from .storage import _connect, _list_sync_groups, delete_source, init_db, list_sources


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
        total_chunks = 0
        for source_path, count in sources:
            print(f"  - {source_path} ({count} chunk{'s' if count != 1 else ''})")
            total_chunks += count
        print(f"\nTotal: {total_chunks} chunk{'s' if total_chunks != 1 else ''} across {len(sources)} source(s)")
        return 0
    finally:
        conn.close()


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete all chunks for a source (with confirmation)."""
    group = args.collection
    source_path = args.source
    collections = _list_sync_groups()
    
    if group not in collections:
        print(f"Error: Collection '{group}' not found.", file=sys.stderr)
        print(f"Available collections: {', '.join(sorted(collections))}", file=sys.stderr)
        return 1
    
    conn = _connect(group)
    try:
        init_db(conn)
        # Check if source exists
        sources = list_sources(conn)
        source_dict = {path: count for path, count in sources}
        
        if source_path not in source_dict:
            print(f"Error: Source '{source_path}' not found in collection '{group}'.", file=sys.stderr)
            print(f"Use 'ragdoll list {group}' to see available sources.", file=sys.stderr)
            return 1
        
        count = source_dict[source_path]
        
        # Confirmation prompt
        if not args.yes:
            print(f"Warning: This will delete {count} chunk{'s' if count != 1 else ''} from source '{source_path}' in collection '{group}'.")
            response = input("Are you sure? (yes/no): ").strip().lower()
            if response not in ("yes", "y"):
                print("Cancelled.")
                return 0
        
        # Delete
        deleted = delete_source(conn, source_path)
        conn.commit()
        
        if deleted > 0:
            print(f"Deleted {deleted} chunk{'s' if deleted != 1 else ''} from source '{source_path}' in collection '{group}'.")
            return 0
        else:
            print(f"No chunks found for source '{source_path}' in collection '{group}'.", file=sys.stderr)
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
        help="Delete all chunks for a source",
        description="Delete all chunks associated with a specific source in a collection. Requires confirmation unless --yes is used."
    )
    delete_parser.add_argument(
        "collection",
        help="Collection name"
    )
    delete_parser.add_argument(
        "source",
        help="Source path to delete chunks for"
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
