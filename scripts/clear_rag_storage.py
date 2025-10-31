#!/usr/bin/env python3
"""
Helper script to clear Qdrant RAG storage.

Use this if you encounter version compatibility issues or want to start fresh.

Run with: python scripts/clear_rag_storage.py
"""

import shutil
import sys
from pathlib import Path


def main():
    """Clear the Qdrant storage directory."""
    project_root = Path(__file__).parent.parent
    storage_path = project_root / "data" / "qdrant_storage"

    print("\n🗑️  RAG Storage Cleanup Utility\n")
    print(f"Storage path: {storage_path}\n")

    if not storage_path.exists():
        print("✓ Storage directory does not exist. Nothing to clear.")
        return

    # Count files
    file_count = sum(1 for _ in storage_path.rglob("*") if _.is_file())

    print(f"Found {file_count} files in storage directory.")

    # Confirm deletion
    response = input("\nDo you want to delete the RAG storage? [y/N]: ").strip().lower()

    if response in ['y', 'yes']:
        try:
            shutil.rmtree(storage_path)
            print("\n✓ RAG storage cleared successfully!")
            print("\nThe storage will be recreated automatically when you run the agent.")
        except Exception as e:
            print(f"\n✗ Error clearing storage: {e}")
            sys.exit(1)
    else:
        print("\n✓ Operation cancelled. No changes made.")


if __name__ == "__main__":
    main()
