"""
CLI utility for creating and managing developer and household accounts without hardcoded production credentials.

Usage:
  python -m backend.app.cli.manage_users create --username <name> --password <pwd> --role [household|developer]
  python -m backend.app.cli.manage_users list
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.database import db_manager
from backend.app.services.user_service import user_service


def main():
    parser = argparse.ArgumentParser(description="Reliable Energy Forecasting - User & Developer CLI Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: create
    create_parser = subparsers.add_parser("create", help="Create a new user account")
    create_parser.add_argument("--username", required=True, help="Unique username (alphanumeric, 3-32 chars)")
    create_parser.add_argument("--password", required=True, help="Plaintext password (min 6 chars)")
    create_parser.add_argument("--role", choices=["household", "developer"], default="household", help="Account role")

    # Command: list
    subparsers.add_parser("list", help="List all registered users (excluding password hashes)")

    args = parser.parse_args()

    # Ensure tables are initialized
    db_manager.init_db()

    if args.command == "create":
        try:
            created = user_service.create_user(
                username=args.username,
                password=args.password,
                role=args.role,
            )
            print(f"[SUCCESS] Created account for user '{created['username']}' with role '{created['role']}' (ID: {created['id']})")
        except Exception as exc:
            print(f"[ERROR] Failed to create user: {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        conn = db_manager.get_connection()
        try:
            cursor = conn.execute("SELECT id, username, role, created_at, is_active FROM users ORDER BY created_at ASC;")
            rows = cursor.fetchall()
            if not rows:
                print("No registered users found in database.")
                return
            print(f"{'ID':<18} | {'USERNAME':<20} | {'ROLE':<12} | {'CREATED AT':<25} | {'ACTIVE'}")
            print("-" * 85)
            for row in rows:
                print(f"{row['id']:<18} | {row['username']:<20} | {row['role']:<12} | {row['created_at']:<25} | {bool(row['is_active'])}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
