"""
Account and data management from the command line (no web server needed).

    python manage.py create-user NAME            # asks for the password
    python manage.py set-password NAME           # reset a forgotten password
    python manage.py import-md NAME FILE.md [--title "..."]
    python manage.py list-users

Uses the same database as the server (NOTES_DB, default notes.db).
"""
from __future__ import annotations

import argparse
import getpass
import sqlite3
import sys
from pathlib import Path

import accounts

def ask_password(confirm: bool = True) -> str:
    password = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Repeat password: ") != password:
        sys.exit("Passwords do not match.")
    return password


def require_user(username: str) -> dict:
    user = accounts.find_user(username)
    if user is None:
        sys.exit(f"No user named {username!r}.")
    return user


def cmd_create_user(args):
    try:
        user_id = accounts.create_user(args.username, args.password or ask_password())
    except ValueError as exc:
        sys.exit(str(exc))
    except sqlite3.IntegrityError:
        sys.exit(f"User {args.username!r} already exists.")
    print(f"Created user {args.username!r} (id {user_id}).")


def cmd_set_password(args):
    user = require_user(args.username)
    try:
        accounts.set_password(user["id"], args.password or ask_password())
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Password changed for {user['username']!r}; existing sign-ins were ended.")


def cmd_import_md(args):
    user = require_user(args.username)
    path = Path(args.file)
    note_id = accounts.import_markdown(
        user["id"], path.read_text(encoding="utf-8-sig"), args.title or path.stem
    )
    print(f"Imported {path.name} into {user['username']!r} as note {note_id}.")


def cmd_list_users(args):
    with accounts.db() as conn:
        rows = conn.execute(
            """SELECT users.username, COUNT(notes.id) AS notes FROM users
               LEFT JOIN notes ON notes.user_id = users.id
               GROUP BY users.id ORDER BY users.username"""
        ).fetchall()
    for row in rows:
        print(f"{row['username']:<32} {row['notes']} notes")
    if not rows:
        print("No users yet.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create-user", help="create an account")
    p.add_argument("username")
    p.add_argument("--password", help="omit to be asked (keeps it out of shell history)")
    p.set_defaults(func=cmd_create_user)

    p = sub.add_parser("set-password", help="reset a password")
    p.add_argument("username")
    p.add_argument("--password", help="omit to be asked")
    p.set_defaults(func=cmd_set_password)

    p = sub.add_parser("import-md", help="import a .md export into an account")
    p.add_argument("username")
    p.add_argument("file")
    p.add_argument("--title", help="used when the file has no '# title' line")
    p.set_defaults(func=cmd_import_md)

    sub.add_parser("list-users", help="show accounts and note counts").set_defaults(func=cmd_list_users)

    args = parser.parse_args()
    accounts.init_db()
    args.func(args)


if __name__ == "__main__":
    main()
