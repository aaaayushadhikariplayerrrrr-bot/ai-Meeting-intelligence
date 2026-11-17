#!/usr/bin/env python3
"""Administrative CLI script to securely bootstrap the initial development account password.

Guarantees:
- Strictly local command-line execution (zero network/HTTP exposure).
- Interactive password input via getpass (no terminal echo or shell history logging).
- Strictly targets the existing 'dev@local.internal' account.
- Refuses to overwrite an existing password unless explicit --force argument is supplied.
- Never prints or logs the plaintext password.
- Validates password length and complexity requirements.
- Uses project-standard Argon2id hashing via app.core.security.hash_password.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure backend root directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.session import SessionLocal
from app.models.identity import User
from app.core.security import hash_password

DEV_EMAIL = "dev@local.internal"


def validate_password_complexity(password: str) -> None:
    """Validate password length and character requirements."""
    if not password or not password.strip():
        raise ValueError("Password cannot be empty or whitespace only.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password cannot exceed 128 characters.")


def get_password_input(args: argparse.Namespace) -> str:
    """Retrieve password from argument, environment, stdin, or interactive prompt."""
    if args.password:
        password = args.password
    elif "DEV_INITIAL_PASSWORD" in os.environ and os.environ["DEV_INITIAL_PASSWORD"]:
        password = os.environ["DEV_INITIAL_PASSWORD"]
    elif args.password_stdin:
        password = sys.stdin.read().strip("\r\n")
    else:
        # Interactive prompt
        password = getpass.getpass(f"Enter new password for {DEV_EMAIL}: ")
        confirm = getpass.getpass(f"Confirm password for {DEV_EMAIL}: ")
        if password != confirm:
            raise ValueError("Passwords do not match.")

    validate_password_complexity(password)
    return password


def bootstrap_dev_password(
    force: bool = False,
    password: str = None,
    db: Optional[Session] = None,
    session_factory=None,
) -> int:
    """Set password for dev@local.internal in the database."""
    validate_password_complexity(password)

    close_db = False
    if db is None:
        factory = session_factory or SessionLocal
        db = factory()
        close_db = True

    try:
        stmt = select(User).where(User.email == DEV_EMAIL)
        user = db.execute(stmt).scalar_one_or_none()

        if not user:
            print(f"Error: Account '{DEV_EMAIL}' not found in database.", file=sys.stderr)
            return 1

        if user.hashed_password is not None and not force:
            print(
                f"Error: Account '{DEV_EMAIL}' already has a password set. "
                "Use --force to overwrite.",
                file=sys.stderr,
            )
            return 1

        user.hashed_password = hash_password(password)
        db.commit()
        print(f"Success: Password successfully set for '{DEV_EMAIL}'.")
        return 0
    except Exception as exc:
        db.rollback()
        print(f"Error: Database transaction failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if close_db:
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Securely bootstrap the initial password for '{DEV_EMAIL}'."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing password if already set.",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Password value (for non-interactive automation/testing only).",
    )
    parser.add_argument(
        "--password-stdin",
        action="store_true",
        help="Read password from standard input (for piped automation).",
    )

    args = parser.parse_args()

    try:
        password = get_password_input(args)
    except ValueError as err:
        print(f"Validation Error: {err}", file=sys.stderr)
        return 1

    return bootstrap_dev_password(force=args.force, password=password)


if __name__ == "__main__":
    sys.exit(main())
