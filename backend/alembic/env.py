import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Ensure backend directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.core.config import settings
from app.database.base import Base
# Import all models to register them on Base.metadata
import app.models  # noqa: F401

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Overwrite sqlalchemy.url with application settings if default placeholder is present
current_url = config.get_main_option("sqlalchemy.url")
if not current_url or "driver://user:pass" in current_url:
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

target_metadata = Base.metadata


def include_object(object, name, type_, reflected, compare_to):
    """Exclude internal migration tracking tables and SQLite FTS virtual/shadow tables from autogenerate diffs."""
    if type_ == "table" and name:
        if name.startswith("_") or name == "transcript_chunks_fts" or name.startswith("transcript_chunks_fts_"):
            return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Enables batch mode for SQLite ALTER TABLE support
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Build configuration dictionary with application database URL
    configuration = config.get_section(config.config_ini_section, {})
    url = config.get_main_option("sqlalchemy.url")
    if not url or "driver://user:pass" in url:
        url = settings.DATABASE_URL
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        if is_sqlite:
            # Disable SQLite foreign keys directly on DBAPI connection outside of any transaction
            raw_conn = connection.connection.dbapi_connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.execute("PRAGMA foreign_keys = OFF")
            raw_cursor.close()

        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,  # Enables batch mode for SQLite ALTER TABLE support
                include_object=include_object,
            )

            with context.begin_transaction():
                context.run_migrations()
        finally:
            if is_sqlite:
                # Always restore SQLite foreign keys on DBAPI connection after transaction completes or fails
                raw_conn = connection.connection.dbapi_connection
                raw_cursor = raw_conn.cursor()
                raw_cursor.execute("PRAGMA foreign_keys = ON")
                raw_cursor.close()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
