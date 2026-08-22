from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from platform_api.config import PlatformSettings, get_platform_settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=4)
def build_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///"):
        relative_path = database_url.removeprefix("sqlite:///")
        if relative_path != ":memory:":
            Path(relative_path).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False, "timeout": 10} if database_url.startswith("sqlite") else {}
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    if database_url.startswith("sqlite"):
        event.listen(engine, "connect", _configure_sqlite)
    return engine


def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.close()


def build_session_factory(settings: PlatformSettings | None = None) -> sessionmaker[Session]:
    selected = settings or get_platform_settings()
    return sessionmaker(build_engine(selected.database_url), expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    factory = build_session_factory()
    with factory() as session:
        yield session


def initialize_development_schema(settings: PlatformSettings | None = None) -> None:
    selected = settings or get_platform_settings()
    if selected.auto_create_schema:
        # Production schema changes are owned by Alembic. This convenience path is
        # intentionally limited to local development and isolated automated tests.
        from platform_api import models  # noqa: F401

        Base.metadata.create_all(build_engine(selected.database_url))
