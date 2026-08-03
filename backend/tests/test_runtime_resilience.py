import asyncio
from unittest.mock import MagicMock, patch

from app import database
from app import main


def test_sqlite_connections_wait_for_writer_lock():
    with database.engine.connect() as connection:
        timeout_ms = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()

    assert timeout_ms == 5000


def test_background_mediamtx_sync_always_closes_session():
    session = MagicMock()
    with (
        patch.object(main, "SessionLocal", return_value=session),
        patch.object(main, "sync_streams") as sync_streams,
    ):
        main._bg_sync()

    sync_streams.assert_called_once_with(session)
    session.close.assert_called_once_with()


def test_lifespan_starts_mediamtx_sync_without_blocking_startup():
    thread = MagicMock()

    async def exercise_lifespan():
        async with main.lifespan(main.app):
            pass

    with (
        patch.object(main, "init_db"),
        patch.object(main.stream_manager, "startup"),
        patch.object(main.stream_manager, "shutdown"),
        patch.object(main.threading, "Thread", return_value=thread) as thread_factory,
    ):
        asyncio.run(exercise_lifespan())

    thread_factory.assert_called_once_with(target=main._bg_sync, daemon=True)
    thread.start.assert_called_once_with()
