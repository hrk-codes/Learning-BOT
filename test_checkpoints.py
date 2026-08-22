from graph.checkpoints import build_sqlite_checkpointer


def test_checkpointer_is_reused_for_a_local_database_path(tmp_path) -> None:
    path = tmp_path / "checkpoints.db"

    first = build_sqlite_checkpointer(path)
    second = build_sqlite_checkpointer(path)

    assert first is second
