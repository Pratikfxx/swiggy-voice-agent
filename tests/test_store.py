from concurrent.futures import ThreadPoolExecutor
import importlib
import os
import sys
from threading import Event, Thread
import types
from unittest.mock import MagicMock, Mock
import warnings


def _fresh_store(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "swiggy.db"))
    sys.modules.pop("store", None)
    return importlib.import_module("store")


def _fresh_agent(monkeypatch, tmp_path, extra_env=None):
    env = {"ANTHROPIC_API_KEY": "test-key", "DEMO_MODE": "true"}
    env.update(extra_env or {})
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ORDER_HISTORY_DB", str(tmp_path / "swiggy.db"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        sys.modules.pop("store", None)
        sys.modules.pop("order_history", None)
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


def _fresh_postgres(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    connection.execute.return_value.fetchone.return_value = None
    connection.execute.return_value.fetchall.return_value = []

    psycopg = types.ModuleType("psycopg")
    psycopg.connect = Mock(return_value=connection)
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/swiggy")
    sys.modules.pop("store", None)
    return importlib.import_module("store"), connection, psycopg, rows


def test_sqlite_backend_round_trips_a_session(monkeypatch, tmp_path):
    store = _fresh_store(monkeypatch, tmp_path)
    history = [{"role": "user", "content": "milk"}]

    store.update_session("call-1", history)

    assert store.get_session("call-1") == history
    store.clear_session("call-1")
    assert store.get_session("call-1") == []


def test_agent_session_history_is_truncated_and_sanitized(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    history = []
    for i in range(12):
        history.extend(
            [
                {"role": "user", "content": [{"type": "text", "text": f"user {i}"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"assistant {i}"},
                        {"type": "tool_use", "id": f"tool-{i}"},
                    ],
                },
            ]
        )

    agent.update_session("call-2", history)
    saved = agent.get_session("call-2")

    assert len(saved) == 20
    assert saved[0] == {"role": "user", "content": "user 2"}
    assert saved[-1] == {"role": "assistant", "content": "assistant 11"}
    assert all(isinstance(message["content"], str) for message in saved)
    assert "tool_use" not in str(saved)


def test_orders_round_trip_with_recent_limit_and_desc_order(monkeypatch, tmp_path):
    store = _fresh_store(monkeypatch, tmp_path)

    store.save_order("wa-1", "grocery", "first", [{"name": "milk"}], total_amount=50)
    store.save_order("wa-1", "grocery", "second", [{"name": "eggs"}], total_amount=80)
    store.save_order("wa-1", "grocery", "third", [{"name": "bread"}], total_amount=40)
    store.save_order("other", "grocery", "other", [{"name": "rice"}], total_amount=100)

    recent = store.get_recent_orders("wa-1", limit=2)

    assert [order["summary"] for order in recent] == ["third", "second"]
    assert recent[0]["items"] == [{"name": "bread"}]
    assert "items_json" not in recent[0]
    assert store.get_last_order("wa-1")["summary"] == "third"


def test_expired_sessions_are_pruned_on_write(monkeypatch, tmp_path):
    store = _fresh_store(monkeypatch, tmp_path)
    monkeypatch.setenv("SESSION_TTL_SECS", "1")

    store.update_session("old", [{"role": "user", "content": "old"}])
    with store._connect() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            ("2000-01-01T00:00:00", "old"),
        )

    store.update_session("new", [{"role": "user", "content": "new"}])

    assert store.get_session("old") == []
    assert store.get_session("new") == [{"role": "user", "content": "new"}]


def test_store_failure_degrades_gracefully(monkeypatch, tmp_path):
    store = _fresh_store(monkeypatch, tmp_path)

    def fail_connect():
        raise OSError("database unavailable")

    monkeypatch.setattr(store, "_connect", fail_connect)

    assert store.get_session("missing") == []
    store.update_session("call-3", [{"role": "user", "content": "milk"}])


def test_slow_storage_miss_does_not_block_cached_session(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    agent._sessions.clear()
    with agent._sessions_lock:
        agent._cache_session("cached", [{"role": "user", "content": "cached"}])

    storage_started = Event()
    release_storage = Event()

    def slow_get_session(session_id):
        if session_id == "slow":
            storage_started.set()
            assert release_storage.wait(2)
        return []

    monkeypatch.setattr(agent.store, "get_session", slow_get_session)
    slow_thread = Thread(target=agent.get_session, args=("slow",))
    slow_thread.start()
    assert storage_started.wait(1)

    cached_result = []
    cached_done = Event()

    def read_cached_session():
        cached_result.append(agent.get_session("cached"))
        cached_done.set()

    cached_thread = Thread(target=read_cached_session)
    cached_thread.start()
    try:
        assert cached_done.wait(0.2), "a slow storage call blocked the session cache"
    finally:
        release_storage.set()
        slow_thread.join(timeout=2)
        cached_thread.join(timeout=2)
    assert cached_result == [[{"role": "user", "content": "cached"}]]


def test_postgres_writes_bypass_sqlite_write_lock(monkeypatch):
    store, connection, _, _ = _fresh_postgres(monkeypatch)
    connect_started = Event()

    def connect():
        connect_started.set()
        return connection

    monkeypatch.setattr(store, "_connect", connect)
    store._WRITE_LOCK.acquire()
    try:
        write_thread = Thread(
            target=store.update_session,
            args=("call-1", [{"role": "user", "content": "milk"}]),
        )
        write_thread.start()
        assert connect_started.wait(0.2), "Postgres write waited on the SQLite lock"
    finally:
        store._WRITE_LOCK.release()
        write_thread.join(timeout=2)


def test_postgres_dialect_is_generated_by_real_store_paths(monkeypatch):
    store, connection, psycopg, rows = _fresh_postgres(monkeypatch)

    store.save_order("call-1", "grocery", "milk", [{"name": "milk"}])
    store.get_last_order("call-1")
    store.get_recent_orders("call-1", limit=2)
    store.get_session("call-1")
    store.update_session("call-1", [{"role": "user", "content": "milk"}])
    store.clear_session("call-1")

    statements = [" ".join(call.args[0].split()) for call in connection.execute.call_args_list]
    orders_ddl = next(sql for sql in statements if sql.startswith("CREATE TABLE IF NOT EXISTS orders"))
    sessions_ddl = next(sql for sql in statements if sql.startswith("CREATE TABLE IF NOT EXISTS sessions"))
    save_sql = next(sql for sql in statements if sql.startswith("INSERT INTO orders"))
    recent_sql = next(sql for sql in statements if sql.startswith("SELECT * FROM orders") and "LIMIT %s" in sql)
    upsert_sql = next(sql for sql in statements if sql.startswith("INSERT INTO sessions"))

    assert psycopg.connect.call_args.args[0] == "postgresql://test.invalid/swiggy"
    assert psycopg.connect.call_args.kwargs["row_factory"] is rows.dict_row
    assert store._placeholder() == "%s"
    assert "id BIGSERIAL PRIMARY KEY" in orders_ddl
    assert "INTEGER PRIMARY KEY AUTOINCREMENT" not in orders_ddl
    assert "session_id TEXT PRIMARY KEY" in sessions_ddl
    assert "VALUES (%s, %s, %s, %s, %s, %s, %s)" in save_sql
    assert "LIMIT %s" in recent_sql
    assert "ON CONFLICT(session_id) DO UPDATE SET" in upsert_sql
    assert "history_json = excluded.history_json," in upsert_sql
    assert "updated_at = excluded.updated_at" in upsert_sql
    assert all("?" not in sql for sql in statements)


def test_sqlite_concurrent_writes_are_persisted(monkeypatch, tmp_path):
    store = _fresh_store(monkeypatch, tmp_path)
    warnings_seen = []
    monkeypatch.setattr(
        store,
        "_log_storage_warning",
        lambda action, error: warnings_seen.append((action, str(error))),
    )

    original_connect = store.sqlite3.connect

    def short_timeout_connect(path, **kwargs):
        kwargs["timeout"] = 0.001
        return original_connect(path, **kwargs)

    monkeypatch.setattr(store.sqlite3, "connect", short_timeout_connect)

    def write(index):
        session_id = f"call-{index}"
        store.update_session(session_id, [{"role": "user", "content": str(index)}])
        store.save_order(session_id, "grocery", str(index), [{"name": "milk"}])

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(64)))

    assert warnings_seen == []
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 64
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 64


def test_agent_session_cache_is_bounded_lru_and_durable(monkeypatch, tmp_path):
    agent = _fresh_agent(monkeypatch, tmp_path)
    monkeypatch.setattr(agent, "_SESSION_CACHE_MAX", 2)
    agent._sessions.clear()

    histories = {
        session_id: [{"role": "user", "content": session_id}]
        for session_id in ("a", "b", "c")
    }
    for session_id, history in histories.items():
        agent.update_session(session_id, history)

    assert list(agent._sessions) == ["b", "c"]
    assert agent.get_session("b") == histories["b"]
    agent.update_session("a", histories["a"])

    assert list(agent._sessions) == ["b", "a"]
    assert len(agent._sessions) == 2
    assert agent.get_session("c") == histories["c"]
