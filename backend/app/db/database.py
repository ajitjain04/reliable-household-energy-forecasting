"""
SQLite persistence management for durable forecast records and user accounts.

Features:
- Lightweight embedded storage via standard library sqlite3.
- Idempotent and safe table initialization and schema migrations.
- Write-Ahead Logging (WAL) mode for concurrency and reliability.
- Strict transaction isolation to prevent partial or corrupted writes.
- Support for custom database paths (including :memory: for testing).
- User authentication tables and record ownership columns.
"""

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, Union

from backend.app.core.config import settings


CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('household', 'developer')),
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
"""

CREATE_FORECAST_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'succeeded', 'failed')),
    input_method TEXT NOT NULL CHECK(input_method IN ('api', 'csv')),
    user_id TEXT,
    owner_user_id TEXT,
    selected_model TEXT,
    model_key TEXT,
    selection_mode TEXT,
    selection_reason TEXT,
    forecast_origin_timestamp TEXT,
    predictions_watts TEXT,
    intervals_json TEXT,
    mean_interval_width_watts REAL,
    input_metadata_json TEXT,
    model_metadata_json TEXT,
    error_code TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_forecast_runs_created_at ON forecast_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_status ON forecast_runs(status);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_user_id ON forecast_runs(user_id);
"""

CREATE_MONTHLY_FORECAST_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS monthly_forecast_runs (
    run_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    horizon_months INTEGER NOT NULL,
    provider_id TEXT,
    phase_type TEXT,
    sanctioned_load_kw REAL,
    is_live INTEGER NOT NULL DEFAULT 1,
    selected_method TEXT NOT NULL,
    method_description TEXT NOT NULL,
    confidence_rating TEXT NOT NULL,
    history_length_months INTEGER NOT NULL,
    last_observed_period TEXT,
    last_observed_end_date TEXT,
    is_stale_history INTEGER NOT NULL DEFAULT 0,
    total_forecast_kwh REAL NOT NULL,
    average_monthly_kwh REAL NOT NULL,
    total_estimated_bill_inr REAL,
    months_with_bill_estimates INTEGER NOT NULL DEFAULT 0,
    bill_estimate_notes TEXT,
    forecast_period_range TEXT,
    warnings_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    forecast_steps_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    backtest_metrics_json TEXT,
    input_records_json TEXT NOT NULL,
    input_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_monthly_forecast_runs_owner ON monthly_forecast_runs(owner_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_monthly_forecast_runs_created_at ON monthly_forecast_runs(created_at DESC);
"""


class DatabaseManager:
    """Manages SQLite connections, schema initialization, and safe migrations."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        self._db_path = db_path
        self._shared_memory_conn: Optional[sqlite3.Connection] = None
        target = str(db_path) if db_path is not None else str(settings.DATABASE_PATH)
        if target == ":memory:" or target.startswith("file:"):
            if target == ":memory:":
                self._shared_uri = f"file:mem_{uuid.uuid4().hex}?mode=memory&cache=shared"
            else:
                self._shared_uri = target
            self._shared_memory_conn = sqlite3.connect(
                self._shared_uri, uri=True, timeout=20.0, check_same_thread=False
            )
        else:
            self._shared_uri = None

    @property
    def db_path(self) -> Path:
        if self._db_path is not None:
            return self._db_path if isinstance(self._db_path, Path) else Path(self._db_path)
        return settings.DATABASE_PATH

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        if self._shared_uri is not None:
            conn = sqlite3.connect(self._shared_uri, uri=True, timeout=20.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn

        target = str(self.db_path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(target, timeout=20.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for file-based databases
        if target != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")

        return conn

    def init_db(self) -> None:
        """Initialize database tables and indexes safely and idempotently with migrations."""
        conn = self.get_connection()
        try:
            with conn:
                # 1. Initialize Users Table
                conn.executescript(CREATE_USERS_TABLE)

                # 2. Initialize Forecast Runs Table (24h hourly)
                conn.executescript(CREATE_FORECAST_RUNS_TABLE)

                # 3. Initialize Monthly Forecast Runs Table
                conn.executescript(CREATE_MONTHLY_FORECAST_RUNS_TABLE)

                # 4. Safe migration check for existing forecast_runs tables (add owner_user_id if missing)
                cursor = conn.execute("PRAGMA table_info(forecast_runs);")
                columns = [row["name"] for row in cursor.fetchall()]
                if "owner_user_id" not in columns:
                    conn.execute("ALTER TABLE forecast_runs ADD COLUMN owner_user_id TEXT;")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_runs_owner_user_id ON forecast_runs(owner_user_id);")

                # 5. Safe migration check for existing monthly_forecast_runs (add input_hash if missing)
                cursor_m = conn.execute("PRAGMA table_info(monthly_forecast_runs);")
                columns_m = [row["name"] for row in cursor_m.fetchall()]
                if "input_hash" not in columns_m:
                    conn.execute("ALTER TABLE monthly_forecast_runs ADD COLUMN input_hash TEXT;")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_monthly_forecast_runs_input_hash ON monthly_forecast_runs(input_hash);")
        finally:
            conn.close()


# Default global database manager
db_manager = DatabaseManager()
