"""
Record Service Layer.

Manages durable persistence for forecast runs using SQLite:
- Creates atomic pending, succeeded, and failed execution records.
- Guarantees privacy: Never persists raw 168-hour time series inputs.
- Provides paginated querying, run status lookups, and detailed run retrieval.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.db.database import DatabaseManager, db_manager
from backend.app.schemas.forecasts import ConformalIntervalStep, ForecastRunResponse
from backend.app.schemas.records import (
    ForecastRecordDetail,
    ForecastRecordSummary,
    ForecastStatusResponse,
)


class ForecastRecordService:
    """Service handling CRUD operations on persistent forecast run records."""

    def __init__(self, db: DatabaseManager = db_manager):
        self.db = db
        # Ensure database and table exist
        self.db.init_db()

    def create_pending_run(
        self,
        run_id: str,
        input_method: str,
        user_id: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        input_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a pending execution record in the database."""
        now_utc = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(input_metadata or {})

        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO forecast_runs (
                        run_id, created_at, status, input_method, user_id, owner_user_id, input_metadata_json
                    ) VALUES (?, ?, 'pending', ?, ?, ?, ?);
                    """,
                    (run_id, now_utc, input_method, user_id, owner_user_id, meta_json),
                )
            return run_id
        finally:
            conn.close()

    def record_success(
        self,
        run_id: str,
        response: ForecastRunResponse,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a pending run with succeeded status, predictions, and uncertainty intervals."""
        preds_json = json.dumps(response.predictions_watts)
        intervals_json = json.dumps([step.model_dump() for step in response.intervals])
        model_meta_json = json.dumps(model_metadata or response.model_metadata.model_dump())

        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE forecast_runs
                    SET status = 'succeeded',
                        selected_model = ?,
                        model_key = ?,
                        selection_mode = ?,
                        selection_reason = ?,
                        forecast_origin_timestamp = ?,
                        predictions_watts = ?,
                        intervals_json = ?,
                        mean_interval_width_watts = ?,
                        model_metadata_json = ?
                    WHERE run_id = ?;
                    """,
                    (
                        response.selected_model,
                        response.model_key,
                        response.selection_mode,
                        response.selection_reason,
                        response.forecast_origin_timestamp,
                        preds_json,
                        intervals_json,
                        response.mean_interval_width_watts,
                        model_meta_json,
                        run_id,
                    ),
                )
        finally:
            conn.close()

    def record_failure(
        self,
        run_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """Update a run with failed status and sanitized error details."""
        conn = self.db.get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE forecast_runs
                    SET status = 'failed',
                        error_code = ?,
                        error_message = ?
                    WHERE run_id = ?;
                    """,
                    (error_code, error_message, run_id),
                )
        finally:
            conn.close()

    def get_record(self, run_id: str) -> Optional[ForecastRecordDetail]:
        """Retrieve full forecast record details by run ID."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM forecast_runs WHERE run_id = ?;", (run_id,))
            row = cur.fetchone()
            if not row:
                return None

            preds = json.loads(row["predictions_watts"]) if row["predictions_watts"] else None
            intervals_raw = json.loads(row["intervals_json"]) if row["intervals_json"] else None
            intervals = (
                [ConformalIntervalStep(**step) for step in intervals_raw]
                if intervals_raw
                else None
            )
            input_meta = (
                json.loads(row["input_metadata_json"]) if row["input_metadata_json"] else None
            )
            model_meta = (
                json.loads(row["model_metadata_json"]) if row["model_metadata_json"] else None
            )

            owner_id = row["owner_user_id"] if "owner_user_id" in row.keys() else None
            return ForecastRecordDetail(
                run_id=row["run_id"],
                created_at=row["created_at"],
                status=row["status"],
                input_method=row["input_method"],
                user_id=row["user_id"],
                owner_user_id=owner_id,
                selected_model=row["selected_model"],
                model_key=row["model_key"],
                selection_mode=row["selection_mode"],
                selection_reason=row["selection_reason"],
                forecast_origin_timestamp=row["forecast_origin_timestamp"],
                predictions_watts=preds,
                intervals=intervals,
                mean_interval_width_watts=row["mean_interval_width_watts"],
                input_metadata=input_meta,
                model_metadata=model_meta,
                error_code=row["error_code"],
                error_message=row["error_message"],
            )
        finally:
            conn.close()

    def get_status(self, run_id: str) -> Optional[ForecastStatusResponse]:
        """Retrieve lightweight run status and metadata."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT run_id, created_at, status, input_method, selected_model, error_code, error_message
                FROM forecast_runs WHERE run_id = ?;
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return ForecastStatusResponse(
                run_id=row["run_id"],
                created_at=row["created_at"],
                status=row["status"],
                input_method=row["input_method"],
                selected_model=row["selected_model"],
                error_code=row["error_code"],
                error_message=row["error_message"],
            )
        finally:
            conn.close()

    def list_records(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        model: Optional[str] = None,
        owner_user_id: Optional[str] = None,
    ) -> Tuple[int, List[ForecastRecordSummary]]:
        """Query paginated forecast summaries ordered chronologically descending, optionally filtered by owner."""
        conn = self.db.get_connection()
        try:
            query = "SELECT * FROM forecast_runs WHERE 1=1"
            count_query = "SELECT COUNT(*) FROM forecast_runs WHERE 1=1"
            params: List[Any] = []

            if status:
                query += " AND status = ?"
                count_query += " AND status = ?"
                params.append(status)

            if model:
                query += " AND (model_key = ? OR selected_model = ?)"
                count_query += " AND (model_key = ? OR selected_model = ?)"
                params.extend([model, model])

            if owner_user_id == "__ANONYMOUS__":
                query += " AND (owner_user_id IS NULL OR owner_user_id = '')"
                count_query += " AND (owner_user_id IS NULL OR owner_user_id = '')"
            elif owner_user_id is not None:
                query += " AND owner_user_id = ?"
                count_query += " AND owner_user_id = ?"
                params.append(owner_user_id)

            # Total count
            cur = conn.cursor()
            cur.execute(count_query, params)
            total_count = cur.fetchone()[0]

            # Paginated query
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?;"
            cur.execute(query, params + [limit, offset])
            rows = cur.fetchall()

            records = [
                ForecastRecordSummary(
                    run_id=r["run_id"],
                    created_at=r["created_at"],
                    status=r["status"],
                    input_method=r["input_method"],
                    user_id=r["user_id"],
                    owner_user_id=r["owner_user_id"] if "owner_user_id" in r.keys() else None,
                    selected_model=r["selected_model"],
                    model_key=r["model_key"],
                    selection_mode=r["selection_mode"],
                    selection_reason=r["selection_reason"],
                    forecast_origin_timestamp=r["forecast_origin_timestamp"],
                    mean_interval_width_watts=r["mean_interval_width_watts"],
                    error_code=r["error_code"],
                    error_message=r["error_message"],
                )
                for r in rows
            ]

            return total_count, records
        finally:
            conn.close()


# Default global service instance
record_service = ForecastRecordService()
