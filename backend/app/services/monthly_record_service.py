"""
Monthly Forecast Record Service Layer.

Manages durable SQLite persistence for monthly household electricity forecasts:
- Atomic save of completed forecast runs and immutable input snapshots.
- Idempotent duplicate-submission protection for repeated clicks.
- Owner-level authorization enforcement for listing and retrieving runs.
- Full round-trip fidelity for monthly predictions, uncertainty bounds, and tariff notices.
"""

import json
import sqlite3
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status

from backend.app.db.database import DatabaseManager, db_manager
from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import (
    BacktestMetrics,
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRecordDetail,
    MonthlyForecastRecordSummary,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    MonthlyForecastStep,
    MonthlyForecastSummary,
    compute_monthly_input_fingerprint,
)


class MonthlyRecordService:
    """Service handling persistence and retrieval for monthly household forecast runs."""

    def __init__(self, db: DatabaseManager = db_manager):
        self.db = db
        self.db.init_db()

    def _row_to_summary(self, row: Any) -> MonthlyForecastRecordSummary:
        """Convert an SQLite Row to a MonthlyForecastRecordSummary."""
        warnings_list = json.loads(row["warnings_json"]) if row["warnings_json"] else []
        return MonthlyForecastRecordSummary(
            run_id=row["run_id"],
            created_at=row["created_at"],
            owner_user_id=row["owner_user_id"],
            horizon_months=row["horizon_months"],
            provider_id=row["provider_id"],
            phase_type=row["phase_type"],
            sanctioned_load_kw=row["sanctioned_load_kw"],
            is_live=bool(row["is_live"]),
            selected_method=MonthlyForecastMethod(row["selected_method"]),
            confidence_rating=ConfidenceRating(row["confidence_rating"]),
            history_length_months=row["history_length_months"],
            last_observed_period=row["last_observed_period"],
            total_forecast_kwh=row["total_forecast_kwh"],
            average_monthly_kwh=row["average_monthly_kwh"],
            total_estimated_bill_inr=row["total_estimated_bill_inr"],
            months_with_bill_estimates=row["months_with_bill_estimates"],
            bill_estimate_notes=row["bill_estimate_notes"],
            forecast_period_range=row["forecast_period_range"],
            warnings_count=len(warnings_list),
        )

    def _row_to_detail(self, row: Any) -> MonthlyForecastRecordDetail:
        """Convert an SQLite Row to a complete MonthlyForecastRecordDetail."""
        warnings_list = json.loads(row["warnings_json"]) if row["warnings_json"] else []
        limitations_list = json.loads(row["limitations_json"]) if row["limitations_json"] else []
        steps_raw = json.loads(row["forecast_steps_json"]) if row["forecast_steps_json"] else []
        forecast_steps = [MonthlyForecastStep(**s) for s in steps_raw]
        summary_raw = json.loads(row["summary_json"]) if row["summary_json"] else {}
        summary = MonthlyForecastSummary(**summary_raw)

        backtest = None
        if row["backtest_metrics_json"]:
            backtest_raw = json.loads(row["backtest_metrics_json"])
            backtest = BacktestMetrics(**backtest_raw)

        input_records_raw = json.loads(row["input_records_json"]) if row["input_records_json"] else []
        input_records = [MonthlyConsumptionRecord(**r) for r in input_records_raw]

        last_end = None
        if row["last_observed_end_date"]:
            try:
                last_end = date.fromisoformat(row["last_observed_end_date"])
            except ValueError:
                pass

        return MonthlyForecastRecordDetail(
            run_id=row["run_id"],
            created_at=row["created_at"],
            owner_user_id=row["owner_user_id"],
            horizon_months=row["horizon_months"],
            provider_id=row["provider_id"],
            phase_type=row["phase_type"],
            sanctioned_load_kw=row["sanctioned_load_kw"],
            is_live=bool(row["is_live"]),
            selected_method=MonthlyForecastMethod(row["selected_method"]),
            confidence_rating=ConfidenceRating(row["confidence_rating"]),
            history_length_months=row["history_length_months"],
            last_observed_period=row["last_observed_period"],
            last_observed_end_date=last_end,
            is_stale_history=bool(row["is_stale_history"]),
            total_forecast_kwh=row["total_forecast_kwh"],
            average_monthly_kwh=row["average_monthly_kwh"],
            total_estimated_bill_inr=row["total_estimated_bill_inr"],
            months_with_bill_estimates=row["months_with_bill_estimates"],
            bill_estimate_notes=row["bill_estimate_notes"],
            forecast_period_range=row["forecast_period_range"],
            warnings_count=len(warnings_list),
            method_description=row["method_description"],
            warnings=warnings_list,
            limitations=limitations_list,
            forecast_steps=forecast_steps,
            summary=summary,
            backtest_metrics=backtest,
            input_records=input_records,
        )

    def _matches_input_hash(self, row: Any, target_hash: str) -> bool:
        """Verify whether an SQLite row matches a target input fingerprint."""
        row_keys = row.keys() if hasattr(row, "keys") else []
        if "input_hash" in row_keys and row["input_hash"]:
            return row["input_hash"] == target_hash

        # Fallback for legacy records stored prior to input_hash migration
        try:
            input_records_raw = json.loads(row["input_records_json"]) if row["input_records_json"] else []
            records = [MonthlyConsumptionRecord(**r) for r in input_records_raw]
            legacy_req = MonthlyForecastRequest(
                records=records,
                horizon_months=row["horizon_months"],
                provider_id=row["provider_id"],
                sanctioned_load_kw=row["sanctioned_load_kw"],
                phase_type=row["phase_type"],
                is_live=bool(row["is_live"]),
            )
            legacy_hash = compute_monthly_input_fingerprint(legacy_req)
            return legacy_hash == target_hash
        except Exception:
            return False

    def check_input_match(self, run_id: str, request: MonthlyForecastRequest) -> bool:
        """
        Check whether an existing saved forecast run matches the given validated input request.
        Returns False if the record does not exist or if the inputs do not match.
        """
        target_hash = compute_monthly_input_fingerprint(request)
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM monthly_forecast_runs WHERE run_id = ?;", (run_id,))
            row = cur.fetchone()
            if not row:
                return False
            return self._matches_input_hash(row, target_hash)
        finally:
            conn.close()

    def save_monthly_forecast(
        self,
        owner_user_id: str,
        request: MonthlyForecastRequest,
        response: Optional[MonthlyForecastResponse] = None,
        forecast_id: Optional[str] = None,
    ) -> MonthlyForecastRecordDetail:
        """
        Durably save a completed monthly forecast run.

        Provides duplicate protection: If run_id already exists for the same owner
        with identical validated input, returns the existing record idempotently.
        If run_id already exists for the same owner with conflicting input, raises HTTP 409 Conflict.
        If run_id exists for a different owner, raises HTTP 404 Not Found to prevent enumeration.
        Handles race conditions where concurrent requests attempt to insert the same run_id.
        """
        input_hash = compute_monthly_input_fingerprint(request)

        if response is None:
            from backend.app.services.monthly_forecast_service import monthly_forecast_service
            response = monthly_forecast_service.generate_forecast(request=request)

        if forecast_id:
            response.forecast_id = forecast_id

        run_id = response.forecast_id

        conn = self.db.get_connection()
        try:
            # 1. Check for existing record before attempting insert
            cur = conn.cursor()
            cur.execute("SELECT * FROM monthly_forecast_runs WHERE run_id = ?;", (run_id,))
            existing = cur.fetchone()
            if existing:
                if existing["owner_user_id"] != owner_user_id:
                    # Return 404 instead of 403 to prevent record-ID enumeration
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Monthly forecast record with run_id '{run_id}' not found.",
                    )
                # Same owner: check input match
                if self._matches_input_hash(existing, input_hash):
                    return self._row_to_detail(existing)
                # Conflicting input: reject with 409 Conflict
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Monthly forecast record with run_id '{run_id}' already exists with different input parameters.",
                )

            # Determine period range
            if response.forecast_steps:
                if len(response.forecast_steps) == 1:
                    period_range = response.forecast_steps[0].month_label
                else:
                    period_range = f"{response.forecast_steps[0].month_label} – {response.forecast_steps[-1].month_label}"
            else:
                period_range = None

            warnings_json = json.dumps(response.warnings)
            limitations_json = json.dumps(response.limitations)
            steps_json = json.dumps([s.model_dump(mode="json") for s in response.forecast_steps])
            summary_json = json.dumps(response.summary.model_dump(mode="json"))
            backtest_json = (
                json.dumps(response.backtest_metrics.model_dump(mode="json"))
                if response.backtest_metrics
                else None
            )
            input_records_json = json.dumps([r.model_dump(mode="json") for r in request.records])

            method_val = (
                response.selected_method.value
                if hasattr(response.selected_method, "value")
                else str(response.selected_method)
            )
            conf_val = (
                response.confidence_rating.value
                if hasattr(response.confidence_rating, "value")
                else str(response.confidence_rating)
            )

            # 2. Attempt insert with concurrency conflict handling
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO monthly_forecast_runs (
                            run_id, owner_user_id, created_at, horizon_months, provider_id, phase_type,
                            sanctioned_load_kw, is_live, selected_method, method_description, confidence_rating,
                            history_length_months, last_observed_period, last_observed_end_date, is_stale_history,
                            total_forecast_kwh, average_monthly_kwh, total_estimated_bill_inr,
                            months_with_bill_estimates, bill_estimate_notes, forecast_period_range,
                            warnings_json, limitations_json, forecast_steps_json, summary_json,
                            backtest_metrics_json, input_records_json, input_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            run_id,
                            owner_user_id,
                            response.created_at,
                            request.horizon_months,
                            request.provider_id,
                            request.phase_type,
                            request.sanctioned_load_kw,
                            1 if request.is_live else 0,
                            method_val,
                            response.method_description,
                            conf_val,
                            response.history_length_months,
                            response.last_observed_period,
                            str(response.last_observed_end_date) if response.last_observed_end_date else None,
                            1 if response.is_stale_history else 0,
                            response.summary.total_forecast_kwh,
                            response.summary.average_monthly_kwh,
                            response.summary.total_estimated_bill_inr,
                            response.summary.months_with_bill_estimates,
                            response.summary.bill_estimate_notes,
                            period_range,
                            warnings_json,
                            limitations_json,
                            steps_json,
                            summary_json,
                            backtest_json,
                            input_records_json,
                            input_hash,
                        ),
                    )
            except sqlite3.IntegrityError:
                # Handle race condition where a concurrent request inserted the same run_id
                cur = conn.cursor()
                cur.execute("SELECT * FROM monthly_forecast_runs WHERE run_id = ?;", (run_id,))
                conflict_row = cur.fetchone()
                if conflict_row:
                    if conflict_row["owner_user_id"] != owner_user_id:
                        # Foreign-owned record: return 404 to avoid ID enumeration
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Monthly forecast record with run_id '{run_id}' not found.",
                        )
                    # Same owner: check input match
                    if self._matches_input_hash(conflict_row, input_hash):
                        return self._row_to_detail(conflict_row)
                    # Conflicting input: reject with 409 Conflict
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Monthly forecast record with run_id '{run_id}' already exists with different input parameters.",
                    )
                # Unrelated integrity error: re-raise without converting to success
                raise

            # Retrieve and return saved detail
            saved = self.get_monthly_record(run_id=run_id)
            if not saved:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve saved monthly forecast '{run_id}' immediately after insert.",
                )
            return saved
        finally:
            conn.close()

    def list_monthly_records(
        self,
        owner_user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[MonthlyForecastRecordSummary]]:
        """
        List paginated monthly forecast summaries.

        If owner_user_id is supplied, restricts list to that owner.
        If owner_user_id is None, returns all runs (developer mode).
        """
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            if owner_user_id is not None:
                cur.execute(
                    "SELECT COUNT(*) as total FROM monthly_forecast_runs WHERE owner_user_id = ?;",
                    (owner_user_id,),
                )
                total = cur.fetchone()["total"]

                cur.execute(
                    """
                    SELECT * FROM monthly_forecast_runs
                    WHERE owner_user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (owner_user_id, limit, offset),
                )
                rows = cur.fetchall()
            else:
                cur.execute("SELECT COUNT(*) as total FROM monthly_forecast_runs;")
                total = cur.fetchone()["total"]

                cur.execute(
                    """
                    SELECT * FROM monthly_forecast_runs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                )
                rows = cur.fetchall()

            records = [self._row_to_summary(r) for r in rows]
            return total, records
        finally:
            conn.close()

    def get_monthly_record(self, run_id: str) -> Optional[MonthlyForecastRecordDetail]:
        """Retrieve full details of a saved monthly forecast run by run_id."""
        conn = self.db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM monthly_forecast_runs WHERE run_id = ?;", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_detail(row)
        finally:
            conn.close()


# Default global monthly record service
monthly_record_service = MonthlyRecordService()
