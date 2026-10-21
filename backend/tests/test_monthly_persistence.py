"""
Unit and API integration tests for persistent monthly forecast runs (Task 2.4 & Task 2.4B).

Verifies:
- Saving and retrieving completed monthly forecast runs with complete fidelity.
- Server-authoritative forecast generation on save (backend determines predictions and bill estimates).
- Rejection of client-supplied forged responses, prediction overrides, or extra fields (HTTP 422).
- Durable persistence across database reconnection.
- User-scoped listing and pagination (owner enforcement).
- Prevention of record-ID enumeration: uniform 404 Not Found for missing vs foreign-owned records.
- Preservation of developer-role access policy (developers can view all user records).
- Concurrent duplicate save protection (sqlite3.IntegrityError handled cleanly without HTTP 500).
- Idempotent repeated saves of the same logical forecast run.
- Prevention of record hijacking on save (HTTP 404 on collision with another user's record).
- Input validation bounds (max records, horizon range, finite numbers, string length limits).
- Unauthenticated access protection (HTTP 401 Unauthorized).
- Multiple forecast horizons (1, 3, 6 months) with tariff disclosures.
"""

import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.db.database import DatabaseManager
from backend.app.main import app
from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import (
    MonthlyForecastRequest,
    SaveMonthlyForecastRequest,
    compute_monthly_input_fingerprint,
)
from backend.app.services.monthly_forecast_service import monthly_forecast_service
from backend.app.services.monthly_record_service import MonthlyRecordService, monthly_record_service
from backend.app.services.user_service import user_service


class TestMonthlyForecastPersistenceUnit(unittest.TestCase):
    """Unit tests for MonthlyRecordService with isolated SQLite databases."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_monthly.db"
        self.db_mgr = DatabaseManager(self.db_path)
        self.service = MonthlyRecordService(self.db_mgr)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_01_save_and_retrieve_monthly_forecast_run(self):
        """1. Save and retrieve a complete monthly forecast run with full fidelity."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                    period_label="Apr 2024",
                )
            ],
            horizon_months=3,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=5.0,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)

        owner_id = f"user_{uuid.uuid4().hex[:8]}"
        saved = self.service.save_monthly_forecast(
            owner_user_id=owner_id,
            request=req,
            response=res,
        )

        self.assertEqual(saved.run_id, res.forecast_id)
        self.assertEqual(saved.owner_user_id, owner_id)
        self.assertEqual(saved.horizon_months, 3)
        self.assertEqual(saved.provider_id, "msedcl")
        self.assertEqual(saved.phase_type, "single_phase")
        self.assertEqual(saved.sanctioned_load_kw, 5.0)
        self.assertEqual(saved.total_forecast_kwh, 501.0)
        self.assertEqual(len(saved.forecast_steps), 3)
        self.assertEqual(len(saved.input_records), 1)
        self.assertEqual(saved.input_records[0].consumption_kwh, 167.0)

        # Retrieve again
        fetched = self.service.get_monthly_record(res.forecast_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.run_id, res.forecast_id)
        self.assertEqual(fetched.owner_user_id, owner_id)
        self.assertEqual(fetched.summary.total_forecast_kwh, 501.0)
        self.assertEqual(fetched.summary.months_with_bill_estimates, 3)
        self.assertEqual(fetched.summary.total_estimated_bill_inr, 4907.73)

    def test_02_persistence_after_database_reconnection(self):
        """2. Persisted runs remain completely intact after service/connection recreation."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)
        owner_id = "user_reconnect_test"

        self.service.save_monthly_forecast(owner_id, req, res)

        # Simulate complete application restart with new DatabaseManager pointing to same file
        reconnected_db = DatabaseManager(self.db_path)
        reconnected_service = MonthlyRecordService(reconnected_db)

        record = reconnected_service.get_monthly_record(res.forecast_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.run_id, res.forecast_id)
        self.assertEqual(record.total_forecast_kwh, 167.0)

    def test_03_empty_history_returns_zero(self):
        """3. Listing history when no records exist returns total_count: 0 and empty list."""
        total, records = self.service.list_monthly_records(owner_user_id="non_existent_user")
        self.assertEqual(total, 0)
        self.assertEqual(len(records), 0)

    def test_04_duplicate_submission_protection_idempotency(self):
        """4. Re-saving the same run_id for the same owner returns the record idempotently without duplicates."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)
        owner_id = "user_idempotent_test"

        first_save = self.service.save_monthly_forecast(owner_id, req, res)
        second_save = self.service.save_monthly_forecast(owner_id, req, res)

        self.assertEqual(first_save.run_id, second_save.run_id)

        # Total count for this user in DB should still be exactly 1
        total, records = self.service.list_monthly_records(owner_user_id=owner_id)
        self.assertEqual(total, 1)

    def test_05_prevention_of_record_hijacking_returns_404(self):
        """5. A different user cannot overwrite another user's run_id and receives 404 (preventing enumeration)."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)

        self.service.save_monthly_forecast("user_original", req, res)

        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast("user_attacker", req, res)
        # 404 Not Found to prevent leaking that user_original's record exists
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.detail.lower())

    def test_06_concurrent_duplicate_submission_handling(self):
        """6. Concurrent insert race condition caught cleanly without 500 error."""
        import concurrent.futures

        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)
        owner_id = "user_concurrent_test"

        # Execute 4 concurrent saves with the same run_id simultaneously
        results = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.service.save_monthly_forecast, owner_id, req, res)
                for _ in range(4)
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as exc:
                    errors.append(exc)

        # All concurrent workers must succeed without unhandled exceptions or 500s
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent saves: {errors}")
        self.assertEqual(len(results), 4)

        # All results must return the exact same run_id
        for r in results:
            self.assertEqual(r.run_id, res.forecast_id)
            self.assertEqual(r.total_forecast_kwh, 501.0)

        # Database must have exactly ONE row for this run_id
        conn = self.db_mgr.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as count FROM monthly_forecast_runs WHERE run_id = ?;", (res.forecast_id,))
            self.assertEqual(cur.fetchone()["count"], 1)
        finally:
            conn.close()

    def test_06b_simulated_uniqueness_collision_recovery(self):
        """6b. Service catches sqlite3.IntegrityError on INSERT, re-queries, verifies owner, and succeeds."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)
        owner_id = "user_simulated_collision"

        # First save to populate the DB
        initial = self.service.save_monthly_forecast(owner_id, req, res)

        # Now simulate a race condition where the pre-check SELECT returns None,
        # but the subsequent INSERT raises sqlite3.IntegrityError because another worker committed first.
        real_conn = self.db_mgr.get_connection()

        class SimulatedCollisionConnection:
            def __init__(self, target_conn):
                self._conn = target_conn
                self._first_select_bypassed = False

            def cursor(self):
                real_cur = self._conn.cursor()
                parent = self
                class InterceptedCursor:
                    def execute(self, sql, params=()):
                        return real_cur.execute(sql, params)
                    def fetchone(self):
                        if not parent._first_select_bypassed:
                            parent._first_select_bypassed = True
                            return None  # Simulate race condition: initial check sees no row!
                        return real_cur.fetchone()
                    def __getattr__(self, name):
                        return getattr(real_cur, name)
                return InterceptedCursor()

            def __enter__(self):
                # When attempting with conn:, raise IntegrityError simulating concurrent insert
                raise sqlite3.IntegrityError("UNIQUE constraint failed: monthly_forecast_runs.run_id")

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def close(self):
                self._conn.close()

            def __getattr__(self, name):
                return getattr(self._conn, name)

        with patch.object(self.db_mgr, "get_connection", return_value=SimulatedCollisionConnection(real_conn)):
            recovered = self.service.save_monthly_forecast(owner_id, req, res)
            self.assertEqual(recovered.run_id, initial.run_id)
            self.assertEqual(recovered.owner_user_id, owner_id)

    def test_07_concurrent_race_condition_different_owner_returns_404(self):
        """7. A concurrent race condition colliding with another user's run_id returns 404."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req)

        # Save under User 1
        self.service.save_monthly_forecast("user_legit", req, res)

        # Attempt to save under User 2 with same run_id: raises 404 Not Found
        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast("user_intruder", req, res)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.detail.lower())

    def test_08_server_authoritative_generation_when_response_omitted(self):
        """8. If response is omitted in service call, backend generates it authoritatively."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        saved = self.service.save_monthly_forecast(
            owner_user_id="user_auto_gen",
            request=req,
            response=None,
        )
        self.assertIsNotNone(saved.run_id)
        self.assertTrue(saved.run_id.startswith("mfc_"))
        self.assertEqual(saved.total_forecast_kwh, 501.0)
        self.assertEqual(len(saved.forecast_steps), 3)

    def test_08b_same_id_conflicting_consumption_raises_409(self):
        """8b. Re-saving same run_id with different consumption records raises HTTP 409 Conflict."""
        req_orig = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req_orig)
        owner_id = "user_conflict_test"

        # Initial save succeeds
        saved = self.service.save_monthly_forecast(owner_id, req_orig, res)
        self.assertEqual(saved.total_forecast_kwh, 501.0)

        # Attempt to re-save with same run_id but different consumption
        req_diff = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=300.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )

        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast(owner_id, req_diff, None, forecast_id=res.forecast_id)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("already exists with different input", ctx.exception.detail)

        # Confirm original record is preserved intact
        current = self.service.get_monthly_record(res.forecast_id)
        self.assertIsNotNone(current)
        self.assertEqual(current.total_forecast_kwh, 501.0)
        self.assertEqual(current.input_records[0].consumption_kwh, 167.0)

    def test_08c_same_id_conflicting_horizon_or_provider_raises_409(self):
        """8c. Re-saving same run_id with changed horizon, provider, or parameters raises 409."""
        base_record = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 4, 1),
            billing_period_end=date(2024, 4, 30),
            consumption_kwh=167.0,
        )
        req_base = MonthlyForecastRequest(
            records=[base_record],
            horizon_months=1,
            is_live=False,
        )
        res = monthly_forecast_service.generate_forecast(req_base)
        owner_id = "user_param_conflict"
        self.service.save_monthly_forecast(owner_id, req_base, res)

        # 1. Changed horizon (1 -> 3)
        req_h3 = MonthlyForecastRequest(records=[base_record], horizon_months=3, is_live=False)
        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast(owner_id, req_h3, None, forecast_id=res.forecast_id)
        self.assertEqual(ctx.exception.status_code, 409)

        # 2. Changed provider (None -> msedcl)
        req_prov = MonthlyForecastRequest(records=[base_record], horizon_months=1, provider_id="msedcl", is_live=False)
        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast(owner_id, req_prov, None, forecast_id=res.forecast_id)
        self.assertEqual(ctx.exception.status_code, 409)

        # 3. Changed phase_type (single_phase -> three_phase)
        req_phase = MonthlyForecastRequest(records=[base_record], horizon_months=1, phase_type="three_phase", is_live=False)
        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast(owner_id, req_phase, None, forecast_id=res.forecast_id)
        self.assertEqual(ctx.exception.status_code, 409)

        # 4. Changed is_live (False -> True)
        req_live = MonthlyForecastRequest(records=[base_record], horizon_months=1, is_live=True)
        with self.assertRaises(HTTPException) as ctx:
            self.service.save_monthly_forecast(owner_id, req_live, None, forecast_id=res.forecast_id)
        self.assertEqual(ctx.exception.status_code, 409)

    def test_08d_simulated_collision_recovery_with_conflicting_input_raises_409(self):
        """8d. SQLite uniqueness collision recovery rejects conflicting input with 409 Conflict."""
        req_a = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )
        res_a = monthly_forecast_service.generate_forecast(req_a)
        owner_id = "user_sim_conflict_recovery"

        # Worker 1 saved req_a into DB
        self.service.save_monthly_forecast(owner_id, req_a, res_a)

        # Worker 2 attempts to save conflicting req_b with the same run_id
        req_b = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=999.0,
                )
            ],
            horizon_months=3,
            is_live=False,
        )

        real_conn = self.db_mgr.get_connection()

        class SimulatedConflictCollisionConnection:
            def __init__(self, target_conn):
                self._conn = target_conn
                self._first_select_bypassed = False

            def cursor(self):
                real_cur = self._conn.cursor()
                parent = self
                class InterceptedCursor:
                    def execute(self, sql, params=()):
                        return real_cur.execute(sql, params)
                    def fetchone(self):
                        if not parent._first_select_bypassed:
                            parent._first_select_bypassed = True
                            return None  # Simulate race condition: initial check sees no row!
                        return real_cur.fetchone()
                    def __getattr__(self, name):
                        return getattr(real_cur, name)
                return InterceptedCursor()

            def __enter__(self):
                raise sqlite3.IntegrityError("UNIQUE constraint failed: monthly_forecast_runs.run_id")

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def close(self):
                self._conn.close()

            def __getattr__(self, name):
                return getattr(self._conn, name)

        with patch.object(self.db_mgr, "get_connection", return_value=SimulatedConflictCollisionConnection(real_conn)):
            with self.assertRaises(HTTPException) as ctx:
                self.service.save_monthly_forecast(owner_id, req_b, None, forecast_id=res_a.forecast_id)
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertIn("already exists with different input", ctx.exception.detail)

    def test_08e_concurrent_conflicting_saves_one_succeeds_one_rejected_409(self):
        """8e. Concurrent requests with same forecast_id but conflicting inputs: one succeeds, one gets 409."""
        import concurrent.futures

        target_run_id = f"mfc_concurrent_{uuid.uuid4().hex[:8]}"
        owner_id = "user_concurrent_conflict"

        req_1 = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=100.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )
        req_2 = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=200.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )

        successes = []
        conflicts = []
        other_errors = []

        def worker(req):
            try:
                res = self.service.save_monthly_forecast(owner_id, req, None, forecast_id=target_run_id)
                successes.append(res)
            except HTTPException as e:
                if e.status_code == 409:
                    conflicts.append(e)
                else:
                    other_errors.append(e)
            except Exception as ex:
                other_errors.append(ex)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(worker, req_1)
            f2 = executor.submit(worker, req_2)
            concurrent.futures.wait([f1, f2])

        self.assertEqual(len(other_errors), 0, f"Unexpected errors: {other_errors}")
        self.assertEqual(len(successes), 1, "Exactly one conflicting concurrent request must succeed")
        self.assertEqual(len(conflicts), 1, "The other conflicting concurrent request must be rejected with 409")
        self.assertIn("already exists with different input", conflicts[0].detail)


class TestMonthlyInputFingerprintUnit(unittest.TestCase):
    """Unit tests for deterministic canonical input fingerprinting."""

    def test_fingerprint_deterministic_and_stable_for_equivalent_inputs(self):
        """Equivalent validated inputs produce the exact same stable fingerprint."""
        rec_jan = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=150.0,
            bill_amount=1200.0,
            period_label="Jan 2024",
        )
        rec_feb = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 2, 1),
            billing_period_end=date(2024, 2, 29),
            consumption_kwh=160.0,
            bill_amount=1300.0,
            period_label="Feb 2024",
        )

        # 1. Order independence: [Jan, Feb] vs [Feb, Jan]
        req_order1 = MonthlyForecastRequest(records=[rec_jan, rec_feb], horizon_months=3, is_live=False)
        req_order2 = MonthlyForecastRequest(records=[rec_feb, rec_jan], horizon_months=3, is_live=False)
        self.assertEqual(
            compute_monthly_input_fingerprint(req_order1),
            compute_monthly_input_fingerprint(req_order2),
        )

        # 2. Whitespace and case normalization on provider
        req_prov1 = MonthlyForecastRequest(records=[rec_jan], horizon_months=3, provider_id="msedcl")
        req_prov2 = MonthlyForecastRequest(records=[rec_jan], horizon_months=3, provider_id="  MSEDCL  ")
        self.assertEqual(
            compute_monthly_input_fingerprint(req_prov1),
            compute_monthly_input_fingerprint(req_prov2),
        )

        # 3. Float precision equivalence: 150.0 vs 150.0000
        rec_float1 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=150.0,
        )
        rec_float2 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=150.0000,
        )
        self.assertEqual(
            compute_monthly_input_fingerprint(MonthlyForecastRequest(records=[rec_float1])),
            compute_monthly_input_fingerprint(MonthlyForecastRequest(records=[rec_float2])),
        )

    def test_fingerprint_changes_on_any_input_variation(self):
        """Fingerprint changes whenever any field affecting forecasting changes."""
        base_rec = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=150.0,
            bill_amount=1200.0,
            period_label="Jan 2024",
        )
        base_req = MonthlyForecastRequest(
            records=[base_rec],
            horizon_months=3,
            provider_id="msedcl",
            sanctioned_load_kw=5.0,
            phase_type="single_phase",
            is_live=False,
            reference_date=date(2024, 2, 1),
        )
        base_hash = compute_monthly_input_fingerprint(base_req)

        # 1. Changed consumption
        rec_c = base_rec.model_copy(update={"consumption_kwh": 151.0})
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"records": [rec_c]})))

        # 2. Changed start date
        rec_s = base_rec.model_copy(update={"billing_period_start": date(2024, 1, 2)})
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"records": [rec_s]})))

        # 3. Changed end date
        rec_e = base_rec.model_copy(update={"billing_period_end": date(2024, 1, 30)})
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"records": [rec_e]})))

        # 4. Changed bill amount
        rec_b = base_rec.model_copy(update={"bill_amount": 1250.0})
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"records": [rec_b]})))

        # 5. Changed horizon
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"horizon_months": 6})))

        # 6. Changed provider
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"provider_id": "bescom"})))

        # 7. Changed load
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"sanctioned_load_kw": 10.0})))

        # 8. Changed phase
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"phase_type": "three_phase"})))

        # 9. Changed live mode
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"is_live": True})))

        # 10. Changed reference date
        self.assertNotEqual(base_hash, compute_monthly_input_fingerprint(base_req.model_copy(update={"reference_date": date(2024, 3, 1)})))


class TestMonthlyForecastPersistenceApi(unittest.TestCase):
    """API integration tests for /api/v1/forecasts/monthly/save and records endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Create Household User A
        cls.user_a_name = f"user_a_{uuid.uuid4().hex[:8]}"
        cls.user_a_pass = "ValidPassword123!"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": cls.user_a_name, "password": cls.user_a_pass},
        )
        login_a = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": cls.user_a_name, "password": cls.user_a_pass},
        )
        cls.token_a = login_a.json()["access_token"]
        cls.headers_a = {"Authorization": f"Bearer {cls.token_a}"}

        # Create Household User B
        cls.user_b_name = f"user_b_{uuid.uuid4().hex[:8]}"
        cls.user_b_pass = "ValidPassword123!"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": cls.user_b_name, "password": cls.user_b_pass},
        )
        login_b = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": cls.user_b_name, "password": cls.user_b_pass},
        )
        cls.token_b = login_b.json()["access_token"]
        cls.headers_b = {"Authorization": f"Bearer {cls.token_b}"}

        # Create Developer User
        cls.dev_username = f"dev_user_{uuid.uuid4().hex[:8]}"
        cls.dev_password = "ValidDevPass123!"
        user_service.create_user(
            username=cls.dev_username,
            password=cls.dev_password,
            role="developer",
        )
        login_dev = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": cls.dev_username, "password": cls.dev_password},
        )
        cls.token_dev = login_dev.json()["access_token"]
        cls.headers_dev = {"Authorization": f"Bearer {cls.token_dev}"}

    def _generate_forecast(self, horizon: int = 3, is_live: bool = True) -> tuple:
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                    period_label="Apr 2025",
                )
            ],
            horizon_months=horizon,
            provider_id="msedcl",
            is_live=is_live,
        )
        res = monthly_forecast_service.generate_forecast(req)
        return req, res

    def test_09_unauthenticated_requests_rejected_401(self):
        """9. Unauthenticated requests to save, list, or get records return 401."""
        # Save without auth
        res_save = self.client.post(f"{self.prefix}/forecasts/monthly/save", json={})
        self.assertEqual(res_save.status_code, 401)

        # List without auth
        res_list = self.client.get(f"{self.prefix}/forecasts/monthly/records")
        self.assertEqual(res_list.status_code, 401)

        # Detail without auth
        res_get = self.client.get(f"{self.prefix}/forecasts/monthly/records/mfc_123456789012")
        self.assertEqual(res_get.status_code, 401)

    def test_10_save_and_retrieve_via_api(self):
        """10. Authenticated save and retrieval of monthly forecast via API."""
        req, res = self._generate_forecast(horizon=3, is_live=True)
        payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }

        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(save_res.status_code, 201)
        saved_data = save_res.json()
        self.assertEqual(saved_data["run_id"], res.forecast_id)
        self.assertEqual(saved_data["horizon_months"], 3)
        self.assertEqual(saved_data["total_forecast_kwh"], 501.0)
        # Future Nov 2026 unverified tariff -> total bill is None
        self.assertIsNone(saved_data["total_estimated_bill_inr"])
        self.assertEqual(saved_data["months_with_bill_estimates"], 0)

        # Retrieve by ID
        get_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{res.forecast_id}",
            headers=self.headers_a,
        )
        self.assertEqual(get_res.status_code, 200)
        get_data = get_res.json()
        self.assertEqual(get_data["run_id"], res.forecast_id)
        self.assertEqual(len(get_data["forecast_steps"]), 3)
        self.assertEqual(len(get_data["input_records"]), 1)

    def test_11_forged_client_response_rejected_422(self):
        """11. A forged client-supplied response or extra field cannot be persisted (HTTP 422)."""
        req, res = self._generate_forecast(horizon=3, is_live=True)
        forged_payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
            "response": {
                "forecast_id": res.forecast_id,
                "created_at": "2026-10-20T00:00:00Z",
                "history_length_months": 1,
                "selected_method": "baseline_only",
                "method_description": "Forged",
                "confidence_rating": "high",
                "forecast_steps": [
                    {
                        "month_index": 1,
                        "month_label": "Nov 2026",
                        "period_start": "2026-11-01",
                        "period_end": "2026-11-30",
                        "forecast_kwh": 999999.0,
                    }
                ],
                "summary": {
                    "total_forecast_kwh": 999999.0,
                    "average_monthly_kwh": 999999.0,
                    "total_estimated_bill_inr": 1.0,
                    "currency": "INR",
                },
            },
        }

        # Submitting extra "response" field is strictly rejected by Pydantic extra="forbid"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=forged_payload,
            headers=self.headers_a,
        )
        self.assertEqual(save_res.status_code, 422)
        self.assertIn("extra", save_res.text.lower())

    def test_12_backend_recomputes_forecast_authoritatively(self):
        """12. Backend generates and persists authoritative predictions regardless of client input."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=200.0,
                )
            ],
            horizon_months=1,
            is_live=False,
        )
        # Submit save with only input request (no pre-computed forecast_id)
        payload = {"request": req.model_dump(mode="json")}

        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(save_res.status_code, 201)
        saved = save_res.json()

        # Backend computed 1 month persistence of 200.0 kWh
        self.assertEqual(saved["total_forecast_kwh"], 200.0)
        self.assertEqual(saved["selected_method"], "baseline_only")
        self.assertEqual(saved["confidence_rating"], "low")
        self.assertEqual(len(saved["forecast_steps"]), 1)
        self.assertEqual(saved["forecast_steps"][0]["forecast_kwh"], 200.0)

    def test_13_duplicate_save_remains_idempotent_via_api(self):
        """13. Duplicate save calls with the same forecast_id remain idempotent and succeed without duplicate DB rows."""
        req, res = self._generate_forecast(horizon=1, is_live=True)
        payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }

        # First save
        first_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(first_res.status_code, 201)

        # Second save (e.g. repeated click)
        second_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        # Should return success and identical run_id
        self.assertIn(second_res.status_code, [200, 201])
        self.assertEqual(second_res.json()["run_id"], first_res.json()["run_id"])

    def test_13b_same_id_conflicting_consumption_rejected_409_via_api(self):
        """13b. Saving an existing forecast_id with different consumption is rejected with HTTP 409 Conflict."""
        req, res = self._generate_forecast(horizon=1, is_live=True)
        payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }

        # Initial save succeeds
        save1 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(save1.status_code, 201)
        orig_kwh = save1.json()["total_forecast_kwh"]

        # Attempt to save with same forecast_id but changed consumption
        conflicting_req = req.model_copy(deep=True)
        conflicting_req.records[0].consumption_kwh = 999.0
        conflicting_payload = {
            "request": conflicting_req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }

        save2 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=conflicting_payload,
            headers=self.headers_a,
        )
        self.assertEqual(save2.status_code, 409)
        self.assertIn("already exists with different input", save2.json()["detail"])

        # Verify the saved record was not modified
        get_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{res.forecast_id}",
            headers=self.headers_a,
        )
        self.assertEqual(get_res.status_code, 200)
        self.assertEqual(get_res.json()["total_forecast_kwh"], orig_kwh)

    def test_13c_same_id_conflicting_horizon_or_provider_rejected_409_via_api(self):
        """13c. Saving an existing forecast_id with changed horizon or provider is rejected with HTTP 409."""
        req, res = self._generate_forecast(horizon=1, is_live=True)
        payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }
        save1 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(save1.status_code, 201)

        # 1. Changed horizon (1 -> 3)
        req_h3 = req.model_copy(update={"horizon_months": 3})
        res_h3 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json={"request": req_h3.model_dump(mode="json"), "forecast_id": res.forecast_id},
            headers=self.headers_a,
        )
        self.assertEqual(res_h3.status_code, 409)

        # 2. Changed provider (msedcl -> bescom)
        req_prov = req.model_copy(update={"provider_id": "bescom"})
        res_prov = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json={"request": req_prov.model_dump(mode="json"), "forecast_id": res.forecast_id},
            headers=self.headers_a,
        )
        self.assertEqual(res_prov.status_code, 409)

    def test_13d_cross_user_reusing_id_with_conflicting_input_returns_404_not_409(self):
        """13d. Cross-user ID reuse with conflicting input returns 404 (not 409) to prevent enumeration."""
        req_a, res_a = self._generate_forecast(horizon=1, is_live=True)
        # User A saves
        self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json={"request": req_a.model_dump(mode="json"), "forecast_id": res_a.forecast_id},
            headers=self.headers_a,
        )

        # User B attempts to save using User A's forecast_id with User B's distinct input
        req_b, _ = self._generate_forecast(horizon=3, is_live=True)
        req_b.records[0].consumption_kwh = 555.0
        res_b = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json={"request": req_b.model_dump(mode="json"), "forecast_id": res_a.forecast_id},
            headers=self.headers_b,
        )
        # MUST return 404 Not Found (not 409 Conflict, not 403 Forbidden)
        self.assertEqual(res_b.status_code, 404)
        self.assertIn("not found", res_b.json()["detail"].lower())

    def test_14_cross_user_isolation_and_enumeration_defense(self):
        """14. Cross-user access returns uniform 404 (preventing enumeration), while developers retain access."""
        req, res = self._generate_forecast(horizon=1, is_live=True)
        payload = {
            "request": req.model_dump(mode="json"),
            "forecast_id": res.forecast_id,
        }

        # User A saves a run
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = res.forecast_id

        # 1. User B queries User A's run_id -> 404 Not Found (NOT 403)
        res_b_foreign = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}",
            headers=self.headers_b,
        )
        self.assertEqual(res_b_foreign.status_code, 404)
        self.assertIn("not found", res_b_foreign.json()["detail"].lower())

        # 2. User B queries non-existent run_id -> 404 Not Found with matching message format
        res_b_nonexistent = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/mfc_nonexistent_xyz",
            headers=self.headers_b,
        )
        self.assertEqual(res_b_nonexistent.status_code, 404)
        self.assertIn("not found", res_b_nonexistent.json()["detail"].lower())

        # Response formats are consistent: both use 404 and detail "Monthly forecast record with run_id '...' not found."
        self.assertEqual(
            res_b_foreign.json()["detail"],
            f"Monthly forecast record with run_id '{run_id}' not found.",
        )

        # 3. Developer queries User A's run_id -> 200 OK (developer policy preserved)
        res_dev = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}",
            headers=self.headers_dev,
        )
        self.assertEqual(res_dev.status_code, 200)
        self.assertEqual(res_dev.json()["run_id"], run_id)

    def test_15_list_records_returns_only_authenticated_user_records(self):
        """15. Listing records returns only the records belonging to the authenticated user."""
        # Create a record for User B
        req_b, res_b = self._generate_forecast(horizon=6, is_live=True)
        self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json={"request": req_b.model_dump(mode="json"), "forecast_id": res_b.forecast_id},
            headers=self.headers_b,
        )

        # List records for User B
        list_b = self.client.get(
            f"{self.prefix}/forecasts/monthly/records",
            headers=self.headers_b,
        )
        self.assertEqual(list_b.status_code, 200)
        data_b = list_b.json()
        for rec in data_b["records"]:
            self.assertEqual(rec["owner_user_id"], user_service.get_user_by_username(self.user_b_name)["id"])

        # Developer list sees all runs
        list_dev = self.client.get(
            f"{self.prefix}/forecasts/monthly/records",
            headers=self.headers_dev,
        )
        self.assertEqual(list_dev.status_code, 200)
        self.assertGreaterEqual(list_dev.json()["total_count"], 1)

    def test_16_missing_run_id_returns_404(self):
        """16. Requesting non-existent run ID returns 404 Not Found."""
        res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/mfc_nonexistent999",
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_17_multiple_horizons_persistence(self):
        """17. Saves and retrieves 1-month and 6-month horizons correctly."""
        for h in [1, 6]:
            req, res = self._generate_forecast(horizon=h, is_live=True)
            save_res = self.client.post(
                f"{self.prefix}/forecasts/monthly/save",
                json={"request": req.model_dump(mode="json"), "forecast_id": res.forecast_id},
                headers=self.headers_a,
            )
            self.assertEqual(save_res.status_code, 201)
            record = save_res.json()
            self.assertEqual(record["horizon_months"], h)
            self.assertEqual(len(record["forecast_steps"]), h)
            self.assertEqual(record["total_forecast_kwh"], round(167.0 * h, 2))

    def test_18_oversized_and_malformed_requests_rejected(self):
        """18. Input size limits and malformed values are rejected with HTTP 422."""
        # 1. Horizon out of range (> 12)
        bad_req_horizon = {
            "request": {
                "records": [
                    {
                        "billing_period_start": "2025-04-01",
                        "billing_period_end": "2025-04-30",
                        "consumption_kwh": 167.0,
                    }
                ],
                "horizon_months": 13,
            }
        }
        res_h = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=bad_req_horizon,
            headers=self.headers_a,
        )
        self.assertEqual(res_h.status_code, 422)

        # 2. Negative consumption kWh
        bad_req_neg = {
            "request": {
                "records": [
                    {
                        "billing_period_start": "2025-04-01",
                        "billing_period_end": "2025-04-30",
                        "consumption_kwh": -50.0,
                    }
                ],
                "horizon_months": 3,
            }
        }
        res_neg = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=bad_req_neg,
            headers=self.headers_a,
        )
        self.assertEqual(res_neg.status_code, 422)

        # 3. Excessive string length (> 32 chars for provider_id)
        bad_req_str = {
            "request": {
                "records": [
                    {
                        "billing_period_start": "2025-04-01",
                        "billing_period_end": "2025-04-30",
                        "consumption_kwh": 167.0,
                    }
                ],
                "horizon_months": 3,
                "provider_id": "a" * 50,
            }
        }
        res_str = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=bad_req_str,
            headers=self.headers_a,
        )
        self.assertEqual(res_str.status_code, 422)

        # 4. Excessive record count (> 120 records)
        excessive_records = [
            {
                "billing_period_start": f"2020-01-{i:02d}" if i <= 28 else "2020-01-28",
                "billing_period_end": "2020-01-29",
                "consumption_kwh": 100.0,
            }
            for i in range(1, 130)
        ]
        bad_req_count = {
            "request": {
                "records": excessive_records,
                "horizon_months": 3,
            }
        }
        res_count = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            json=bad_req_count,
            headers=self.headers_a,
        )
        self.assertEqual(res_count.status_code, 422)


if __name__ == "__main__":
    unittest.main()
