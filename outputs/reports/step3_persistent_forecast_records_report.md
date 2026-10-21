# Step 3 Report: Persistent Forecast Records and CSV Input Management

**Project:** Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information  
**Implementation Date:** October 11, 2026  
**Deliverable:** `outputs/reports/step3_persistent_forecast_records_report.md`  
**Endpoints Added:**
- `POST /api/v1/forecasts/upload-csv`
- `GET /api/v1/forecasts/records`
- `GET /api/v1/forecasts/records/{run_id}`
- `GET /api/v1/forecasts/records/{run_id}/status`
**Status:** Successfully Implemented, Verified, and Audited (53/53 Backend Tests Passing; 0 Servers Started)

---

## 1. Database Architecture and Schema

To provide durable, zero-configuration persistence on a single-machine deployment without external database server overhead, the application uses Python standard library `sqlite3` with Write-Ahead Logging (`PRAGMA journal_mode=WAL`).

### 1.1 Physical Storage
- **File Location:** `data/forecast_records.db` (configurable via `DATABASE_PATH` environment variable).
- **Concurrency & Integrity:** WAL mode with foreign keys enabled, row factory set to `sqlite3.Row`, and transactional execution via context managers to prevent partial or corrupted writes.

### 1.2 Table Definition: `forecast_runs`

```sql
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'succeeded', 'failed')),
    input_method TEXT NOT NULL CHECK(input_method IN ('api', 'csv')),
    user_id TEXT,
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
```

### 1.3 State Lifecycle
1. **Pending:** Instantiated immediately upon API ingestion with `run_id`, UTC timestamp, `input_method` (`api` or `csv`), and input metadata summary.
2. **Succeeded:** Updated atomically after PyTorch forward pass, storing 24 point predictions in Watts, 24 conformal interval steps, and model metadata.
3. **Failed:** Recorded whenever input validation or neural inference encounters an error. Stores sanitized error code and error message without fabricated predictions or success flags.

---

## 2. Privacy and Data Governance

Household smart meter active power profiles can expose sensitive occupancy and appliance usage patterns. The persistence layer enforces strict privacy rules:

1. **Zero Raw Time Series Persistence:**
   Complete 168-hour active power and weather sequences are **never stored** in the database.
2. **Metadata-Only Tracking:**
   The `input_metadata_json` column stores only aggregate summary statistics:
   ```json
   {
     "electricity_points": 168,
     "timestamps_supplied": true,
     "weather_supplied": true,
     "weather_features": ["temp", "feels_like", "humidity", "pressure", "speed", "wind_sin", "wind_cos"],
     "source_filename": "uploaded_data.csv"
   }
   ```
3. **Pseudonymous Attribution:**
   `user_id` is an optional, pseudonymous identifier. Authentication and tenant isolation will be implemented in future phases. A `run_id` alone does not represent secure authorization.

---

## 3. CSV Input Management Contract

The new endpoint `POST /api/v1/forecasts/upload-csv` provides robust, validated ingestion of user-uploaded CSV files.

### 3.1 Input Requirements
1. **Row Count:** Exactly 168 rows (7 calendar days of continuous hourly history).
2. **File Size Limit:** 5 MB maximum. Must be UTF-8 encoded plain text.
3. **Active Power Column:** Must contain either `hourly_active_power` or `active_power`. Values must be finite real numbers (no NaN, null, or Inf).
4. **Timestamp Column (Optional):** Supports `date`, `timestamp`, `datetime`, or `time`.
   - Must be chronologically sorted ascending.
   - Must not contain duplicate timestamps.
   - Must follow strictly consecutive 1-hour spacing (`timedelta == 1 hour`). Gaps or irregular sampling frequencies are rejected with descriptive line numbers.
5. **Weather Features (Optional):**
   - For multimodal models (**Early Fusion**, **Late Fusion**, **Hybrid Fusion**), all 7 weather features must be present: `temp`, `feels_like`, `humidity`, `pressure`, `speed`, and either (`wind_sin` and `wind_cos`) or (`deg` $\in [0, 360]$).
   - If `deg` is provided, circular decomposition ($\sin(\text{deg} \cdot \pi / 180)$, $\cos(\text{deg} \cdot \pi / 180)$) is computed automatically prior to scaling.
   - Partial weather columns (e.g. `temp` provided without `pressure`) are rejected with HTTP 422 to prevent silent degradation.
   - If weather columns are absent, the automatic model selector chooses **Energy-Only**.

---

## 4. Complete Endpoint Directory

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/forecasts/run` | Execute live inference from structured JSON payload and persist execution record. |
| `POST` | `/api/v1/forecasts/upload-csv` | Multipart CSV upload: validate 168-row file, run live inference, and persist record with `input_method='csv'`. |
| `GET` | `/api/v1/forecasts/records` | Paginated query of recent forecast summaries with optional status and model filters. |
| `GET` | `/api/v1/forecasts/records/{run_id}` | Retrieve complete forecast run details, 24 point predictions, and conformal intervals. |
| `GET` | `/api/v1/forecasts/records/{run_id}/status` | Lightweight status polling endpoint (`pending`, `succeeded`, `failed`). |
| `GET` | `/api/v1/forecasts/test-windows` | *(Preserved)* Out-of-sample test partition metadata (587 windows). |
| `GET` | `/api/v1/forecasts/test-window/{window_idx}` | *(Preserved)* Test partition multi-model rollout for window index. |

---

## 5. Files Created and Modified

| File | Status | Description |
| :--- | :--- | :--- |
| [`backend/app/db/database.py`](file:///e:/reliable-household-energy-forecasting/backend/app/db/database.py) | **Created** | SQLite connection manager, WAL mode configuration, and idempotent table initialization. |
| [`backend/app/db/__init__.py`](file:///e:/reliable-household-energy-forecasting/backend/app/db/__init__.py) | **Created** | Database package entrypoint. |
| [`backend/app/schemas/records.py`](file:///e:/reliable-household-energy-forecasting/backend/app/schemas/records.py) | **Created** | Pydantic schemas for `ForecastRecordSummary`, `ForecastRecordDetail`, `ForecastRecordListResponse`, and `ForecastStatusResponse`. |
| [`backend/app/services/record_service.py`](file:///e:/reliable-household-energy-forecasting/backend/app/services/record_service.py) | **Created** | Durable CRUD service for forecast runs with privacy-safe metadata tracking. |
| [`backend/app/services/csv_parser_service.py`](file:///e:/reliable-household-energy-forecasting/backend/app/services/csv_parser_service.py) | **Created** | CSV validation service enforcing 168 rows, non-finite checks, duplicate detection, and wind conversion. |
| [`backend/app/core/config.py`](file:///e:/reliable-household-energy-forecasting/backend/app/core/config.py) | **Modified** | Added `DATABASE_PATH` property pointing to `data/forecast_records.db`. |
| [`backend/app/schemas/forecasts.py`](file:///e:/reliable-household-energy-forecasting/backend/app/schemas/forecasts.py) | **Modified** | Added optional `user_id` to `ForecastRunRequest`. |
| [`backend/app/routers/forecasts.py`](file:///e:/reliable-household-energy-forecasting/backend/app/routers/forecasts.py) | **Modified** | Integrated durable persistence into `/run`, added `/upload-csv`, `/records`, `/records/{run_id}`, and `/records/{run_id}/status`. |
| [`backend/app/main.py`](file:///e:/reliable-household-energy-forecasting/backend/app/main.py) | **Modified** | Ensured database schema initialization during `create_app()`. |
| [`backend/requirements.txt`](file:///e:/reliable-household-energy-forecasting/backend/requirements.txt) | **Modified** | Added `python-multipart>=0.0.9`. |
| [`backend/tests/test_persistent_forecast_records.py`](file:///e:/reliable-household-energy-forecasting/backend/tests/test_persistent_forecast_records.py) | **Created** | 15 comprehensive unit and integration tests covering database, CSV upload, pagination, and privacy. |
| [`outputs/reports/step3_persistent_forecast_records_report.md`](file:///e:/reliable-household-energy-forecasting/outputs/reports/step3_persistent_forecast_records_report.md) | **Created** | Comprehensive Step 3 deliverable report. |

---

## 6. Verification and Test Results

### 6.1 Persistent Records Test Suite (`backend/tests/test_persistent_forecast_records.py`)
Ran `python -m unittest backend/tests/test_persistent_forecast_records.py`:
- `test_database_initialization_and_reconnection`: **PASSED** (Schema creation and persistence across reconnection).
- `test_forecast_run_persists_succeeded_record`: **PASSED** (`POST /run` persists record with 24 predictions and conformal bounds).
- `test_failed_run_records_failure_without_fabricated_data`: **PASSED** (Failed validation persists error details without fabricated predictions).
- `test_stored_records_omit_raw_time_series`: **PASSED** (Verified input metadata stores summary stats only, omitting raw series).
- `test_csv_upload_energy_only_success`: **PASSED** (CSV upload with active power executes Energy-Only under auto mode).
- `test_csv_upload_multimodal_with_deg_success`: **PASSED** (CSV upload with weather and wind degrees executes Early Fusion).
- `test_csv_upload_wrong_row_count`: **PASSED** (HTTP 422 on CSV length $\ne 168$).
- `test_csv_upload_missing_electricity_column`: **PASSED** (HTTP 422 on missing active power column).
- `test_csv_upload_duplicate_timestamps`: **PASSED** (HTTP 422 on duplicate timestamp rows).
- `test_csv_upload_non_hourly_intervals`: **PASSED** (HTTP 422 on timestamp interval discontinuity).
- `test_csv_upload_nan_values`: **PASSED** (HTTP 422 on empty or NaN cells).
- `test_csv_upload_partial_weather_rejected`: **PASSED** (HTTP 422 on partial weather features).
- `test_records_pagination`: **PASSED** (Pagination limit, offset, and total count).
- `test_get_nonexistent_record_returns_404`: **PASSED** (HTTP 404 on missing run_id).
- `test_research_endpoints_still_functional`: **PASSED** (Clean benchmarks, robustness matrix, statistics, XAI, conformal, and test windows verified).

**Result:** `15 tests passed in 3.219s (OK)`.

### 6.2 Complete Backend Test Battery
Ran `python -m unittest discover -s backend/tests`:
- `test_health.py`: 4 tests passed.
- `test_artifact_endpoints.py`: 16 tests passed.
- `test_model_inference_audit.py`: 6 tests passed.
- `test_live_inference_api.py`: 12 tests passed.
- `test_persistent_forecast_records.py`: 15 tests passed.

**Total Test Count:** **53 tests passed in 4.369s — OK (100% pass rate).**

### 6.3 Frontend Production Build Check
Ran `npm run build` in `frontend/`:
- TypeScript compilation (`tsc -b`): **Clean, 0 errors**.
- Vite production bundle: **Built in 1m 4s (`dist/assets/index-D9tdg2Y_.js` 699.55 kB)**.

---

## 7. Remaining Limitations

1. **Authentication & Authorization:** Currently, records are indexed by `run_id` and optional pseudonymous `user_id`. Tenant-isolated authorization and secure session management remain to be implemented.
2. **Batch / Multiple Window CSV Ingestion:** The CSV parser strictly validates a single 168-hour rolling window. Multiple-window continuous streaming from a large historical CSV is not supported in this endpoint.
3. **Frontend Console Integration:** Durable records are queryable via REST APIs, but the React frontend developer-facing console and history views have not yet been built.
