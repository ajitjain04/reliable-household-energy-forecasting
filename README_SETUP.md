# System Setup and Manual Launch Guide

**Project:** Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information  
**Architecture:** FastAPI (Backend) + React / Vite / TypeScript (Frontend) + SQLite (Persistence)

This guide documents the configuration, environment settings, operational workflows, security hardening, and testing procedures for the application.

---

## 1. Prerequisites and Environment Setup

### 1.1 Python Environment (Backend)
- Python 3.11 or higher.
- Create and activate a virtual environment:
  ```bash
  # Windows PowerShell
  python -m venv .venv
  .venv\Scripts\Activate.ps1

  # Linux / macOS
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install backend and scientific dependencies:
  ```bash
  pip install -r backend/requirements.txt
  ```

### 1.2 Node.js Environment (Frontend)
- Node.js 18 or higher with `npm`.
- Install frontend dependencies:
  ```bash
  cd frontend
  npm install
  cd ..
  ```

### 1.3 Electricity Bill Extraction & OCR Platform Requirements
The application provides automated parsing of monthly electricity bills (`.pdf`, `.png`, `.jpg`, `.jpeg`) via `BillParserService` (`backend/app/services/bill_parser_service.py`). Platform support varies depending on document format:

1. **Digital / Text-Based PDFs (Cross-Platform):**
   - Digital PDFs with native text streams are parsed using `pypdf` with standard library `zlib` FlateDecode stream fallback.
   - **Fully Cross-Platform:** Operates independently of OCR on Windows, Linux, and macOS without requiring external OCR engines or language packs.

2. **Scanned PDFs and Image Files (`.png`, `.jpg`, `.jpeg`):**
   - Scanned raster documents require optical character recognition. The current implementation utilizes **Windows Native OCR** (`winrt.windows.media.ocr.OcrEngine`) alongside `pypdfium2` (for PDF rasterization) and `Pillow` (for image orientation and contrast enhancement).
   - **Windows Environment:** Requires Windows 10/11 with the `winrt` Python bindings (installed automatically via `backend/requirements.txt` on `sys_platform == 'win32'`) and an installed Windows OCR language pack (e.g. *English (United States)* or *English (India)* via **Windows Settings > Time & Language > Language & Region > Language Options > Basic Typing / OCR**).
   - **Linux & Non-Windows Environments:** Scanned OCR relies on Windows Runtime APIs and is not supported natively on Linux. When a scanned PDF or image is uploaded in a Linux environment, the parser gracefully reports informative notices:
     - `Windows Native OCR runtime bindings (winrt) are not available. Scanned PDF OCR fallback disabled.`
     - `Windows Native OCR runtime bindings (winrt) are not available. Image OCR disabled.`

3. **Mandatory User Review & Explicit Confirmation:**
   - In accordance with our non-generative safety policy, missing or low-confidence values are **never fabricated or guessed**.
   - Parsed records start as unconfirmed (`confirmed_by_user = false`). Users must review all extracted dates, consumption values (`kWh`), and bill amounts (`₹`) in the interactive editable review table and explicitly confirm them before monthly forecasting is enabled.

4. **Fallback & Manual Entry Workflow:**
   - If OCR is unavailable (e.g., Linux deployments without Windows Runtime OCR) or image clarity is insufficient, users can:
     - Use the built-in **interactive review table** to manually enter or correct billing dates and consumption figures.
     - Upload a digital (vector/text) PDF downloaded directly from their utility provider's online portal.

---

### 1.4 Guided Electricity Account Connection & Provider Portals

The application provides a guided utility connection workflow in the **Connect Account & Bill** section (`/forecast/new`).

#### Operational Workflow (Manual Download and Upload)
1. **Provider Selection:** The user selects their electricity distribution company (Discom).
2. **Contextual Account Guidance:** The application displays provider-specific identifier requirements (e.g., Consumer Number, Billing Unit, K-Number, Account ID, or Service Connection Number).
3. **Open Official Portal in Separate Tab:** Clicking **Open Official Portal** opens the verified utility portal in a new browser tab (`target="_blank" rel="noopener noreferrer"`), keeping the user's active application session and forecast draft intact.
4. **Direct Utility Download:** The user signs into the official utility portal directly, navigates to their billing history, and downloads their official e-bill (PDF or image).
5. **In-App Upload on Same Screen:** Returning to the application tab, the user drags and drops or browses for the downloaded file in the in-app dropzone without navigating away.
6. **Editable Review & Confirmation:** The document is parsed, populated into the review table, and requires explicit user confirmation before forecasting.
7. **Direct Upload Fallback:** Users who already have their bill on disk can toggle **Direct Upload Fallback** to skip the portal guidance.

#### Important Architectural Disclosures
- **Manual Retrieval Only:** Direct automatic bill-history retrieval and automated utility scraping are **not implemented**. Indian state utilities do not offer public, open consumer OAuth/REST APIs for domestic bill retrieval.
- **Zero Credential Storage Guarantee:** The application **never** requests, collects, intercepts, or stores electricity provider passwords, security PINs, or OTPs. Users sign in exclusively on the provider's official domain.
- **No CAPTCHA Bypass or Imitation:** The platform does not emulate browser sessions or attempt to bypass provider security controls.

#### Verified Official Provider Portals

| Provider | Utility Name & Region | Primary Official Portal URL | Primary Consumer Identifier | Notes & Secondary Links |
| :--- | :--- | :--- | :--- | :--- |
| **MSEDCL** | Maharashtra State Electricity Distribution Co. Ltd. (Mahavitaran) | `https://wss.mahadiscom.in/wss/wss` | 12-digit Consumer Number & 4-digit Billing Unit (BU) | Official Web Self-Service (WSS) portal for bill view and download. Monthly billing cycle. |
| **BESCOM** | Bangalore Electricity Supply Company Ltd. (Karnataka) | `https://bescom.karnataka.gov.in` | 10-digit Account ID or RR Number (e.g. `E3-12345`) | Official Consumer Portal. Monthly billing cycle. |
| **JVVNL** | Jaipur Vidyut Vitran Nigam Ltd. (Rajasthan) | `https://www.bijlimitra.com/custumerLoginPage` | 12-digit K-Number | **Primary Portal:** Bijli Mitra Consumer Login Portal for bill view and download.<br>**Optional Quick Payment Link:** `https://pay.billdesk.com/instapayweb/jvvnl` (BillDesk Instapay destination for instant payments only; **not** a bill-download portal). |
| **TANGEDCO / TNPDCL** | Tamil Nadu Generation and Distribution Corp. / Power Distribution Corp. | `https://www.tnebltd.gov.in/` | Service Connection Number (Region + Section + Dist + Service) | Official TNEB Online Bill Portal. Bi-monthly (~60 days) billing cycle. |

---

### 1.5 Tariff Coverage, Rate Schedules & Monetary Estimation Limitations

The platform couples physical energy demand forecasting (`kWh`) with electricity tariff modeling under strict verification boundaries:

1. **MSEDCL LT-I(B) Residential (Verified):**
   - Implements verified Maharashtra Electricity Regulatory Commission (MERC) slab tariff structure for domestic consumers:
     - Energy Charges: Slab 1 (0–100 kWh @ ₹4.41), Slab 2 (101–300 kWh @ ₹9.64), Slab 3 (301–500 kWh @ ₹13.61), Slab 4 (>500 kWh @ ₹15.57).
     - Wheeling Charge: ₹1.17/kWh across all consumption.
     - Fixed Charges: ₹128/month (Single Phase) / ₹438/month (Three Phase).
     - Electricity Duty: 16% on taxable energy, wheeling, and fixed charges.
     - Supports Fuel Adjustment Charges (FAC) and pro-rata cycle scaling for non-30-day billing windows.
   - **Schedule Expiry & Out-of-Window Policy:** The verified MSEDCL LT-I(B) schedule is effective through FY 2024-25 (ending March 31, 2025). When a live monthly forecast projection extends into periods without verified tariff orders (e.g., FY 2026-27), monetary bill estimates (₹) are withheld to avoid fabricating charges.

2. **Unverified Providers (BESCOM, JVVNL, TANGEDCO/TNPDCL):**
   - Rate schedules for BESCOM, JVVNL, and TANGEDCO are **unverified** in the platform.
   - **Monetary Withholding Policy:** To maintain financial accuracy, monetary bill estimates (`₹`) are safely withheld and displayed as `Unavailable` with an explanatory notice. Physical energy demand forecasts (`kWh`) are fully computed, trended, and displayed.

---

## 2. Environment Configuration

### 2.1 Backend Environment (`.env`)
Create a `.env` file in the project root (`E:\reliable-household-energy-forecasting\.env`) to configure backend settings. All variables correspond directly to `Settings` in `backend/app/core/config.py`:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `development` | Deployment environment (`development` or `production`). |
| `DEBUG` | `True` | Enables debug logging and detailed error traces. Set to `False` in production. |
| `HOST` | `127.0.0.1` | Local network binding host. |
| `PORT` | `8000` | Local network binding port. |
| `API_PREFIX` | `/api/v1` | URL prefix for all API endpoints. |
| `JWT_SECRET_KEY` | `reliable_energy_forecasting_secret_key_change_in_prod` | Cryptographic signing secret for HS256 tokens. **Must be overridden in production.** |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` (24h) | JWT access token lifetime in minutes. |
| `INITIAL_DEV_USERNAME` | `""` | Optional username for automatic developer account provisioning on boot. |
| `INITIAL_DEV_PASSWORD` | `""` | Strong password for automatic developer provisioning on boot. |
| `DATABASE_PATH` | `data/forecast_records.db` | Absolute or relative path to SQLite database. |
| `CORS_ORIGINS` | `["http://localhost:5173", ...]` | JSON array of permitted origins for browser CORS headers. |

### 2.2 Frontend Environment (`frontend/.env`)
Optional client overrides:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000/api/v1` | Target endpoint for backend requests. |

---

### 2.3 Production Hardening and Security Configuration

When deploying the application in production, apply the following configuration safeguards:

1. **Override `JWT_SECRET_KEY`:**
   - The development default key (`reliable_energy_forecasting_secret_key_change_in_prod`) must **never** be used in production.
   - Generate a cryptographically secure 256-bit random key:
     ```bash
     openssl rand -hex 32
     ```
   - Assign this generated value to `JWT_SECRET_KEY` in your production environment.

2. **Set `ENVIRONMENT=production` and `DEBUG=False`:**
   - Disables debug endpoints, detailed traceback exposure, and development convenience behaviors.

3. **Restrict `CORS_ORIGINS`:**
   - In production, set `CORS_ORIGINS` to an explicit list containing only your trusted production domain(s) (e.g. `["https://energy.example.com"]`), rather than localhost bindings.

4. **Secure `DATABASE_PATH`:**
   - Ensure the SQLite database directory has restricted file system permissions (e.g., readable/writable only by the application process user).
   - Configure regular snapshot backups of `data/forecast_records.db`.

5. **Clear Bootstrap Credentials:**
   - `INITIAL_DEV_USERNAME` and `INITIAL_DEV_PASSWORD` are intended solely for initial bootstrapping. Once the administrative developer account is created, remove or clear these variables from your production environment.

---

## 3. Developer Account Provisioning

Developer access provides administrative inspection across all user forecast runs, database records, and internal model diagnostics.

### Method A: Administrative CLI (Recommended)
From the project root with the virtual environment active:
```bash
python -m backend.app.cli.manage_users create --username <developer_username> --password <strong_password> --role developer
```

### Method B: Environment Bootstrap
Set the following in `.env` before starting the server:
```env
INITIAL_DEV_USERNAME=admin_dev
INITIAL_DEV_PASSWORD=YourSecurePassword123!
```
The server will automatically bootstrap the developer account during startup if it does not already exist. Clear these variables after first boot.

---

## 4. Manual Server Launch Instructions

Launch each server in a separate terminal:

### Terminal 1: Backend API Server
From the project root:
```bash
# Windows PowerShell
.venv\Scripts\python.exe backend/run_server.py

# Linux / macOS
python backend/run_server.py
```
*Expected console output:*
```
Starting Reliable Multimodal Household Energy Forecasting API v1.0.0
Project root: E:\reliable-household-energy-forecasting
Docs URL: http://127.0.0.1:8000/docs
Health URL: http://127.0.0.1:8000/api/v1/health
Press Ctrl+C to stop.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2: Frontend Web Application
From the `frontend` subdirectory:
```bash
cd frontend
npm run dev
```
*Expected console output:*
```
  VITE v6.4.4  ready in 320 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## 5. Verification Checklist

Once both servers are running:

1. **System Health Check:**
   - Visit `http://127.0.0.1:8000/api/v1/health`. Verify `"status": "ok"` and `"artifacts_present": true`.
2. **Interactive API Documentation:**
   - Visit `http://127.0.0.1:8000/docs` to test endpoints and schema models.
3. **Frontend Application:**
   - Visit `http://localhost:5173`.
   - Verify the top header displays `API v1.0.0 Online` with a green indicator.
4. **Hourly Forecasting Flow:**
   - Register a household account at `http://localhost:5173/login`.
   - Navigate to **New Forecast** (`/forecast/new`).
   - Run a test forecast using the sample 168-hour active power profile.
   - Verify that the 24-hour prediction chart and 90% conformal bands render.
   - Click **Download PDF Report** to verify hourly vector PDF generation.
5. **Guided Account Connection & Monthly Forecasting:**
   - Switch to the **Connect Account & Bill** tab.
   - Select a utility (e.g. MSEDCL or JVVNL). Verify official portal links and identifier fields.
   - Upload a test bill or sample document. Review parsed rows in the editable table.
   - Click **Confirm Extracted Records** to unlock monthly forecasting (1, 3, or 6 months).
   - Generate and download the monthly vector PDF report.
6. **Energy Coach:**
   - Navigate to **Energy Coach** (`/coach`).
   - Test the appliance simulator (AC, Refrigerator, Geyser) and verify baseline household comparison.
7. **Record Ownership Isolation:**
   - Log out, register a second household account, and verify that the first account's runs do not appear in **Forecast History**.
8. **Developer Console:**
   - Log in with developer credentials.
   - Navigate to **Developer Console -> Run Inspector** (`/developer/forecast-records`).
   - Verify all forecast records across the platform are visible with raw JSON inspection.

---

## 6. Testing Guidelines: Safe Unit Tests vs. Persistent Database Tests

The codebase includes distinct test suites designed for different operational safety profiles:

### 6.1 Safe Isolated Backend Unit Tests (Zero Database Mutation)
These suites test pure business logic, Pydantic schemas, OCR parsing, tariff mathematics, and coach physics. They execute using synthetic fixtures, in-memory databases (`:memory:`), or isolated temporary directories (`tempfile.TemporaryDirectory()`). They **never** write to `data/forecast_records.db`.

- **Command (161 tests across 10 service modules):**
  ```bash
  .venv\Scripts\python.exe -m unittest backend.tests.test_bill_schemas backend.tests.test_bill_parser_service backend.tests.test_msedcl_tariff backend.tests.test_tariff_architecture backend.tests.test_coach_service backend.tests.test_monthly_persistence.TestMonthlyForecastPersistenceUnit backend.tests.test_monthly_persistence.TestMonthlyInputFingerprintUnit backend.tests.test_monthly_forecast_service.TestMonthlyForecastService backend.tests.test_health backend.tests.test_artifact_endpoints
  ```
  *Actual Result: Ran 161 tests in ~17s — OK.*

- **Model Checkpoint & Conformal Calibration Audit (6 tests):**
  ```bash
  .venv\Scripts\python.exe -m unittest backend.tests.test_model_inference_audit
  ```
  *Actual Result: Ran 6 tests in ~5s — OK.*

- **Combined Non-Destructive Backend Total:** 167 tests (161 unit/service tests + 6 model audit tests).

### 6.2 Frontend Unit & Integration Tests
Client-side tests run via the native Node.js test runner and verify state machines, validation rules, theme toggles, and security contracts in-memory:

- **Command (37 tests across 3 suites):**
  ```bash
  node --test frontend/tests/auth_workflow.test.mjs frontend/tests/theme_and_saving_plan.test.mjs frontend/tests/monthly_bill_workflow.test.mjs
  ```
  *Actual Result: 37 tests passed in ~200ms.*

- **TypeScript Static Verification:**
  ```bash
  cd frontend
  npx tsc --noEmit
  cd ..
  ```
  *Actual Result: Exited with code 0. Zero errors, zero warnings.*

### 6.3 Caution on Persistent Database Integration Tests
Certain backend integration tests (such as `test_auth_and_pdf.py`, `test_auth_gated_workflow.py`, `test_persistent_forecast_records.py`, `TestMonthlyForecastPersistenceApi`, and `test_coach_api.py`) instantiate `TestClient(app)` against the default `DATABASE_PATH` and insert test accounts and forecast records directly into the SQLite database.

> [!CAUTION]
> **Do not run persistent integration tests against a database containing real or user data.**
> To safely run persistent integration tests, configure a separate disposable database file via the `DATABASE_PATH` environment variable:
> ```bash
> # PowerShell example with disposable database
> $env:DATABASE_PATH = "data/disposable_test.db"
> .venv\Scripts\python.exe -m unittest backend.tests.test_auth_and_pdf
> Remove-Item "data/disposable_test.db*" -ErrorAction SilentlyContinue
> $env:DATABASE_PATH = $null
> ```
