# Step 6 Implementation Report: PDF Reports, Authentication, and Forecast Access Control

**Project Root:** `E:\reliable-household-energy-forecasting`  
**Date:** October 2026  
**Status:** Completed and Verified  
**Backend Test Suite:** 67/67 Tests Passing (`OK`, 8.92s)  
**Frontend Production Build:** Passing (`tsc -b && vite build`, 12.33s, 0 errors)

---

## Executive Summary

Step 6 transitions the Reliable Household Energy Forecasting system from an open diagnostic prototype into a multi-tenant, role-aware application with cryptographic authentication, record-level ownership isolation, and genuine downloadable vector PDF report generation.

Prior to Step 6, the system stored durable forecast records with an optional pseudonymous tracking string (`user_id`), but provided zero server-side authentication or access boundaries. Step 6 resolves this vulnerability by introducing:
1. **Cryptographic Authentication:** Argon2id password hashing and HS256 signed JWT bearer tokens with configurable expiration.
2. **Role-Based Access Control (RBAC):** Distinct `household` and `developer` roles. Household users can register self-service; developer accounts require administrative authorization via CLI or bootstrap environment variables.
3. **Record Ownership Enforcement:** The durable database schema was migrated safely to include `owner_user_id`. Forecast ownership is strictly bound to the authenticated JWT identity, preventing request-body spoofing. Household users can only list, view, and download PDF reports for their own runs.
4. **Legacy Record Preservation:** Existing anonymous records (`owner_user_id IS NULL`) are preserved without arbitrary reassignment. Household users cannot view or take over unassigned records. Developers and legacy callers can inspect them.
5. **Genuine Vector PDF Reports:** An in-memory, publication-grade 2-page PDF generator using Matplotlib's `backend_pdf.PdfPages`. Generates complete point forecasts, 90% conformal bands, model metadata, and statutory uncertainty disclosures directly from persistent records without rerunning inference.
6. **Frontend Integration:** Authentication context, login/register views, protected routes for developer diagnostics, session indicators in the header, and wired "Download PDF Report" buttons on both household and developer inspection screens.

---

## 1. Security Architecture and Inspection

### 1.1 Baseline Vulnerability Audit

Before Step 6, the system security posture was analyzed across backend endpoints and database schemas:
- **Database Schema:** `forecast_runs` stored `user_id` as an untrusted client-supplied text string. No `users` table existed.
- **Record Endpoints:** `GET /api/v1/forecasts/records` and `GET /api/v1/forecasts/records/{run_id}` returned records to any anonymous caller without verification.
- **Inference Endpoints:** `POST /api/v1/forecasts/run` and `POST /api/v1/forecasts/upload-csv` accepted any `user_id` provided in the JSON body or multipart form, allowing arbitrary users to claim or overwrite provenance tags.
- **Developer Console:** Route `/developer/forecast-records` was accessible to anyone with a browser, exposing system logs, failure tracebacks, and internal model parameter counts.

### 1.2 Authentication and Authorization Design

```
                     ┌──────────────────────────────────────────────┐
                     │            Client Request                    │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                           Authorization: Bearer <JWT>?
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │ YES                                           │ NO
                    ▼                                               ▼
         Verify HS256 Token                              current_user = None
         Decode sub, role, exp                           (Anonymous Caller)
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    role: "household"     role: "developer"
    (User A / User B)    (Full System Access)
```

1. **Password Hashing:**
   - Algorithm: `Argon2id` via the `argon2-cffi` library (`PasswordHasher(time_cost=2, memory_cost=65536, parallelism=4)`).
   - Passwords are never stored in plaintext, never logged, and never returned in API payloads.
2. **Token Specification:**
   - Standard: Signed JWT tokens using HMAC-SHA256 (`HS256`).
   - Claims: `sub` (User UUID), `username`, `role`, `iat` (issued at), and `exp` (expiration timestamp).
   - Secret Key: Loaded from `JWT_SECRET_KEY` in `Settings` (defaults to a persistent development secret, overridable via environment variable).
   - Expiration: Default 1440 minutes (24 hours), configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`.
3. **Role Architecture:**
   - `household`: Standard consumer role. Can execute forecasts, view own history, download own PDFs. Blocked from developer console and other users' records.
   - `developer`: Administrative/engineering role. Can inspect all forecast records across the platform, view detailed diagnostic metadata, error logs, and download any generated report.
4. **Administrative Developer Provisioning (Zero Hardcoded Production Credentials):**
   - Developer accounts cannot be created through the public registration API (`POST /auth/register` strictly forces `role="household"`).
   - Developers are provisioned either through the administrative CLI command:
     ```bash
     python -m backend.app.cli.manage_users create --username dev_admin --password <strong_password> --role developer
     ```
   - Or on initial startup via environment variables:
     ```env
     INITIAL_DEV_USERNAME=ops_lead
     INITIAL_DEV_PASSWORD=SecretBootstrapPassword!
     ```

---

## 2. Database Migration and Record Ownership

### 2.1 Schema Definition and Migration

The SQLite database (`backend/data/forecast_records.db`) was updated with an idempotent migration routine executed on engine initialization:

```sql
-- 1. Users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'household',
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- 2. Forecast runs migration (safe ALTER TABLE if missing)
-- Checks PRAGMA table_info(forecast_runs) before altering
ALTER TABLE forecast_runs ADD COLUMN owner_user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_forecast_runs_owner_user_id ON forecast_runs(owner_user_id);
```

### 2.2 Ownership Enforcement Policy

| Caller Identity | `POST /forecasts/run` & `/upload-csv` | `GET /forecasts/records` | `GET /forecasts/records/{run_id}` | `GET /records/{run_id}/pdf` |
| :--- | :--- | :--- | :--- | :--- |
| **Anonymous (Unauthenticated)** | Persists run with `owner_user_id = NULL`. Client-supplied `user_id` stored only as cosmetic tag. | Returns only legacy anonymous records (`owner_user_id IS NULL`). | Allowed if record is anonymous (`owner_user_id IS NULL`). If owned: HTTP 401 Unauthorized. | Allowed if record is anonymous. If owned: HTTP 401 Unauthorized. |
| **Household User A** | Persists run with `owner_user_id = current_user.id`. Spoofed body IDs ignored. | Returns **only** records where `owner_user_id = user_a.id`. | Allowed for own records. HTTP 403 Forbidden for User B or anonymous runs. | Allowed for own records. HTTP 403 Forbidden for User B or anonymous runs. |
| **Developer / Admin** | Persists run with `owner_user_id = dev.id`. | Returns **all** records across system (User A, User B, anonymous). | Allowed for **all** records. | Allowed for **all** successful records. |

### 2.3 Legacy Record Handling

Existing records created during Steps 3–5 or through automated unauthenticated test runs retain `owner_user_id = NULL`.
- They are **not** assigned to any user arbitrarily.
- Household users cannot query or claim legacy records (returns `HTTP 403 Forbidden`).
- Developers can view and audit all legacy records.
- Backward compatibility with prior testing pipelines is maintained.

---

## 3. Genuine PDF Report Generation

### 3.1 Architecture and Implementation

PDF reports are generated on demand via `backend/app/services/pdf_service.py` using `matplotlib.backends.backend_pdf.PdfPages` into an in-memory buffer (`io.BytesIO`).

**Key Architectural Properties:**
- **Zero Re-Inference:** Reports are generated entirely from persisted point forecasts, conformal intervals, timestamps, and model metadata stored in SQLite. No PyTorch forward passes or weather scaler transforms are executed.
- **In-Memory Streaming:** Generated bytes are returned directly via FastAPI `Response(content=pdf_bytes, media_type="application/pdf")`. No temporary files are written to disk.
- **Safe Vector Rendering:** Uses pure vector graphics, ensuring clean printing and rendering on all devices.
- **Attachment Disposition:** Returns `Content-Disposition: attachment; filename="forecast_report_{run_id}.pdf"`.

### 3.2 Report Content Structure

The generated report is a 2-page document containing:

#### Page 1: Executive Dashboard & Forecast Trajectory
1. **Report Header:** Title, Run ID, Generation Timestamp, Forecast Origin, Horizon (24 hours), Active Power Target (Watts).
2. **Selected Model Architecture & Metadata:** Model Name, Architecture Key, Trainable Parameters count, History Length (168h), Input Features used (Electricity + Weather).
3. **Key Performance Indicators (KPIs):** Total Estimated Daily Energy (kWh), Peak Power Load (W), Minimum Power Load (W), Average Power Load (W), Conformal Coverage Level (90.0%), Mean Interval Width (W).
4. **24-Hour Forecast Chart:**
   - Blue step-line representing point predictions $\hat{y}_{t+h}$.
   - Amber shaded region indicating 90% conformal prediction intervals $[L_{t+h}, U_{t+h}]$ with clipping at zero watts.
   - Dual-axis x-ticks showing both horizon hours ($+1\text{h}\dots+24\text{h}$) and target calendar timestamps.
5. **Statutory Uncertainty Disclosure:**
   > *"Prediction intervals represent empirical 90% conformal coverage estimates calibrated across historical validation data. They are statistical bounds and do not constitute deterministic performance guarantees for an individual household."*
6. **Domain Generalization Disclosure:**
   > *"Forecasts reflect the training distribution of the underlying benchmark models. Performance may degrade when applied to households with substantially different load profiles, appliance mixes, or unobserved meteorological conditions."*

#### Page 2: Granular 24-Hour Forecast Schedule & Technical Footprint
1. **Granular Hourly Schedule Table:** All 24 horizon hours tabulated with Target Timestamp, Point Forecast (W), Lower Bound (W), Upper Bound (W), Interval Width (W), and Estimated Energy (kWh).
2. **Technical Provenance Footer:** Input method (`api` or `csv`), conformal calibration method, and hash of run identifier.

---

## 4. Frontend Authentication & Protected Routes

### 4.1 State Management (`AuthContext.tsx`)
- Provides `user: UserProfile | null`, `token: string | null`, `isAuthenticated: boolean`, and `isDeveloper: boolean`.
- Automatically restores session from `localStorage` on page load by querying `GET /api/v1/auth/me`.
- Gracefully clears tokens on 401 unauthorized responses.

### 4.2 User Navigation and UI Elements
- **Header Component:** Displays live user badge (e.g. `household_user (household)` or `dev_admin (developer)`), role indicator, and "Sign Out" button. When unauthenticated, displays "Sign In" link.
- **Sidebar Component:** Clearly separates Household Forecasting, Developer Console, and Research & Benchmarks. If caller is not developer, Developer Console displays a lock icon and "Restricted" status tag.
- **Login / Register Page (`/login`):** Unified modern tabbed view allowing household self-registration or login. Displays clear instructions for developer credentials.
- **Protected Routes (`ProtectedRoute.tsx`):**
  - Developer routes (`/developer/forecast-records` and `/developer/forecast-records/:runId`) require `role="developer"`.
  - Non-developer authenticated callers receive a clean "Authorized Role Required" banner with option to return home or switch accounts.
  - Unauthenticated callers are redirected to `/login` with preserved return path.
- **Public Research Routes:**
  - All 7 research modules (`/overview`, `/benchmarks`, `/reliability`, `/statistical-tests`, `/xai`, `/uncertainty`, `/explorer`) remain accessible to ensure scientific verification without barriers.

### 4.3 Download PDF Action Wiring
- **Household Result Screen (`/forecast/:runId`):** Prominent "Download PDF Report" button added to the header action bar. Calls `api.downloadForecastPdf(runId)` to download the real binary report.
- **Developer Detail Screen (`/developer/forecast-records/:runId`):** Matching "Download PDF" action added to the toolbar, allowing authorized developers to download reports for any successful run.

---

## 5. Verification and Test Results

### 5.1 Backend Test Suite

A dedicated integration test suite `backend/tests/test_auth_and_pdf.py` was created, verifying:
1. Registration, JWT login, and `/auth/me` profile inspection.
2. Rejection of invalid credentials (wrong password, unknown username) with HTTP 401.
3. Passwords stored strictly as Argon2id hashes, never in plaintext.
4. Record ownership isolation:
   - User A can view and download their own run.
   - User B is rejected with `HTTP 403 Forbidden` from User A's run and PDF.
   - User B's record list excludes User A's runs.
   - User ID spoofing in body is prevented (ownership assigned from JWT).
5. Role authorization:
   - Developer accounts can inspect User A's runs and download PDFs.
   - Ordinary household users are rejected from developer endpoints.
6. Legacy anonymous records:
   - Preserved with `owner_user_id IS NULL`.
   - Household users forbidden from claiming anonymous records (HTTP 403).
   - Developer accounts can inspect legacy records.
7. PDF generation and delivery:
   - Valid binary vector header (`%PDF-1.`).
   - `Content-Type: application/pdf`.
   - `Content-Disposition: attachment; filename="forecast_report_{run_id}.pdf"`.
   - File size > 10 KB with full 2-page contents.
   - Missing run returns HTTP 404; failed run returns HTTP 400.
8. Backward compatibility:
   - All 59 existing unit and integration tests from Steps 1–5 pass without regressions.

**Backend Execution Result:**
```
Ran 67 tests in 8.923s
OK
```

### 5.2 Frontend Production Build

The React TypeScript application was verified via production compilation:
```bash
npm run build
```
**Build Output:**
```
> frontend@1.0.0 build
> tsc -b && vite build

vite v6.4.4 building for production...
transforming...
✓ 2223 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.97 kB │ gzip:   0.57 kB
dist/assets/index-DcqNakqC.css   32.82 kB │ gzip:   6.36 kB
dist/assets/index-BXfvFLoP.js   825.62 kB │ gzip: 213.71 kB
✓ built in 12.33s
```
Zero TypeScript errors, zero lint warnings, zero bundle errors.

---

## 6. Documented Limitations and Future Hardening

While Step 6 achieves production-ready access control and PDF report generation for the application's current scope, the following operational limitations should be noted:

1. **Token Revocation (Denylist):** JWT tokens are stateless. Calling `POST /auth/logout` clears the token on the client side; server-side token invalidation before expiration would require a Redis token blocklist or DB-backed session revocation table.
2. **Single SQLite Database:** SQLite is utilized for single-machine deployment with WAL mode and busy timeouts. Under heavy concurrent multi-tenant writes (>100 requests/sec), migrating to PostgreSQL is recommended.
3. **Password Reset Workflow:** There is currently no email-based self-service password reset. Administrative password resets must be performed via the CLI `manage_users` tool.
4. **Rate Limiting:** IP-based brute-force throttling on `/auth/login` is not yet implemented. In an internet-exposed production environment, a reverse proxy (e.g., Nginx, Cloudflare) with rate-limiting rules should be placed in front of the API.
