"""
Forecasts router for precomputed test evaluation rollouts, live inference,
CSV upload management, durable forecast record tracking, and PDF generation.
"""

import io
import uuid
from typing import Any, Dict, Optional
from PIL import Image

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Response,
    UploadFile,
    status as http_status,
)

HTTP_422 = getattr(http_status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
HTTP_413 = getattr(http_status, "HTTP_413_CONTENT_TOO_LARGE", 413)

from backend.app.core.dependencies import get_current_user, require_developer
from backend.app.schemas.forecasts import (
    ForecastRunRequest,
    ForecastRunResponse,
    LocationInput,
    TestWindowsMetadataResponse,
    WindowForecastResponse,
)
from backend.app.schemas.bills import ExtractedBillData
from backend.app.schemas.records import (
    ForecastRecordDetail,
    ForecastRecordListResponse,
    ForecastStatusResponse,
)
from backend.app.services.artifact_service import artifact_service
from backend.app.services.bill_parser_service import (
    ALLOWED_BILL_EXTENSIONS,
    MAX_BILL_FILE_SIZE_BYTES,
    MAX_PDF_SIZE_BYTES,
    bill_parser_service,
)
from backend.app.services.csv_parser_service import csv_parser_service
from backend.app.services.inference_service import inference_service
from backend.app.services.pdf_service import generate_forecast_pdf, generate_monthly_forecast_pdf
from backend.app.services.record_service import record_service

from backend.app.schemas.monthly_forecast import (
    MonthlyForecastRecordDetail,
    MonthlyForecastRecordListResponse,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    SaveMonthlyForecastRequest,
)
from backend.app.services.monthly_forecast_service import monthly_forecast_service
from backend.app.services.monthly_record_service import monthly_record_service

router = APIRouter(prefix="/forecasts", tags=["Forecasts"])


# -----------------------------------------------------------------------------
# 1. Precomputed Research Evaluation Partition Endpoints (Step 10 Retained)
# -----------------------------------------------------------------------------
@router.get(
    "/test-windows",
    response_model=TestWindowsMetadataResponse,
    summary="Get Test Partition Windows Metadata",
    description=(
        "Retrieve metadata describing the 587 rolling evaluation test windows, "
        "including forecast origin bounds, multi-step horizon length, and available models."
    ),
)
def get_test_windows_metadata() -> TestWindowsMetadataResponse:
    """Return test partition evaluation window metadata."""
    return artifact_service.get_test_windows_metadata()


@router.get(
    "/test-window/{window_idx}",
    response_model=WindowForecastResponse,
    summary="Get Multi-Model Forecast Rollouts for Test Window",
    description=(
        "Retrieve ground truth actual power load alongside 24-hour predictions from all four "
        "evaluated models (Energy-Only, Early Fusion, Late Fusion, Hybrid Fusion) and window error "
        "metrics for a specific chronological test window index (0 to 586)."
    ),
)
def get_test_window_forecast(
    window_idx: int = Path(..., ge=0, le=586, description="Test window index (0 to 586)"),
) -> WindowForecastResponse:
    """Return multi-model predictions and actual power for selected test window."""
    return artifact_service.get_test_window_forecast(window_idx=window_idx)


# -----------------------------------------------------------------------------
# 2. Live Inference Endpoints (API Payload & Multipart CSV Upload)
# -----------------------------------------------------------------------------
@router.post(
    "/run",
    response_model=ForecastRunResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Run Live 24-Hour Electricity Forecast with Conformal Uncertainty",
    description=(
        "Execute end-to-end forward inference on user-provided 168-hour history. "
        "Supports automatic model selection (Early Fusion when valid weather is present, "
        "Energy-Only when weather is absent) or authorized explicit overrides. "
        "Durably persists run metadata, ownership, and conformal bounds."
    ),
)
def run_live_forecast(
    payload: ForecastRunRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ForecastRunResponse:
    """Generate 24-hour forecast from user-provided historical observations and persist record."""
    run_id = f"fc_{uuid.uuid4().hex[:12]}"
    owner_id = current_user["id"]

    # Input metadata summary (never saving raw time series arrays)
    input_meta = {
        "electricity_points": len(payload.electricity_active_power) if isinstance(payload.electricity_active_power, list) else 0,
        "timestamps_supplied": payload.timestamps is not None,
        "weather_supplied": payload.weather is not None,
        "model_requested": payload.model,
    }
    if payload.location is not None:
        input_meta["location"] = payload.location.model_dump()

    # Record pending run with authenticated owner if available
    record_service.create_pending_run(
        run_id=run_id,
        input_method="api",
        user_id=payload.user_id,
        owner_user_id=owner_id,
        input_metadata=input_meta,
    )

    try:
        response = inference_service.run_forecast(request=payload, run_id=run_id)
        record_service.record_success(run_id=run_id, response=response)
        return response

    except HTTPException as he:
        record_service.record_failure(
            run_id=run_id,
            error_code=f"HTTP_{he.status_code}",
            error_message=str(he.detail),
        )
        raise

    except Exception as exc:
        record_service.record_failure(
            run_id=run_id,
            error_code="INTERNAL_INFERENCE_ERROR",
            error_message=f"Inference pipeline execution error: {str(exc)}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed: {str(exc)}",
        )


@router.post(
    "/upload-csv",
    response_model=ForecastRunResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Run Live Forecast via Uploaded 168-Hour CSV File",
    description=(
        "Ingests a user-provided CSV file covering exactly 168 hourly steps. "
        "Validates active power and weather columns, checks for missing/non-finite values, "
        "creates a durable pending record, runs forward inference, and updates persistent status."
    ),
)
def upload_forecast_csv(
    file: UploadFile = File(..., description="CSV file containing 168 hourly observations"),
    model: str = Form(default="auto", description="Model selection mode ('auto' or specific architecture key)"),
    user_id: Optional[str] = Form(default=None, description="Optional display tracking tag"),
    latitude: Optional[float] = Form(default=None, description="Optional household latitude (-90 to 90)"),
    longitude: Optional[float] = Form(default=None, description="Optional household longitude (-180 to 180)"),
    location_label: Optional[str] = Form(default=None, description="Optional household location label"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ForecastRunResponse:
    """Validate 168-hour CSV, record pending execution, execute inference, and persist result."""
    run_id = f"fc_{uuid.uuid4().hex[:12]}"
    owner_id = current_user["id"]

    filename = file.filename or "uploaded.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file '{filename}' must have a .csv extension.",
        )

    try:
        content = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file content: {str(e)}",
        )

    try:
        run_request, meta_summary = csv_parser_service.parse_csv(
            file_bytes=content,
            filename=filename,
            model=model,
        )
    except HTTPException as he:
        # Record failed parse attempt
        record_service.create_pending_run(
            run_id=run_id,
            input_method="csv",
            user_id=user_id,
            owner_user_id=owner_id,
            input_metadata={"source_filename": filename, "file_size_bytes": len(content)},
        )
        record_service.record_failure(
            run_id=run_id,
            error_code="CSV_VALIDATION_ERROR",
            error_message=str(he.detail),
        )
        raise

    if (latitude is not None and longitude is None) or (latitude is None and longitude is not None):
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both latitude and longitude must be provided for location-based weather retrieval.",
        )

    if latitude is not None and longitude is not None:
        try:
            loc = LocationInput(latitude=latitude, longitude=longitude, label=location_label)
            run_request.location = loc
            meta_summary["location"] = loc.model_dump()
        except Exception as e:
            raise HTTPException(
                status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid location parameters: {str(e)}",
            )

    # Record pending run
    record_service.create_pending_run(
        run_id=run_id,
        input_method="csv",
        user_id=user_id,
        owner_user_id=owner_id,
        input_metadata=meta_summary,
    )

    try:
        response = inference_service.run_forecast(request=run_request, run_id=run_id)
        record_service.record_success(run_id=run_id, response=response)
        return response

    except HTTPException as he:
        record_service.record_failure(
            run_id=run_id,
            error_code=f"HTTP_{he.status_code}",
            error_message=str(he.detail),
        )
        raise

    except Exception as exc:
        record_service.record_failure(
            run_id=run_id,
            error_code="INTERNAL_INFERENCE_ERROR",
            error_message=f"Inference execution error: {str(exc)}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed: {str(exc)}",
        )


# -----------------------------------------------------------------------------
# 3. Forecast Record Endpoints (History, Developer Console & Access Control)
# -----------------------------------------------------------------------------
@router.get(
    "/records",
    response_model=ForecastRecordListResponse,
    summary="List Forecast Run Records (Owner Enforced)",
    description=(
        "Retrieve paginated forecast run summaries ordered by creation time descending. "
        "Enforces record-level authorization: household users only see their own runs. "
        "Developer users can inspect all records across the platform. "
        "Unauthenticated callers receive anonymous unassigned records."
    ),
)
def list_forecast_records(
    limit: int = Query(default=50, ge=1, le=100, description="Page size limit (1 to 100)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(default=None, description="Filter by status (succeeded, failed, pending)"),
    model: Optional[str] = Query(default=None, description="Filter by model key or display name"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ForecastRecordListResponse:
    """Return paginated list of forecast run summaries with owner-level authorization."""
    if current_user.get("role") == "developer":
        owner_filter = None
    else:
        owner_filter = current_user["id"]

    total, records = record_service.list_records(
        limit=limit,
        offset=offset,
        status=status,
        model=model,
        owner_user_id=owner_filter,
    )
    return ForecastRecordListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        records=records,
    )


@router.get(
    "/developer/records",
    response_model=ForecastRecordListResponse,
    summary="List All Forecast Records (Developer Only)",
    description=(
        "Retrieve all forecast runs across all users and legacy runs. "
        "Strictly requires an authenticated developer or administrator role."
    ),
)
def list_developer_records(
    limit: int = Query(default=50, ge=1, le=100, description="Page size limit (1 to 100)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(default=None, description="Filter by status (succeeded, failed, pending)"),
    model: Optional[str] = Query(default=None, description="Filter by model key or display name"),
    current_dev: Dict[str, Any] = Depends(require_developer),
) -> ForecastRecordListResponse:
    """Return all forecast records across the platform, strictly guarded by require_developer."""
    total, records = record_service.list_records(
        limit=limit,
        offset=offset,
        status=status,
        model=model,
        owner_user_id=None,
    )
    return ForecastRecordListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        records=records,
    )



@router.get(
    "/records/{run_id}",
    response_model=ForecastRecordDetail,
    summary="Get Forecast Record Detail by Run ID",
    description=(
        "Retrieve complete details for a persisted forecast run. "
        "Authorized: household users can only access their own runs; developers can inspect any run."
    ),
)
def get_forecast_record(
    run_id: str = Path(..., description="Unique forecast run identifier (e.g. 'fc_...')"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ForecastRecordDetail:
    """Return detailed forecast run record by run ID with ownership verification and anti-enumeration."""
    record = record_service.get_record(run_id=run_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast record with run_id '{run_id}' not found.",
        )

    # Ownership enforcement: non-developers receive 404 for foreign-owned or unassigned records to prevent ID enumeration
    if current_user.get("role") != "developer" and record.owner_user_id != current_user.get("id"):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast record with run_id '{run_id}' not found.",
        )

    return record


@router.get(
    "/records/{run_id}/status",
    response_model=ForecastStatusResponse,
    summary="Get Forecast Run Status",
    description="Retrieve lightweight execution status and metadata for a specific run ID.",
)
def get_forecast_run_status(
    run_id: str = Path(..., description="Unique forecast run identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ForecastStatusResponse:
    """Return run status with ownership authorization and anti-enumeration."""
    record = record_service.get_record(run_id=run_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast run '{run_id}' not found.",
        )

    # Ownership check: non-developers receive 404 for foreign-owned or unassigned records to prevent ID enumeration
    if current_user.get("role") != "developer" and record.owner_user_id != current_user.get("id"):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast run '{run_id}' not found.",
        )

    stat = record_service.get_status(run_id=run_id)
    if stat is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast run '{run_id}' not found.",
        )
    return stat


@router.get(
    "/records/{run_id}/pdf",
    summary="Download Forecast PDF Report",
    description=(
        "Generate and stream a genuine 2-page publication vector PDF report for a successful forecast run. "
        "Enforces ownership: household callers can only download PDFs for their own runs; developers can download for any run. "
        "Returns uniform 404 Not Found for missing or foreign-owned records to prevent ID enumeration."
    ),
)
def download_forecast_pdf(
    run_id: str = Path(..., description="Unique forecast run identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Generate in-memory PDF report from persistent record and return as downloadable attachment."""
    record = record_service.get_record(run_id=run_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast record with run_id '{run_id}' not found.",
        )

    # Ownership check: non-developers receive 404 for foreign-owned or unassigned records to prevent ID enumeration
    if current_user.get("role") != "developer" and record.owner_user_id != current_user.get("id"):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Forecast record with run_id '{run_id}' not found.",
        )

    # Status check: only successful runs have predictions and intervals
    if record.status != "succeeded":
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"PDF reports can only be generated for successful forecasts (current status is '{record.status}').",
        )

    try:
        pdf_bytes = generate_forecast_pdf(record)
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(exc)}",
        )

    filename = f"forecast_report_{run_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# -----------------------------------------------------------------------------
# 5. Monthly Bill Ingestion Endpoints (Phase 2)
# -----------------------------------------------------------------------------
@router.post(
    "/monthly/parse-bill",
    response_model=ExtractedBillData,
    status_code=http_status.HTTP_200_OK,
    summary="Parse Electricity Bill Document (PDF, PNG, JPEG)",
    description=(
        "Ingests an uploaded digital or scanned electricity bill (PDF, PNG, or JPEG). Extracts billing cycle dates, "
        "monthly consumption (kWh), bill amount, consumer/provider metadata, and historical "
        "consumption tables without persisting data or running automated forecasts."
    ),
)
def parse_monthly_bill_pdf(
    file: UploadFile = File(..., description="Electricity bill document (PDF, PNG, or JPEG)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ExtractedBillData:
    """Parse an uploaded electricity bill PDF or image into structured consumption records."""
    filename = file.filename or "uploaded_bill.pdf"
    lower_name = filename.lower()
    ext = f".{lower_name.rsplit('.', 1)[-1]}" if "." in lower_name else ""

    # 1. Filename extension validation
    if ext not in ALLOWED_BILL_EXTENSIONS:
        raise HTTPException(
            status_code=HTTP_422,
            detail=f"Uploaded file '{filename}' has an unsupported file format. Only PDF, PNG, and JPEG files are accepted.",
        )

    # 2. Read binary content safely
    try:
        content = file.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file content: {str(exc)}",
        )

    # 3. Empty file check
    if not content or len(content) == 0:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty (0 bytes).",
        )

    # 4. Upload size validation
    if len(content) > MAX_BILL_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=HTTP_413,
            detail=f"Uploaded file size ({len(content) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of 10 MB.",
        )

    # 5. Magic header verification (do not trust filename/content-type alone)
    if ext == ".pdf":
        if not (content.startswith(b"%PDF-") or b"%PDF-" in content[:1024]):
            raise HTTPException(
                status_code=HTTP_422,
                detail="Uploaded file is not a valid PDF document (missing %PDF- header).",
            )
    elif ext == ".png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(
                status_code=HTTP_422,
                detail="Uploaded file is not a valid PNG image (invalid magic bytes).",
            )
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception:
            raise HTTPException(
                status_code=HTTP_422,
                detail="Uploaded PNG image file is corrupted or malformed.",
            )
    elif ext in (".jpg", ".jpeg"):
        if not content.startswith(b"\xff\xd8\xff"):
            raise HTTPException(
                status_code=HTTP_422,
                detail="Uploaded file is not a valid JPEG image (invalid magic bytes).",
            )
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception:
            raise HTTPException(
                status_code=HTTP_422,
                detail="Uploaded JPEG image file is corrupted or malformed.",
            )

    # 6. Parse document using isolated BillParserService
    try:
        extracted = bill_parser_service.parse_bill_file(file_bytes=content, filename=filename)
    except ValueError as ve:
        raise HTTPException(
            status_code=HTTP_422,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while parsing the bill document.",
        )

    # 7. Check for encrypted / password protected PDFs
    if extracted.raw_metadata and extracted.raw_metadata.get("is_encrypted"):
        err_msg = (
            extracted.warnings[0]
            if extracted.warnings
            else "Encrypted or password-protected PDF detected. Please remove password protection or decrypt the e-bill before uploading."
        )
        raise HTTPException(
            status_code=HTTP_422,
            detail=err_msg,
        )

    # 8. Check for unreadable scanned documents where OCR could not extract records
    if extracted.raw_metadata and extracted.raw_metadata.get("is_scanned_image") and len(extracted.records) == 0:
        specific_warning = next(
            (
                w for w in extracted.warnings
                if any(k in w.lower() for k in ["not available", "initialization error", "could not be initialized", "bindings", "failed"])
            ),
            None,
        )
        err_msg = specific_warning or (
            extracted.warnings[0]
            if extracted.warnings
            else "Image or scanned document could not be read clearly by Optical Character Recognition (OCR). Please verify scan quality or enter readings manually."
        )
        raise HTTPException(
            status_code=HTTP_422,
            detail=err_msg,
        )

    # 9. Check for extraction failure (no recognizable consumption records)
    if len(extracted.records) == 0:
        specific_warning = next(
            (
                w for w in extracted.warnings
                if any(k in w.lower() for k in ["not available", "initialization error", "could not be initialized", "bindings", "failed"])
            ),
            None,
        )
        err_msg = specific_warning or next(
            (
                w for w in extracted.warnings
                if "too low" in w.lower()
                or "resolution" in w.lower()
                or "could not" in w.lower()
                or "unreadable" in w.lower()
                or "not supported" in w.lower()
                or "error" in w.lower()
            ),
            extracted.warnings[0] if extracted.warnings else "Could not reliably identify billing periods or kilowatt-hour consumption values from the document.",
        )
        raise HTTPException(
            status_code=HTTP_422,
            detail=err_msg,
        )

    return extracted


# -----------------------------------------------------------------------------
# 6. Monthly Household Electricity Consumption Forecasting
# -----------------------------------------------------------------------------
@router.post(
    "/monthly/run",
    response_model=MonthlyForecastResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Execute Monthly Household Electricity Consumption Forecast",
    description=(
        "Executes a conservative, history-dependent monthly electricity consumption forecast "
        "from verified household billing records. Supports 1, 3, and 6 month forecast horizons, "
        "chronological backtesting, defensible uncertainty bounds, and decoupled tariff estimation."
    ),
)
def run_monthly_forecast(
    request: MonthlyForecastRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MonthlyForecastResponse:
    """Execute monthly forecasting pipeline on confirmed billing records."""
    try:
        response = monthly_forecast_service.generate_forecast(request=request)
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating the monthly electricity forecast.",
        )


@router.post(
    "/monthly/save",
    response_model=MonthlyForecastRecordDetail,
    status_code=http_status.HTTP_201_CREATED,
    summary="Save Completed Monthly Forecast Run",
    description=(
        "Persists a monthly forecast run for the authenticated user with server-authoritative results. "
        "Recomputes forecast using backend service to guarantee data integrity. "
        "Guarantees that ownership is strictly derived from the authenticated principal. "
        "Protects against duplicate records on repeated submissions."
    ),
)
def save_monthly_forecast(
    payload: SaveMonthlyForecastRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MonthlyForecastRecordDetail:
    """Save a monthly forecast run with server-authoritative generation, owner-level authorization and duplicate protection."""
    owner_id = current_user["id"]
    try:
        # Check if record already exists for this forecast_id to return idempotently or detect conflict
        if payload.forecast_id:
            existing = monthly_record_service.get_monthly_record(run_id=payload.forecast_id)
            if existing:
                if existing.owner_user_id != owner_id:
                    # If foreign-owned, return 404 to prevent ID enumeration
                    raise HTTPException(
                        status_code=http_status.HTTP_404_NOT_FOUND,
                        detail=f"Monthly forecast record with run_id '{payload.forecast_id}' not found.",
                    )
                # Same owner: verify whether validated input matches
                if monthly_record_service.check_input_match(payload.forecast_id, payload.request):
                    return existing
                # Conflicting input: reject with 409 Conflict
                raise HTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail=f"Monthly forecast record with run_id '{payload.forecast_id}' already exists with different input parameters.",
                )

        # Server-authoritative computation: generate forecast strictly from validated input request
        computed_response = monthly_forecast_service.generate_forecast(request=payload.request)
        if payload.forecast_id:
            computed_response.forecast_id = payload.forecast_id

        record = monthly_record_service.save_monthly_forecast(
            owner_user_id=owner_id,
            request=payload.request,
            response=computed_response,
            forecast_id=payload.forecast_id,
        )
        return record
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist monthly forecast record: {str(exc)}",
        )


@router.get(
    "/monthly/records",
    response_model=MonthlyForecastRecordListResponse,
    summary="List Saved Monthly Forecast Runs (Owner Enforced)",
    description=(
        "Retrieve paginated monthly forecast run summaries ordered by creation time descending. "
        "Enforces record-level authorization: household users only see their own runs; "
        "developers can inspect all monthly runs."
    ),
)
def list_monthly_forecast_records(
    limit: int = Query(default=50, ge=1, le=100, description="Page size limit (1 to 100)"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MonthlyForecastRecordListResponse:
    """Return paginated list of saved monthly forecast run summaries with owner-level authorization."""
    if current_user.get("role") == "developer":
        owner_filter = None
    else:
        owner_filter = current_user["id"]

    total, records = monthly_record_service.list_monthly_records(
        owner_user_id=owner_filter,
        limit=limit,
        offset=offset,
    )
    return MonthlyForecastRecordListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        records=records,
    )


@router.get(
    "/monthly/records/{run_id}",
    response_model=MonthlyForecastRecordDetail,
    summary="Get Saved Monthly Forecast Record Detail by Run ID",
    description=(
        "Retrieve complete details for a persisted monthly forecast run. "
        "Authorized: household users can only access their own runs; developers can inspect any run."
    ),
)
def get_monthly_forecast_record(
    run_id: str = Path(..., description="Unique monthly forecast run identifier (e.g. 'mfc_...')"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> MonthlyForecastRecordDetail:
    """Return detailed monthly forecast run record by run ID with ownership verification."""
    record = monthly_record_service.get_monthly_record(run_id=run_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Monthly forecast record with run_id '{run_id}' not found.",
        )

    # Ownership check: non-developers receive 404 for foreign-owned records to prevent ID enumeration
    if current_user.get("role") != "developer" and record.owner_user_id != current_user.get("id"):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Monthly forecast record with run_id '{run_id}' not found.",
        )

    return record


@router.get(
    "/monthly/records/{run_id}/report.pdf",
    summary="Download Monthly Forecast PDF Report",
    description=(
        "Generate and download a genuine publication-grade vector PDF report for a saved monthly forecast snapshot. "
        "Generated strictly from the authoritative database snapshot without re-running or mutating forecasts. "
        "Household users can only download their own runs; developers can download any run. "
        "Returns uniform 404 for missing or unauthorized records to prevent ID enumeration."
    ),
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Monthly electricity forecast vector PDF report document.",
        },
        401: {"description": "Authentication credentials missing or invalid."},
        404: {"description": "Monthly forecast record not found or access unauthorized."},
    },
)
@router.get(
    "/monthly/records/{run_id}/pdf",
    include_in_schema=False,
)
def download_monthly_forecast_pdf(
    run_id: str = Path(..., description="Unique monthly forecast run identifier (e.g. 'mfc_...')"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """Generate and stream publication-grade vector PDF for a saved monthly forecast snapshot."""
    record = monthly_record_service.get_monthly_record(run_id=run_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Monthly forecast record with run_id '{run_id}' not found.",
        )

    # Ownership check: non-developers receive 404 for foreign-owned records to prevent ID enumeration
    if current_user.get("role") != "developer" and record.owner_user_id != current_user.get("id"):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Monthly forecast record with run_id '{run_id}' not found.",
        )

    try:
        pdf_bytes = generate_monthly_forecast_pdf(record)
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate monthly forecast PDF report: {str(exc)}",
        )

    safe_run_id = "".join(c for c in run_id if c.isalnum() or c in ("-", "_"))
    filename = f"monthly_forecast_report_{safe_run_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )




