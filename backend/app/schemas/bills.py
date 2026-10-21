"""
Pydantic schemas and validation for monthly household electricity bill data.

Supports:
- Monthly consumption records with strictly validated non-negative kWh and bill amount.
- Extracted bill data from OCR, PDF, or provider interfaces with document extraction confidence.
- User-corrected bill data following manual verification.
- Detection of duplicate and overlapping billing periods.
- Incomplete billing history preservation without inventing synthetic months.

Design principles:
- Extraction confidence is kept strictly separate from forecast confidence.
- No tariff schedules or rates are hardcoded in this data model.
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class BillExtractionSource(str, Enum):
    """Origin of the extracted billing record."""

    PDF = "pdf"
    IMAGE = "image"
    MANUAL = "manual"
    PROVIDER_API = "provider_api"
    MOCK = "mock"


class MonthlyConsumptionRecord(BaseModel):
    """
    A single billing cycle's electricity consumption record.

    Represents verified consumption and optional billed monetary amount
    for a defined billing period.
    """

    model_config = {"extra": "forbid"}

    billing_period_start: date = Field(
        ...,
        description="Start date of the billing period (inclusive).",
    )
    billing_period_end: date = Field(
        ...,
        description="End date of the billing period (exclusive or inclusive depending on provider; must be after start date).",
    )
    consumption_kwh: float = Field(
        ...,
        ge=0.0,
        le=10_000_000.0,
        description="Electricity consumption in kilowatt-hours (kWh). Must be non-negative, finite, and within reasonable residential bounds.",
    )
    bill_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100_000_000.0,
        description="Total billed monetary amount for the cycle. Must be non-negative and finite if provided.",
    )
    period_label: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional human-readable label for the billing cycle (e.g., 'April 2024' or '2024-04').",
    )

    @model_validator(mode="after")
    def validate_period_dates(self) -> "MonthlyConsumptionRecord":
        """Ensure billing period start date is strictly before end date and numeric fields are finite."""
        if self.billing_period_start >= self.billing_period_end:
            raise ValueError(
                f"Billing period start date ({self.billing_period_start}) must be strictly before "
                f"end date ({self.billing_period_end})."
            )
        import math
        if not math.isfinite(self.consumption_kwh):
            raise ValueError("consumption_kwh must be a finite number.")
        if self.bill_amount is not None and not math.isfinite(self.bill_amount):
            raise ValueError("bill_amount must be a finite number.")
        return self

    @property
    def duration_days(self) -> int:
        """Calculate the duration of the billing period in days."""
        return (self.billing_period_end - self.billing_period_start).days


def validate_and_sort_billing_records(
    records: List[MonthlyConsumptionRecord],
) -> Tuple[List[MonthlyConsumptionRecord], List[str]]:
    """
    Validate a collection of monthly billing records for temporal consistency.

    Checks:
    - No duplicate billing start dates or periods.
    - No overlapping billing periods.
    - Preserves incomplete history without inventing missing months.
    - Generates informational warnings when gaps between billing cycles are detected.

    Returns:
        Tuple of (chronologically sorted records, list of gap/informational warnings).

    Raises:
        ValueError: If duplicate or overlapping periods are detected.
    """
    if not records:
        return [], []

    # Sort records chronologically by start date, then end date
    sorted_records = sorted(
        records,
        key=lambda r: (r.billing_period_start, r.billing_period_end),
    )

    warnings: List[str] = []

    for i in range(len(sorted_records) - 1):
        curr = sorted_records[i]
        nxt = sorted_records[i + 1]

        # 1. Check duplicate start dates
        if curr.billing_period_start == nxt.billing_period_start:
            raise ValueError(
                f"Duplicate billing period detected: multiple records starting on {curr.billing_period_start}."
            )

        # 2. Check duplicate end dates
        if curr.billing_period_end == nxt.billing_period_end:
            raise ValueError(
                f"Duplicate billing period detected: multiple records ending on {curr.billing_period_end}."
            )

        # 3. Check overlapping periods
        # If the next period starts before the current period ends, they overlap.
        if nxt.billing_period_start < curr.billing_period_end:
            raise ValueError(
                f"Overlapping billing periods detected: period [{curr.billing_period_start} to "
                f"{curr.billing_period_end}] overlaps with [{nxt.billing_period_start} to "
                f"{nxt.billing_period_end}]."
            )

        # 4. Check for temporal gaps (Incomplete history handling)
        # Standard utility cycles have 0 to 2 days gap between reading dates.
        # If there is a gap > 3 days, document as an informative warning without generating synthetic data.
        gap_days = (nxt.billing_period_start - curr.billing_period_end).days
        if gap_days > 3:
            warnings.append(
                f"Incomplete billing history: gap of {gap_days} days detected between "
                f"{curr.billing_period_end} and {nxt.billing_period_start}. "
                "No synthetic observations were inserted."
            )

    return sorted_records, warnings


class ExtractedBillData(BaseModel):
    """
    Output payload produced by automated bill document extraction (PDF, Image/OCR, or Provider API).

    Note on extraction confidence:
    - 'extraction_confidence' reflects the optical/text parsing accuracy of the document.
    - This is completely distinct from statistical forecast uncertainty or conformal prediction intervals.
    """

    model_config = {"extra": "forbid"}

    records: List[MonthlyConsumptionRecord] = Field(
        default_factory=list,
        description="Extracted monthly billing records.",
    )
    extraction_source: BillExtractionSource = Field(
        ...,
        description="Method used to extract bill data: pdf, image, manual, provider_api, or mock.",
    )
    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Document parsing or OCR quality confidence score (0.0 to 1.0). "
            "Kept strictly separate from statistical forecasting confidence."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Sanitized warnings regarding gaps, low OCR confidence, or extraction ambiguities.",
    )
    consumer_number: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional extracted consumer or account identifier.",
    )
    billing_unit: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional billing unit or division code (e.g. MSEDCL BU).",
    )
    rr_number: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional BESCOM Revenue Register (RR) alphanumeric identifier.",
    )
    utility_provider: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Utility or distribution company name (e.g. 'MSEDCL').",
    )
    raw_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Technical extraction metadata such as file name, page count, or parser version.",
    )

    @model_validator(mode="after")
    def validate_records(self) -> "ExtractedBillData":
        """Sort records, validate temporal consistency, and append any gap warnings."""
        if self.records:
            sorted_records, gap_warnings = validate_and_sort_billing_records(self.records)
            self.records = sorted_records
            for w in gap_warnings:
                if w not in self.warnings:
                    self.warnings.append(w)
        return self


class UserCorrectedBillData(BaseModel):
    """
    Validated billing dataset following user review and manual correction of extracted values.
    """

    model_config = {"extra": "forbid"}

    records: List[MonthlyConsumptionRecord] = Field(
        ...,
        min_length=1,
        max_length=120,
        description="User-verified monthly consumption records. Must contain at least one valid record (up to 120 cycles).",
    )
    source: BillExtractionSource = Field(
        default=BillExtractionSource.MANUAL,
        description="Original source of data before user review and correction.",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional user notes or commentary regarding modifications made.",
    )
    confirmed_by_user: bool = Field(
        default=True,
        description="Explicit confirmation that the user has reviewed and verified these consumption values.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Informational warnings regarding gaps or historical notes.",
    )

    @model_validator(mode="after")
    def validate_records(self) -> "UserCorrectedBillData":
        """Sort records, validate temporal consistency, and append gap warnings."""
        sorted_records, gap_warnings = validate_and_sort_billing_records(self.records)
        self.records = sorted_records
        for w in gap_warnings:
            if w not in self.warnings:
                self.warnings.append(w)
        return self
