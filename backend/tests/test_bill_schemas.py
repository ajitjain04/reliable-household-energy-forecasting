"""
Unit tests for monthly electricity bill Pydantic schemas and temporal validations.

Covers:
- Valid records creation and ISO date parsing.
- Invalid dates (start >= end).
- Negative consumption and bill amounts.
- Duplicate billing periods.
- Overlapping billing periods.
- Incomplete history with gap detection (without synthetic data insertion).
- Separation of bill extraction confidence.
- User-corrected bill data verification.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError
from backend.app.schemas.bills import (
    BillExtractionSource,
    ExtractedBillData,
    MonthlyConsumptionRecord,
    UserCorrectedBillData,
    validate_and_sort_billing_records,
)


class TestMonthlyConsumptionRecord(unittest.TestCase):
    """Tests for single MonthlyConsumptionRecord validation."""

    def test_valid_record_with_amount(self):
        """Test valid record with consumption and monetary amount."""
        rec = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=245.5,
            bill_amount=1850.75,
            period_label="Jan 2024",
        )
        self.assertEqual(rec.billing_period_start, date(2024, 1, 1))
        self.assertEqual(rec.billing_period_end, date(2024, 2, 1))
        self.assertEqual(rec.consumption_kwh, 245.5)
        self.assertEqual(rec.bill_amount, 1850.75)
        self.assertEqual(rec.period_label, "Jan 2024")
        self.assertEqual(rec.duration_days, 31)

    def test_valid_record_without_amount(self):
        """Test valid record where monetary amount is optional."""
        rec = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 2, 1),
            billing_period_end=date(2024, 3, 1),
            consumption_kwh=190.0,
        )
        self.assertEqual(rec.consumption_kwh, 190.0)
        self.assertIsNone(rec.bill_amount)
        self.assertEqual(rec.duration_days, 29)

    def test_iso_string_date_parsing(self):
        """Test that ISO date strings are correctly parsed into date objects."""
        rec = MonthlyConsumptionRecord.model_validate({
            "billing_period_start": "2024-03-01",
            "billing_period_end": "2024-04-01",
            "consumption_kwh": 310.2,
        })
        self.assertEqual(rec.billing_period_start, date(2024, 3, 1))
        self.assertEqual(rec.billing_period_end, date(2024, 4, 1))

    def test_invalid_dates_start_equals_end(self):
        """Test rejection when start date equals end date."""
        with self.assertRaises(ValidationError) as ctx:
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 1, 1),
                consumption_kwh=100.0,
            )
        self.assertIn("must be strictly before end date", str(ctx.exception))

    def test_invalid_dates_start_after_end(self):
        """Test rejection when start date is after end date."""
        with self.assertRaises(ValidationError) as ctx:
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 2, 1),
                billing_period_end=date(2024, 1, 1),
                consumption_kwh=100.0,
            )
        self.assertIn("must be strictly before end date", str(ctx.exception))

    def test_negative_consumption_rejected(self):
        """Test rejection of negative kWh consumption."""
        with self.assertRaises(ValidationError) as ctx:
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 2, 1),
                consumption_kwh=-15.0,
            )
        self.assertIn("greater_than_equal", str(ctx.exception))

    def test_negative_bill_amount_rejected(self):
        """Test rejection of negative bill amount."""
        with self.assertRaises(ValidationError) as ctx:
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 2, 1),
                consumption_kwh=150.0,
                bill_amount=-250.0,
            )
        self.assertIn("greater_than_equal", str(ctx.exception))


    def test_zero_consumption_allowed(self):
        """Test that zero consumption (e.g. vacant house) is permitted."""
        rec = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=0.0,
            bill_amount=120.0,  # Fixed minimum charges
        )
        self.assertEqual(rec.consumption_kwh, 0.0)


class TestBillingCollectionValidation(unittest.TestCase):
    """Tests for multi-record temporal consistency and gap handling."""

    def test_duplicate_start_date_rejected(self):
        """Test that records with duplicate start dates raise ValueError."""
        r1 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=200.0,
        )
        r2 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=210.0,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_and_sort_billing_records([r1, r2])
        self.assertIn("Duplicate billing period detected", str(ctx.exception))

    def test_duplicate_end_date_rejected(self):
        """Test that records with duplicate end dates raise ValueError."""
        r1 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=200.0,
        )
        r2 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 15),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=100.0,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_and_sort_billing_records([r1, r2])
        self.assertIn("Duplicate billing period detected", str(ctx.exception))

    def test_overlapping_periods_rejected(self):
        """Test that overlapping billing periods raise ValueError."""
        r1 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 1, 31),
            consumption_kwh=220.0,
        )
        r2 = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 20),
            billing_period_end=date(2024, 2, 20),
            consumption_kwh=230.0,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_and_sort_billing_records([r1, r2])
        self.assertIn("Overlapping billing periods detected", str(ctx.exception))

    def test_contiguous_and_disjoint_periods_accepted(self):
        """Test that strictly contiguous and calendar-separated periods are valid and sorted."""
        r_feb = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 2, 1),
            billing_period_end=date(2024, 3, 1),
            consumption_kwh=180.0,
        )
        r_jan = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 2, 1),
            consumption_kwh=200.0,
        )
        # Passed in reverse order
        sorted_recs, warnings = validate_and_sort_billing_records([r_feb, r_jan])
        self.assertEqual(len(sorted_recs), 2)
        self.assertEqual(sorted_recs[0].billing_period_start, date(2024, 1, 1))
        self.assertEqual(sorted_recs[1].billing_period_start, date(2024, 2, 1))
        self.assertEqual(len(warnings), 0)

    def test_incomplete_history_preserved_without_synthetic_data(self):
        """
        Verify that incomplete history is preserved without inventing synthetic months.
        Gaps must produce informative warnings while keeping records intact.
        """
        r1 = MonthlyConsumptionRecord(
            billing_period_start=date(2023, 1, 1),
            billing_period_end=date(2023, 1, 31),
            consumption_kwh=150.0,
        )
        # 3-month gap: Feb, Mar, Apr missing
        r2 = MonthlyConsumptionRecord(
            billing_period_start=date(2023, 5, 1),
            billing_period_end=date(2023, 5, 31),
            consumption_kwh=280.0,
        )

        sorted_recs, warnings = validate_and_sort_billing_records([r1, r2])

        # Exactly 2 records must remain (no synthetic months inserted)
        self.assertEqual(len(sorted_recs), 2)
        self.assertEqual(sorted_recs[0].consumption_kwh, 150.0)
        self.assertEqual(sorted_recs[1].consumption_kwh, 280.0)

        # Gap must be detected and clearly documented in warnings
        self.assertEqual(len(warnings), 1)
        self.assertIn("Incomplete billing history: gap of 90 days detected", warnings[0])
        self.assertIn("No synthetic observations were inserted", warnings[0])


class TestExtractedBillDataSchema(unittest.TestCase):
    """Tests for ExtractedBillData schema and extraction confidence."""

    def test_valid_extracted_bill_data(self):
        """Test valid extracted bill payload with OCR extraction confidence."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 2, 1),
                consumption_kwh=210.0,
                bill_amount=1600.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 2, 1),
                billing_period_end=date(2024, 3, 1),
                consumption_kwh=195.0,
                bill_amount=1480.0,
            ),
        ]

        extracted = ExtractedBillData(
            records=records,
            extraction_source=BillExtractionSource.PDF,
            extraction_confidence=0.96,
            consumer_number="012345678901",
            billing_unit="4123",
            utility_provider="MSEDCL",
        )

        self.assertEqual(len(extracted.records), 2)
        self.assertEqual(extracted.extraction_source, BillExtractionSource.PDF)
        self.assertEqual(extracted.extraction_confidence, 0.96)
        self.assertEqual(extracted.consumer_number, "012345678901")

    def test_extraction_confidence_bounds(self):
        """Test that extraction_confidence is strictly bounded between 0.0 and 1.0."""
        with self.assertRaises(ValidationError):
            ExtractedBillData(
                records=[],
                extraction_source=BillExtractionSource.IMAGE,
                extraction_confidence=1.2,  # > 1.0
            )

        with self.assertRaises(ValidationError):
            ExtractedBillData(
                records=[],
                extraction_source=BillExtractionSource.IMAGE,
                extraction_confidence=-0.1,  # < 0.0
            )

    def test_extracted_bill_gap_warnings_automatically_appended(self):
        """Test that gap warnings from incomplete history are appended to ExtractedBillData warnings."""
        r1 = MonthlyConsumptionRecord(
            billing_period_start=date(2023, 1, 1),
            billing_period_end=date(2023, 1, 31),
            consumption_kwh=150.0,
        )
        r2 = MonthlyConsumptionRecord(
            billing_period_start=date(2023, 4, 1),
            billing_period_end=date(2023, 4, 30),
            consumption_kwh=250.0,
        )

        extracted = ExtractedBillData(
            records=[r1, r2],
            extraction_source=BillExtractionSource.PDF,
            extraction_confidence=0.92,
            warnings=["Initial OCR warning: blurry footer"],
        )

        self.assertEqual(len(extracted.records), 2)
        self.assertEqual(len(extracted.warnings), 2)
        self.assertEqual(extracted.warnings[0], "Initial OCR warning: blurry footer")
        self.assertIn("Incomplete billing history: gap of 60 days detected", extracted.warnings[1])


class TestUserCorrectedBillDataSchema(unittest.TestCase):
    """Tests for UserCorrectedBillData schema."""

    def test_valid_user_corrected_data(self):
        """Test valid user confirmed billing records."""
        recs = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 2, 1),
                consumption_kwh=225.0,
                bill_amount=1700.0,
            )
        ]

        corrected = UserCorrectedBillData(
            records=recs,
            source=BillExtractionSource.PDF,
            notes="Adjusted January kWh from 22.5 (OCR typo) to 225.0.",
            confirmed_by_user=True,
        )

        self.assertEqual(len(corrected.records), 1)
        self.assertTrue(corrected.confirmed_by_user)
        self.assertIn("OCR typo", corrected.notes)

    def test_empty_records_rejected_for_user_corrected(self):
        """Test that user corrected data cannot be empty (min_length=1)."""
        with self.assertRaises(ValidationError) as ctx:
            UserCorrectedBillData(records=[])
        self.assertIn("too_short", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
