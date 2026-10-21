"""
Focused unit tests for BillParserService using synthetic digital bill PDFs.

Covers:
- Synthetic MSEDCL digital bill extraction (provider, consumer no, BU, current cycle, 4-month history).
- Generic utility bill extraction (single period, valid dates and kWh).
- Incomplete history with gap detection without synthetic data insertion.
- Image-only / scanned PDF detection (returns clear OCR limitation message).
- Error handling: empty files, non-PDF files, oversized files.
- Separation of extraction confidence from forecasting uncertainty.
"""

import io
import re
import sys
import unittest
from datetime import date
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pre-initialize PyTorch runtime on Windows before WinRT loads
try:
    import torch  # noqa: F401
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from backend.app.schemas.bills import BillExtractionSource, ExtractedBillData, MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import MonthlyForecastRequest
from backend.app.services.bill_parser_service import BillParserService, bill_parser_service
from backend.app.services.monthly_forecast_service import monthly_forecast_service


def create_synthetic_image_file(text_content: str, img_format: str = "png") -> bytes:
    """Helper to generate a genuine in-memory raster image (PNG or JPEG) with text rendered via matplotlib."""
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
    ax.axis("off")
    ax.text(0.05, 0.95, text_content, fontsize=11, verticalalignment="top", fontfamily="sans-serif")
    buf = io.BytesIO()
    fig.savefig(buf, format=img_format, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def create_synthetic_camera_jpeg(text_content: str, orientation: int = 6) -> bytes:
    """Helper to generate a JPEG bill image with EXIF orientation metadata mimicking smartphone camera uploads."""
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
    ax.axis("off")
    ax.text(0.05, 0.95, text_content, fontsize=11, verticalalignment="top", fontfamily="sans-serif")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    base = Image.open(buf).convert("RGB")
    if orientation == 6:
        sensor_img = base.transpose(Image.ROTATE_90)
    elif orientation == 8:
        sensor_img = base.transpose(Image.ROTATE_270)
    elif orientation == 3:
        sensor_img = base.transpose(Image.ROTATE_180)
    else:
        sensor_img = base

    exif = sensor_img.getexif()
    exif[0x0112] = orientation
    jpeg_buf = io.BytesIO()
    sensor_img.save(jpeg_buf, format="JPEG", exif=exif)
    return jpeg_buf.getvalue()


def create_synthetic_pdf(text_content: str) -> bytes:
    """Helper to generate a genuine, in-memory digital PDF with text streams using matplotlib."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.95, text_content, fontsize=9, verticalalignment="top", fontfamily="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def create_synthetic_scanned_pdf(text_content: str) -> bytes:
    """Helper to generate a scanned/image-only PDF (no digital text streams) by embedding a raster PNG."""
    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
    ax.axis("off")
    ax.text(0.05, 0.95, text_content, fontsize=11, verticalalignment="top", fontfamily="sans-serif")
    png_buf = io.BytesIO()
    plt.savefig(png_buf, format="png", bbox_inches="tight")
    plt.close(fig)

    png_buf.seek(0)
    img = Image.open(png_buf).convert("RGB")
    pdf_buf = io.BytesIO()
    img.save(pdf_buf, format="PDF")
    return pdf_buf.getvalue()


def create_synthetic_two_page_scanned_pdf(page1_text: str, page2_text: str, skew: float = 0.0) -> bytes:
    """Helper to generate a realistic two-page scanned PDF with optional rotation/skew."""
    def _render(txt: str, rot: float = 0.0) -> Image.Image:
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax.axis("off")
        ax.text(0.08, 0.90, txt, fontsize=9, verticalalignment="top", fontfamily="monospace")
        png_buf = io.BytesIO()
        plt.savefig(png_buf, format="png", bbox_inches="tight")
        plt.close(fig)
        png_buf.seek(0)
        img = Image.open(png_buf).convert("RGB")
        if rot != 0.0:
            img = img.rotate(rot, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
        return img

    img1 = _render(page1_text, rot=skew)
    img2 = _render(page2_text, rot=-skew if skew else 0.0)

    pdf_buf = io.BytesIO()
    img1.save(pdf_buf, format="PDF", save_all=True, append_images=[img2])
    return pdf_buf.getvalue()


class TestBillParserService(unittest.TestCase):
    """Test suite for BillParserService."""

    def setUp(self):
        self.parser = BillParserService()

    def test_synthetic_msedcl_bill_extraction(self):
        """Test extraction from a synthetic MSEDCL e-bill containing current cycle and historical table."""
        sample_bill_text = (
            "MAHAVITARAN - MSEDCL ELECTRICITY BILL\n"
            "CONSUMER NO: 012345678901   BILLING UNIT: 4123\n"
            "BILLING PERIOD: 01/01/2024 to 31/01/2024\n"
            "CURRENT CONSUMPTION: 250.00 kWh\n"
            "NET BILL AMOUNT: Rs. 2676.70\n"
            "\n"
            "PREVIOUS CONSUMPTION HISTORY:\n"
            "Jan-2024 250.00 2676.70\n"
            "Dec-2023 210.00 2150.00\n"
            "Nov-2023 195.00 1980.00\n"
            "Oct-2023 230.00 2400.00\n"
        )

        pdf_bytes = create_synthetic_pdf(sample_bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="msedcl_sample.pdf")

        self.assertIsInstance(extracted, ExtractedBillData)
        self.assertEqual(extracted.extraction_source, BillExtractionSource.PDF)
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(extracted.billing_unit, "4123")

        # Must have extracted 4 monthly records
        self.assertEqual(len(extracted.records), 4)

        # Verified chronological sorting
        self.assertEqual(extracted.records[0].billing_period_start, date(2023, 10, 1))
        self.assertEqual(extracted.records[1].billing_period_start, date(2023, 11, 1))
        self.assertEqual(extracted.records[2].billing_period_start, date(2023, 12, 1))
        self.assertEqual(extracted.records[3].billing_period_start, date(2024, 1, 1))

        # Check values
        self.assertEqual(extracted.records[3].consumption_kwh, 250.0)
        self.assertEqual(extracted.records[3].bill_amount, 2676.70)

        # Extraction confidence should be high and bounded
        self.assertIsNotNone(extracted.extraction_confidence)
        self.assertGreaterEqual(extracted.extraction_confidence, 0.85)
        self.assertLessEqual(extracted.extraction_confidence, 1.0)

    def test_generic_utility_bill_extraction(self):
        """Test extraction from a generic non-MSEDCL utility bill with single cycle."""
        generic_text = (
            "STATE POWER CORPORATION LIMITED\n"
            "ACCOUNT NO: 9876543210\n"
            "BILLING PERIOD: 01/03/2024 to 31/03/2024\n"
            "TOTAL UNITS BILLED: 310.5 kWh\n"
            "AMOUNT PAYABLE: Rs 2950.00\n"
        )

        pdf_bytes = create_synthetic_pdf(generic_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="generic_bill.pdf")

        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]
        self.assertEqual(rec.billing_period_start, date(2024, 3, 1))
        self.assertEqual(rec.billing_period_end, date(2024, 3, 31))
        self.assertEqual(rec.consumption_kwh, 310.5)
        self.assertEqual(rec.bill_amount, 2950.00)
        self.assertGreaterEqual(extracted.extraction_confidence, 0.70)

    def test_incomplete_history_gap_handling(self):
        """Test that missing months in historical table produce gap warnings without synthetic insertion."""
        gap_bill_text = (
            "ELECTRICITY UTILITY\n"
            "CONSUMER NO: 555544443333\n"
            "CONSUMPTION HISTORY:\n"
            "Jan-2024 240.0\n"
            "Oct-2023 210.0\n"  # Nov and Dec missing
        )

        pdf_bytes = create_synthetic_pdf(gap_bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes)

        # Exactly 2 records must remain
        self.assertEqual(len(extracted.records), 2)
        self.assertEqual(extracted.records[0].billing_period_start, date(2023, 10, 1))
        self.assertEqual(extracted.records[1].billing_period_start, date(2024, 1, 1))

        # Gap warning must be captured
        self.assertTrue(
            any("gap" in w.lower() or "missing" in w.lower() for w in extracted.warnings),
            "Gap between Oct 2023 and Jan 2024 must emit an informational warning",
        )

    def test_image_only_blank_scan_pdf_fails_safely(self):
        """Test that unreadable blank scanned PDFs return empty records and clear OCR notice."""
        # Create a blank PDF with no text stream
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        blank_pdf_bytes = buf.getvalue()

        extracted = self.parser.parse_bill_pdf(blank_pdf_bytes, filename="scanned_bill.pdf")
        self.assertEqual(len(extracted.records), 0)
        self.assertEqual(extracted.extraction_confidence, 0.0)
        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertTrue(
            any("ocr" in w.lower() for w in extracted.warnings),
            "Must return clear message indicating that OCR could not read document clearly",
        )
        self.assertTrue(extracted.raw_metadata.get("is_scanned_image"))

    def test_synthetic_scanned_msedcl_bill_ocr_extraction(self):
        """Test extraction from an image-only scanned MSEDCL bill via Windows Native OCR."""
        sample_bill_text = (
            "MAHAVITARAN - MSEDCL ELECTRICITY BILL\n"
            "CONSUMER NO: 012345678901   BILLING UNIT: 4123\n"
            "BILLING PERIOD: 01/01/2024 to 31/01/2024\n"
            "CURRENT CONSUMPTION: 250.00 kWh\n"
            "NET BILL AMOUNT: Rs. 2676.70\n"
            "\n"
            "PREVIOUS CONSUMPTION HISTORY:\n"
            "Jan-2024 250.00 2676.70\n"
            "Dec-2023 210.00 2150.00\n"
            "Nov-2023 195.00 1980.00\n"
            "Oct-2023 230.00 2400.00\n"
        )

        scanned_pdf_bytes = create_synthetic_scanned_pdf(sample_bill_text)

        # Verify that raw digital text stream has no extractable text
        digital_text = self.parser.extract_raw_text_from_pdf(scanned_pdf_bytes)
        self.assertLess(len(digital_text.strip()), 15, "Scanned PDF fixture must not contain digital text streams")

        # Parse through full service with OCR fallback
        extracted = self.parser.parse_bill_pdf(scanned_pdf_bytes, filename="scanned_msedcl.pdf")

        self.assertIsInstance(extracted, ExtractedBillData)
        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(extracted.billing_unit, "4123")

        # Must have extracted all 4 records via OCR
        self.assertEqual(len(extracted.records), 4)
        self.assertEqual(extracted.records[0].billing_period_start, date(2023, 10, 1))
        self.assertEqual(extracted.records[3].billing_period_start, date(2024, 1, 1))
        self.assertEqual(extracted.records[3].consumption_kwh, 250.0)
        self.assertEqual(extracted.records[3].bill_amount, 2676.70)

        # Confidence should be penalised for OCR uncertainty (< 0.90) but still solid
        self.assertIsNotNone(extracted.extraction_confidence)
        self.assertGreaterEqual(extracted.extraction_confidence, 0.70)
        self.assertLessEqual(extracted.extraction_confidence, 0.90)

        # Clear warning regarding OCR source
        self.assertTrue(
            any("ocr fallback" in w.lower() for w in extracted.warnings),
            "Must flag OCR derivation in warnings",
        )
        self.assertTrue(extracted.raw_metadata.get("ocr_applied"))
        self.assertEqual(extracted.raw_metadata.get("ocr_engine"), "Windows.Media.Ocr")

    def test_synthetic_scanned_generic_bill_ocr_extraction(self):
        """Test extraction from an image-only scanned generic utility bill via Windows Native OCR."""
        generic_text = (
            "STATE POWER CORPORATION LIMITED\n"
            "ACCOUNT NO: 9876543210\n"
            "BILLING PERIOD: 01/03/2024 to 31/03/2024\n"
            "TOTAL UNITS BILLED: 310.5 kWh\n"
            "AMOUNT PAYABLE: Rs 2950.00\n"
        )
        scanned_pdf = create_synthetic_scanned_pdf(generic_text)
        extracted = self.parser.parse_bill_pdf(scanned_pdf, filename="scanned_generic.pdf")

        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].billing_period_start, date(2024, 3, 1))
        self.assertEqual(extracted.records[0].billing_period_end, date(2024, 3, 31))
        self.assertEqual(extracted.records[0].consumption_kwh, 310.5)
        self.assertEqual(extracted.records[0].bill_amount, 2950.00)
        self.assertTrue(extracted.raw_metadata.get("ocr_applied"))

    def test_ocr_non_utility_scan_fails_safely(self):
        """Test that scanned documents containing non-utility text return zero records and clear warning."""
        prose_text = (
            "History of Modern Architecture\n"
            "This chapter examines structural steel framing, cantilevered balconies,\n"
            "and curtain walls in mid-twentieth century residential construction.\n"
        )
        scanned_pdf = create_synthetic_scanned_pdf(prose_text)
        extracted = self.parser.parse_bill_pdf(scanned_pdf, filename="scanned_architecture.pdf")

        self.assertEqual(len(extracted.records), 0)
        self.assertEqual(extracted.extraction_confidence, 0.0)
        self.assertTrue(
            any("could not reliably identify" in w.lower() for w in extracted.warnings),
            "Must flag failure to identify utility consumption records from OCR text",
        )

    def test_empty_file_rejected(self):
        """Test rejection of 0-byte file."""
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_pdf(b"", filename="empty.pdf")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_invalid_non_pdf_rejected(self):
        """Test rejection of non-PDF bytes."""
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_pdf(b"This is a text file, not a pdf.", filename="fake.pdf")
        self.assertIn("missing %pdf-", str(ctx.exception).lower())

    def test_oversized_file_rejected(self):
        """Test rejection of oversized PDF (> 10 MB)."""
        oversized_bytes = b"%PDF-1.4\n" + (b"0" * (11 * 1024 * 1024))
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_pdf(oversized_bytes, filename="oversized.pdf")
        self.assertIn("exceeds maximum allowed limit", str(ctx.exception).lower())

    def test_malformed_corrupted_pdf_fails_safely(self):
        """Test that malformed PDF with valid %PDF- header but corrupted objects fails safely with warnings."""
        corrupted_bytes = b"%PDF-1.4\nCorrupted stream data with no valid PDF xref, catalog, or objects.\n%%EOF"
        extracted = self.parser.parse_bill_pdf(corrupted_bytes, filename="corrupted.pdf")
        self.assertEqual(len(extracted.records), 0)
        self.assertEqual(extracted.extraction_confidence, 0.0)
        self.assertTrue(
            any("failed to open" in w.lower() or "format error" in w.lower() or "ocr" in w.lower() for w in extracted.warnings),
            "Must capture PDF format failure warning safely",
        )


    def test_encrypted_pdf_detection(self):
        """Test that encrypted / password-protected PDFs return empty records and clear encryption notice."""
        encrypted_pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"3 0 obj\n<< /Filter /Standard /V 2 /R 3 >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R /Encrypt 3 0 R >>\n%%EOF"
        )
        extracted = self.parser.parse_bill_pdf(encrypted_pdf_bytes, filename="encrypted_bill.pdf")
        self.assertEqual(len(extracted.records), 0)
        self.assertEqual(extracted.extraction_confidence, 0.0)
        self.assertTrue(
            any("encrypted" in w.lower() or "password" in w.lower() for w in extracted.warnings),
            "Must return clear message indicating that encrypted/password-protected PDFs cannot be parsed",
        )

    def test_disambiguation_full_dates_not_treated_as_month_table(self):
        """Test that full dates like 15/01/2024 with numbers are not confused with monthly table rows."""
        bill_text = (
            "UTILITY BILL\n"
            "CONSUMER NO: 123456789012\n"
            "BILL DATE: 15/01/2024 250.00\n"  # 15/01/2024 followed by number (e.g. deposit or charge)
            "CONSUMPTION HISTORY:\n"
            "Feb-2024 180.00 1900.00\n"
        )
        pdf_bytes = create_synthetic_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes)

        # Must only extract the legitimate Feb-2024 historical record
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].billing_period_start, date(2024, 2, 1))
        self.assertEqual(extracted.records[0].consumption_kwh, 180.0)

    def test_historical_table_with_meter_readings_extracts_units_not_meter_index(self):
        """Test that table rows with [PrevReading, CurrReading, Units, Amount] extract units, not meter readings."""
        table_text = (
            "ELECTRICITY DISTRIBUTION CO\n"
            "CONSUMER NO: 998877665544\n"
            "CONSUMPTION HISTORY:\n"
            "Jan-2024 12500 12750 250.00 2650.00\n"  # 12750 - 12500 = 250.00 units
        )
        pdf_bytes = create_synthetic_pdf(table_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes)

        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]
        self.assertEqual(rec.consumption_kwh, 250.00)
        self.assertEqual(rec.bill_amount, 2650.00)

    def test_non_utility_pdf_fails_safely(self):
        """Test that text PDFs without utility fields fail safely with zero records and clear warning."""
        essay_text = (
            "Introduction to Machine Learning\n"
            "This document discusses gradient descent, deep neural networks, and loss functions.\n"
            "No utility billing information or consumption metrics are present in this text."
        )
        pdf_bytes = create_synthetic_pdf(essay_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="lecture_notes.pdf")

        self.assertEqual(len(extracted.records), 0)
        self.assertEqual(extracted.extraction_confidence, 0.0)
        self.assertTrue(
            any("could not reliably identify" in w.lower() for w in extracted.warnings),
            "Must produce clear warning when document is not an electricity bill",
        )
    def test_synthetic_scanned_msedcl_2digit_year_and_history(self):
        """Test extraction from scanned MSEDCL bill with 2-digit years in history and reading dates."""
        sample_bill_text = (
            "MAHAVITARAN - MSEDCL ELECTRICITY BILL\n"
            "CONSUMER NO: 012345678901   BILLING UNIT: 4123\n"
            "BILL MONTH: APR-24\n"
            "PREV READING DATE: 05/03/24\n"
            "READING DATE: 05/04/24\n"
            "UNITS BILLED: 250.00 kWh\n"
            "NET BILL AMOUNT: Rs. 2676.70\n"
            "\n"
            "PREVIOUS CONSUMPTION HISTORY:\n"
            "APR-24 250.00 2676.70\n"
            "MAR-24 210.00 2150.00\n"
            "FEB-24 195.00 1980.00\n"
            "JAN-24 230.00 2400.00\n"
        )
        scanned_pdf_bytes = create_synthetic_scanned_pdf(sample_bill_text)
        extracted = self.parser.parse_bill_pdf(scanned_pdf_bytes, filename="scanned_msedcl_apr24.pdf")

        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(len(extracted.records), 4)
        self.assertEqual(extracted.records[0].billing_period_start, date(2024, 1, 1))
        self.assertEqual(extracted.records[3].billing_period_start, date(2024, 4, 1))
        self.assertEqual(extracted.records[3].consumption_kwh, 250.0)
        self.assertEqual(extracted.records[3].bill_amount, 2676.70)
        self.assertLessEqual(extracted.extraction_confidence, 0.90)

    def test_synthetic_scanned_msedcl_single_month_april_bill(self):
        """Test extraction from scanned MSEDCL bill containing single month cycle without history table."""
        single_bill_text = (
            "MAHAVITARAN - MSEDCL ELECTRICITY BILL\n"
            "CONSUMER NO: 012345678901   BILLING UNIT: 4123\n"
            "BILL MONTH: APR-24\n"
            "PREV READING DATE: 01/03/24\n"
            "READING DATE: 01/04/24\n"
            "UNITS BILLED: 250.00\n"
            "NET AMOUNT: Rs. 2676.70\n"
        )
        scanned_pdf_bytes = create_synthetic_scanned_pdf(single_bill_text)
        extracted = self.parser.parse_bill_pdf(scanned_pdf_bytes, filename="Electricity Bill April.pdf")

        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].billing_period_start, date(2024, 3, 1))
        self.assertEqual(extracted.records[0].billing_period_end, date(2024, 4, 1))
        self.assertEqual(extracted.records[0].consumption_kwh, 250.0)
        self.assertEqual(extracted.records[0].bill_amount, 2676.70)

    def test_unreadable_scan_returns_actionable_error_at_index_0(self):
        """Test that unreadable scanned documents place the actionable error warning at index 0, not info notes."""
        prose_text = (
            "Abstract algebra studies algebraic structures such as groups, rings, and fields.\n"
            "This scanned document has no meter readings or electricity utility data."
        )
        scanned_pdf = create_synthetic_scanned_pdf(prose_text)
        extracted = self.parser.parse_bill_pdf(scanned_pdf, filename="algebra_notes.pdf")

        self.assertEqual(len(extracted.records), 0)
        self.assertTrue(len(extracted.warnings) > 0)
        # First warning MUST be the actionable rejection cause, NOT an OCR-success notice
        first_warning = extracted.warnings[0].lower()
        self.assertFalse(first_warning.startswith("text extracted using"))
        self.assertTrue(
            "could not reliably identify" in first_warning
            or "unreadable" in first_warning
            or "no billing records" in first_warning,
            f"Expected actionable error at index 0, got: {extracted.warnings[0]}",
        )
    def test_synthetic_scanned_msedcl_two_page_bill_with_history_table(self):
        """Test extraction from a realistic 2-page scanned MSEDCL bill with reading table on page 1 and history on page 2."""
        page1_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "TAX INVOICE / ELECTRICITY BILL\n"
            "Consumer No: 012345678901   BU: 4123   Bill Month: Apr 2025\n"
            "Tariff: LT-I (B) Residential   Sanctioned Load: 5.00 kW\n"
            "\n"
            "Meter No      Prev Date     Curr Date     Prev Read   Curr Read   Units Billed\n"
            "88776655      05/03/2025    05/04/2025    12500       12750       250\n"
            "\n"
            "Bill Date: 08/04/2025        Due Date: 28/04/2025\n"
            "Energy Charges: Rs. 1782.00\n"
            "Fixed Charges: Rs. 128.00\n"
            "Wheeling Charges: Rs. 292.50\n"
            "Net Amount Payable: Rs. 2630.00\n"
        )
        page2_text = (
            "CONSUMPTION AND PAYMENT HISTORY FOR LAST 6 MONTHS\n"
            "\n"
            "Month      Units Billed   Bill Amount (Rs.)   Payment Date   Payment (Rs.)\n"
            "Apr 2025       250           2630.00            Pending           0.00\n"
            "Mar 2025       210           2150.00           22/03/2025       2150.00\n"
            "Feb 2025       195           1980.00           20/02/2025       1980.00\n"
            "Jan 2025       230           2400.00           18/01/2025       2400.00\n"
            "Dec 2024       180           1820.00           21/12/2024       1820.00\n"
            "Nov 2024       220           2290.00           19/11/2024       2290.00\n"
            "\n"
            "Thank you for prompt payment. For complaints visit mahadiscom.in\n"
        )
        pdf_bytes = create_synthetic_two_page_scanned_pdf(page1_text, page2_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="msedcl_2page_scan.pdf")

        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(extracted.billing_unit, "4123")

        # Ground truth: Exactly 6 monthly records, sorted chronologically from Nov 2024 to Apr 2025
        self.assertEqual(len(extracted.records), 6, f"Expected 6 records, got {len(extracted.records)}")

        expected_records = [
            (date(2024, 11, 1), date(2024, 11, 30), 220.0, 2290.0),
            (date(2024, 12, 1), date(2024, 12, 31), 180.0, 1820.0),
            (date(2025, 1, 1), date(2025, 1, 31), 230.0, 2400.0),
            (date(2025, 2, 1), date(2025, 2, 28), 195.0, 1980.0),
            (date(2025, 3, 1), date(2025, 3, 31), 210.0, 2150.0),
            (date(2025, 4, 1), date(2025, 4, 30), 250.0, 2630.0),
        ]

        for i, (exp_start, exp_end, exp_kwh, exp_amt) in enumerate(expected_records):
            rec = extracted.records[i]
            self.assertEqual(rec.billing_period_start, exp_start, f"Record {i} start date mismatch")
            self.assertEqual(rec.billing_period_end, exp_end, f"Record {i} end date mismatch")
            self.assertEqual(rec.consumption_kwh, exp_kwh, f"Record {i} kWh mismatch")
            self.assertEqual(rec.bill_amount, exp_amt, f"Record {i} amount mismatch")

        # Verify payment amounts (0.00, 2150.00) were NOT misclassified as consumption
        for rec in extracted.records:
            self.assertNotEqual(rec.consumption_kwh, 0.0)

        # Verify meter readings (12500, 12750) were NOT misclassified as consumption
        for rec in extracted.records:
            self.assertLess(rec.consumption_kwh, 1000.0)

        # Verify diagnostic metadata recorded safely
        self.assertEqual(extracted.raw_metadata.get("pages_processed"), 2)
        self.assertTrue(extracted.raw_metadata.get("ocr_runtime_available"))
        self.assertEqual(extracted.raw_metadata.get("records_after_validation"), 6)
        self.assertEqual(len(extracted.raw_metadata.get("page_dimensions", [])), 2)

    def test_synthetic_scanned_msedcl_mild_skew_and_low_res(self):
        """Test that deskewed spatial clustering correctly parses mildly rotated scans (1.2 deg skew)."""
        page1_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "TAX INVOICE / ELECTRICITY BILL\n"
            "Consumer No: 012345678901   BU: 4123   Bill Month: Apr 2025\n"
            "Meter No      Prev Date     Curr Date     Prev Read   Curr Read   Units Billed\n"
            "88776655      05/03/2025    05/04/2025    12500       12750       250\n"
            "Bill Date: 08/04/2025        Due Date: 28/04/2025\n"
            "Net Amount Payable: Rs. 2630.00\n"
        )
        page2_text = (
            "CONSUMPTION AND PAYMENT HISTORY FOR LAST 6 MONTHS\n"
            "Month      Units Billed   Bill Amount (Rs.)   Payment Date   Payment (Rs.)\n"
            "Apr 2025       250           2630.00            Pending           0.00\n"
            "Mar 2025       210           2150.00           22/03/2025       2150.00\n"
            "Feb 2025       195           1980.00           20/02/2025       1980.00\n"
            "Jan 2025       230           2400.00           18/01/2025       2400.00\n"
            "Dec 2024       180           1820.00           21/12/2024       1820.00\n"
            "Nov 2024       220           2290.00           19/11/2024       2290.00\n"
        )
        pdf_bytes = create_synthetic_two_page_scanned_pdf(page1_text, page2_text, skew=1.2)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="skewed_scan.pdf")

        self.assertEqual(len(extracted.records), 6)
        self.assertEqual(extracted.records[0].billing_period_start, date(2024, 11, 1))
        self.assertEqual(extracted.records[5].billing_period_start, date(2025, 4, 1))
        self.assertEqual(extracted.records[5].consumption_kwh, 250.0)

    def test_synthetic_scanned_msedcl_bill_of_supply_with_chart_and_load(self):
        """
        Regression test: Scanned MSEDCL bill with 'Bill of Supply for APR-2025',
        'w.e.f. 01/04/2020', 'Sanctioned Load : 9 KW', 'Due Date: 25/04/2025',
        'Net Bill Amount Rs. 1,760.00', and floating unassociated chart labels.

        Verifies:
        - Billing cycle resolved to April 2025 (NOT 2020).
        - Consumption resolved to 167.0 kWh (NOT 9 kW load, NOT chart labels).
        - Bill amount resolved to Rs. 1,760.00 (NOT Rs. 25 due date).
        - Floating chart numbers are NOT converted into fake historical records.
        - Confidence is calibrated honestly (~0.62).
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "TAX INVOICE / BILL OF SUPPLY FOR APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4123\n"
            "Tariff: LT-I (B) Residential   w.e.f. 01/04/2020\n"
            "Sanctioned Load : 9.00 KW   Phase: 1   Pole No: 42\n"
            "Meter No: 87654321   Make: SECURE   Status: NORMAL\n"
            "Prev Date: 05/03/2025   Current Date: 05/04/2025\n"
            "Previous Reading: 12500   Current Reading: 12667   MF: 1.0   Units: 167\n"
            "Bill Details:\n"
            "Bill Date: 09/04/2025   Due Date: 25/04/2025\n"
            "Net Bill Amount Rs. 1,760.00\n"
            "Amount Payable by 20/04/2025 Rs. 1,740.00\n"
            "Total Amount Payable up to 25/04/2025 Rs. 1,760.00\n"
            "Total Amount Payable after 25/04/2025 Rs. 1,780.00\n"
            "Consumption History Chart:\n"
            "160  156  109  84  116  148  139  176  67  2  0\n"
            "May-24 Jun-24 Jul-24 Aug-24 Sep-24 Oct-24 Nov-24 Dec-24 Jan-25 Feb-25 Mar-25\n"
        )
        pdf_bytes = create_synthetic_scanned_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="Electricity Bill April.pdf")

        # Provider and identifiers
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(extracted.billing_unit, "4123")

        # Exactly 1 verified record (chart labels NOT converted to fake records)
        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]

        # Verify correct cycle (2025, NOT 2020)
        self.assertEqual(rec.billing_period_start.year, 2025)
        self.assertEqual(rec.billing_period_end.year, 2025)
        self.assertIn(rec.billing_period_end.month, (4, 5))

        # Verify correct consumption (167 units, NOT 9 kW load, NOT 160 chart value)
        self.assertEqual(rec.consumption_kwh, 167.0)

        # Verify correct amount (Rs. 1,760, NOT Rs. 25 due date)
        self.assertEqual(rec.bill_amount, 1760.0)

        # Verify calibrated confidence
        self.assertGreaterEqual(extracted.extraction_confidence, 0.55)
        self.assertLessEqual(extracted.extraction_confidence, 0.70)

    def test_regression_spaced_header_tokens_and_delimiters(self):
        """
        Regression test: Verify month/year header parsing tolerates OCR whitespace around
        separators and line breaks, such as 'APR - 2025', 'APR / 2025', and 'APR-2025'.
        """
        header_variants = [
            ("Bill of Supply for APR - 2025\nConsumer No: 012345678901\nUnits: 167\nBill Amount Rs. 1760.00", 2025, 4),
            ("Bill of Supply for APR / 2025\nConsumer No: 012345678901\nUnits: 167\nBill Amount Rs. 1760.00", 2025, 4),
            ("TAX INVOICE / BILL OF SUPPLY\nFOR APR - 2025\nConsumer No: 012345678901\nUnits: 167\nBill Amount Rs. 1760.00", 2025, 4),
            ("Billing Month : APR - 2025\nConsumer No: 012345678901\nUnits: 167\nBill Amount Rs. 1760.00", 2025, 4),
        ]
        for text, exp_year, exp_month in header_variants:
            pdf_bytes = create_synthetic_pdf(text)
            extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="spaced_header.pdf")
            self.assertEqual(len(extracted.records), 1, f"Expected 1 record for header variant: {text[:30]}")
            rec = extracted.records[0]
            self.assertEqual(rec.billing_period_start.year, exp_year)
            self.assertEqual(rec.billing_period_start.month, exp_month)
            self.assertEqual(rec.consumption_kwh, 167.0)
            self.assertEqual(rec.bill_amount, 1760.0)

        # Also verify Devanagari header directly in parser (bypassing matplotlib PDF glyph absence)
        dev_text = "देयक महिना : APR - 2025\nUnits: 167\nBill Amount Rs. 1760.00"
        dev_rec, _ = self.parser._extract_current_period_record(dev_text)
        self.assertIsNotNone(dev_rec)
        self.assertEqual(dev_rec.billing_period_start.year, 2025)
        self.assertEqual(dev_rec.billing_period_start.month, 4)
        self.assertEqual(dev_rec.consumption_kwh, 167.0)
        self.assertEqual(dev_rec.bill_amount, 1760.0)

    def test_regression_adjacent_reading_labels_and_values(self):
        """
        Regression test: Verify meter reading extraction when table labels (Previous Reading,
        Current Reading, MF, Units) and numeric values (12500, 12667, 1.0, 167) are on adjacent OCR lines.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "BILL OF SUPPLY FOR APR - 2025\n"
            "Consumer No: 012345678901   Billing Unit: 4123\n"
            "Sanctioned Load : 9.00 KW\n"
            "Prev Date: 05/03/2025   Current Date: 05/04/2025\n"
            "Previous Reading   Current Reading   Multiplier   Units\n"
            "12500   12667   1.0   167\n"
            "Net Bill Amount Rs. 1,760.00\n"
        )
        pdf_bytes = create_synthetic_scanned_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="adjacent_readings.pdf")

        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]
        # Verified reading delta arithmetic: (12667 - 12500) * 1.0 = 167 units
        self.assertEqual(rec.consumption_kwh, 167.0)
        self.assertEqual(rec.billing_period_start, date(2025, 3, 5))
        self.assertEqual(rec.billing_period_end, date(2025, 4, 5))
        self.assertEqual(rec.bill_amount, 1760.0)

    def test_regression_due_date_and_reliable_currency_context(self):
        """
        Regression test: Verify that a due date's day number (e.g. 25 from '25/04/2025') is never
        extracted as a bill amount, and that reliable currency context (Rs. 1,760.00) is required.
        """
        # Case A: Amount payable with intermediate due date clause
        text_with_due_date = (
            "BILL OF SUPPLY FOR APR - 2025\n"
            "Consumer No: 012345678901\n"
            "Units: 167\n"
            "Due Date: 25/04/2025\n"
            "Net Bill Amount Payable by Due Date: 25/04/2025 Rs. 1,760.00\n"
        )
        pdf_bytes = create_synthetic_pdf(text_with_due_date)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="due_date_amount.pdf")
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].bill_amount, 1760.0)
        self.assertNotEqual(extracted.records[0].bill_amount, 25.0)

        # Case B: Due date alone without currency amount must not extract 25
        text_without_currency = (
            "BILL OF SUPPLY FOR APR - 2025\n"
            "Consumer No: 012345678901\n"
            "Units: 167\n"
            "Net Bill Amount Payable by Due Date: 25/04/2025\n"
        )
        pdf_bytes_b = create_synthetic_pdf(text_without_currency)
        extracted_b = self.parser.parse_bill_pdf(pdf_bytes_b, filename="no_currency.pdf")
        self.assertEqual(len(extracted_b.records), 1)
        # Bill amount must be None (not 25.0)
        self.assertIsNone(extracted_b.records[0].bill_amount)

    def test_regression_legitimate_historical_bill_support(self):
        """
        Regression test: Verify that legitimate historical bills (e.g. July 2018) parse correctly
        and are not rejected by artificial minimum-year filters.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "TAX INVOICE / BILL OF SUPPLY FOR JUL-2018\n"
            "Consumer No: 012345678901   Billing Unit: 4123\n"
            "Prev Date: 05/06/2018   Current Date: 05/07/2018\n"
            "Previous Reading: 8200   Current Reading: 8410   Units: 210\n"
            "Net Amount Payable: Rs. 1,420.00\n"
        )
        pdf_bytes = create_synthetic_scanned_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="historical_2018_bill.pdf")

        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]
        self.assertEqual(rec.billing_period_start, date(2018, 6, 5))
        self.assertEqual(rec.billing_period_end, date(2018, 7, 5))
        self.assertEqual(rec.consumption_kwh, 210.0)
        self.assertEqual(rec.bill_amount, 1420.0)
        self.assertEqual(rec.period_label, "Jul 2018")

    def test_regression_msedcl_curr_prev_column_order(self):
        """
        Regression test: Verify that MSEDCL's verified column order (Current Reading 1257,
        Previous Reading 1090, Multiplier 1.0, Billed Units 167) is correctly parsed via
        bidirectional meter reading arithmetic.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4599\n"
            "Current Reading Date: 26-APR-25\n"
            "Current Reading Unit Adj. Unit Total\n"
            "Reading prompt payment discount\n"
            "1257 1090 01 167 167 payment charges applicable\n"
            "Bill Amount Rs: 1,760.00\n"
        )
        pdf_bytes = create_synthetic_scanned_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="msedcl_order.pdf")

        self.assertEqual(len(extracted.records), 1)
        rec = extracted.records[0]
        self.assertEqual(rec.billing_period_start.year, 2025)
        self.assertEqual(rec.billing_period_start.month, 4)
        self.assertEqual(rec.consumption_kwh, 167.0)
        self.assertEqual(rec.bill_amount, 1760.0)

    def test_regression_currency_colon_format(self):
        """
        Regression test: Verify that 'Rs: 1,760.00' and other colon-attached currency formats
        are correctly recognized by the bill amount parser.
        """
        # Test ASCII variants via synthetic PDF (matplotlib's default font drops unicode currency glyphs)
        variants = [
            "Bill Amount Rs: 1,760.00",
            "Bill Amount Rs.: 1,760.00",
            "Net Bill Amount INR: 1,760.00",
        ]
        for v in variants:
            text = f"Bill of supply For: APR-2025\nConsumer No: 012345678901\nUnits: 167\n{v}"
            pdf_bytes = create_synthetic_pdf(text)
            extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="currency_colon.pdf")
            self.assertEqual(len(extracted.records), 1, f"Failed for variant: {v}")
            self.assertEqual(extracted.records[0].bill_amount, 1760.0, f"Failed amount for variant: {v}")

        # Test unicode currency symbols directly via extractor to avoid font rendering drop
        for sym in ["₹", "रु", "Rs"]:
            text_unicode = f"Bill of supply For: APR-2025\nConsumer No: 012345678901\nUnits: 167\nBill Amount {sym}: 1,760.00"
            rec, _ = self.parser._extract_current_period_record(text_unicode)
            self.assertIsNotNone(rec, f"Failed to extract record for symbol: {sym}")
            self.assertEqual(rec.bill_amount, 1760.0, f"Failed bill amount for symbol: {sym}")

    def test_regression_exclude_regulatory_footnotes_and_full_dates(self):
        """
        Regression test: Verify that regulatory footnote sentences (e.g. '01 April 2020. on or before 09-MAY-25')
        and 'w.e.f.' notices are never extracted as historical consumption table records, while
        legitimate historical records are preserved.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901\n"
            "Units: 167\n"
            "Bill Amount Rs: 1,760.00\n"
            "Notice:\n"
            "01 April 2020. on or before 09-MAY-25\n"
            "Revised Tariff applicable w.e.f. 01-APR-20\n"
            "Historical Consumption:\n"
            "May 2024   220   2290.00\n"
            "Jun 2024   210   2150.00\n"
        )
        pdf_bytes = create_synthetic_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="footnotes.pdf")

        # Must have current cycle (Apr 2025) plus May 2024 and Jun 2024, NOT April 2020
        extracted_years_months = [(r.billing_period_start.year, r.billing_period_start.month) for r in extracted.records]
        self.assertNotIn((2020, 4), extracted_years_months, "April 2020 footnote must NOT be extracted as a record")
        self.assertIn((2025, 4), extracted_years_months, "April 2025 current cycle must be present")
        self.assertIn((2024, 5), extracted_years_months, "May 2024 historical record must be present")
        self.assertIn((2024, 6), extracted_years_months, "June 2024 historical record must be present")

    # -------------------------------------------------------------------------
    # Task 2.6B: Step 6 Comprehensive Four-Provider Bill Parsing Tests
    # -------------------------------------------------------------------------
    def test_task26b_1_jvvnl_k_number_extraction_labelled(self):
        """
        1. JVVNL K-number extraction:
        Verify extraction from 'K.No.', 'K Number', and 'K-Number' labels,
        and ensure meter numbers, old K-numbers, feeder codes, and binders are NOT confused.
        """
        variants = [
            ("K.No. 210123456789", "K.No. dot format"),
            ("K Number: 210123456789", "K Number space format"),
            ("K-Number: 210123456789", "K-Number hyphen format"),
        ]
        for label_text, desc in variants:
            text = (
                "JAIPUR VIDYUT VITRAN NIGAM LIMITED (JVVNL)\n"
                f"{label_text}\n"
                "Meter No: 88776655\n"
                "Old K-No: 111122223333\n"
                "Feeder Code: 9988   Binder: 1234\n"
                "Bill Month: 03/2024\n"
                "Bill Period: 01/03/2024 to 31/03/2024\n"
                "Units Consumed: 215 kWh\n"
                "Amount Payable: Rs. 1850.00\n"
            )
            consumer_no, bu, rr, meta = self.parser._detect_consumer_identifiers(text, provider_key="jvvnl")
            self.assertEqual(consumer_no, "210123456789", f"Failed for {desc}")
            self.assertEqual(meta.get("provider_detected"), "JVVNL")
            self.assertNotEqual(consumer_no, "88776655", "Meter number was incorrectly extracted as consumer number")
            self.assertNotEqual(consumer_no, "111122223333", "Old K-number was incorrectly extracted as consumer number")

    def test_task26b_2_bescom_account_id_and_rr_number_extraction(self):
        """
        2. BESCOM Account ID and RR Number extraction:
        Support 10-digit Account ID and alphanumeric RR Number (with subdivision prefix and hyphens).
        Do not silently treat arbitrary numbers as Account ID.
        """
        text = (
            "BANGALORE ELECTRICITY SUPPLY COMPANY LIMITED (BESCOM)\n"
            "Account ID: 1029384756\n"
            "RR No: E3-45678\n"
            "Bill Date: 15/02/2024\n"
            "Period: 01/01/2024 to 31/01/2024\n"
            "Consumption Units: 180\n"
            "Total Due: Rs. 1420.00\n"
        )
        consumer_no, bu, rr, meta = self.parser._detect_consumer_identifiers(text, provider_key="bescom")
        self.assertEqual(consumer_no, "1029384756")
        self.assertEqual(rr, "E3-45678")
        self.assertEqual(meta.get("provider_detected"), "BESCOM")

        # Test arbitrary numbers without label are not treated as Account ID
        text_unlabeled = (
            "BESCOM ELECTRICITY BILL\n"
            "Some Code: 554433221199\n"
            "Units: 150\n"
        )
        c_no, _, _, _ = self.parser._detect_consumer_identifiers(text_unlabeled, provider_key="bescom")
        self.assertIsNone(c_no, "Arbitrary 12-digit number should NOT be silently treated as 10-digit BESCOM Account ID")

    def test_task26b_3_tangedco_hyphenated_and_continuous_consumer_number(self):
        """
        3. TANGEDCO hyphenated and continuous consumer-number extraction:
        Normalize identifier representation while preserving raw source text for auditability.
        """
        # Hyphenated
        text_hyphen = (
            "TAMIL NADU GENERATION AND DISTRIBUTION CORPORATION (TANGEDCO)\n"
            "Consumer No: 02-123-456-789\n"
            "Section: 02   Distribution: 123\n"
            "Reading Date: 10-02-2024\n"
            "Units: 310\n"
        )
        c_no, bu, rr, meta = self.parser._detect_consumer_identifiers(text_hyphen, provider_key="tangedco")
        self.assertEqual(c_no, "02123456789")
        self.assertEqual(meta.get("raw_source"), "02-123-456-789")
        self.assertEqual(meta.get("provider_detected"), "TANGEDCO")

        # Continuous numeric format
        text_cont = (
            "TANGEDCO Low Tension Bill\n"
            "Consumer No: 02123456789\n"
            "Units: 290\n"
        )
        c_no_cont, _, _, meta_cont = self.parser._detect_consumer_identifiers(text_cont, provider_key="tangedco")
        self.assertEqual(c_no_cont, "02123456789")
        self.assertEqual(meta_cont.get("provider_detected"), "TANGEDCO")

    def test_task26b_4_valid_tangedco_50_to_70_day_billing_period(self):
        """
        4. Valid TANGEDCO 50–70-day billing periods:
        Permit bi-monthly intervals for TANGEDCO, preserve total kWh (no synthetic division),
        and emit clear bi-monthly cycle warning.
        """
        text = (
            "TAMIL NADU GENERATION AND DISTRIBUTION CORPORATION LIMITED\n"
            "Consumer No: 02-123-456-789\n"
            "Billing Period: 05/01/2024 to 05/03/2024\n"
            "Units Consumed: 450.00 kWh\n"
            "Total Bill Amount: Rs. 1650.00\n"
        )
        rec, warning = self.parser._extract_current_period_record(text, provider_key="tangedco")
        self.assertIsNotNone(rec, "Valid TANGEDCO 60-day billing cycle should be extracted")
        self.assertEqual(rec.billing_period_start, date(2024, 1, 5))
        self.assertEqual(rec.billing_period_end, date(2024, 3, 5))
        self.assertEqual(rec.consumption_kwh, 450.0)
        self.assertEqual(rec.bill_amount, 1650.0)
        self.assertIn("Jan - Mar 2024", rec.period_label)
        self.assertIsNotNone(warning)
        self.assertTrue(any("bi-monthly billing cycle" in w.lower() for w in warning))

    def test_task26b_5_invalid_dates_and_monthly_provider_interval_regressions(self):
        """
        5. Invalid dates and monthly-provider interval regressions:
        Monthly providers (MSEDCL, BESCOM, JVVNL) must strictly reject 60-day intervals.
        Invalid dates (start >= end) must also be rejected across all providers.
        """
        text_60_days = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD. (MSEDCL)\n"
            "Consumer No: 012345678901\n"
            "Billing Period: 01/01/2024 to 01/03/2024\n"
            "Units: 450 kWh\n"
        )
        # MSEDCL must reject 60 days
        rec_msedcl, _ = self.parser._extract_current_period_record(text_60_days, provider_key="msedcl")
        self.assertIsNone(rec_msedcl, "MSEDCL must reject 60-day billing period (exceeds max 45 days)")

        # Inverted dates (start > end) must be rejected
        text_inverted = (
            "TANGEDCO\n"
            "Consumer No: 02123456789\n"
            "Billing Period: 01/05/2024 to 01/03/2024\n"
            "Units: 300\n"
        )
        rec_inv, _ = self.parser._extract_current_period_record(text_inverted, provider_key="tangedco")
        self.assertIsNone(rec_inv, "Inverted date range must be rejected")

    def test_task26b_6_msedcl_english_and_marathi_extraction_regressions(self):
        """
        6. MSEDCL English and Marathi extraction regressions:
        Ensure no regressions in extracting 12-digit consumer number and 4-digit BU in English and Marathi bills.
        """
        # English
        text_en = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Bill Date: 10-04-2024\n"
            "Units: 167\n"
        )
        c_no_en, bu_en, _, _ = self.parser._detect_consumer_identifiers(text_en, provider_key="msedcl")
        self.assertEqual(c_no_en, "012345678901")
        self.assertEqual(bu_en, "4567")

        # Marathi
        text_mr = (
            "महाराष्ट्र राज्य विद्युत वितरण कंपनी मर्यादित (महावितरण)\n"
            "ग्राहक क्रमांक (Consumer No): 012345678901\n"
            "बिलिंग युनिट (BU): 4567\n"
            "वापर युनिट्स: 167\n"
        )
        c_no_mr, bu_mr, _, _ = self.parser._detect_consumer_identifiers(text_mr, provider_key="msedcl")
        self.assertEqual(c_no_mr, "012345678901")
        self.assertEqual(bu_mr, "4567")

    def test_task26b_7_direct_jpeg_png_uploads(self):
        """
        7. Direct JPEG/PNG uploads:
        Verify parse_bill_file extracts structured records from raster PNG and JPEG images.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        # Test PNG
        png_bytes = create_synthetic_image_file(bill_text, "png")
        extracted_png = self.parser.parse_bill_file(png_bytes, filename="bill.png")
        self.assertEqual(extracted_png.extraction_source, "image")
        self.assertEqual(extracted_png.consumer_number, "012345678901")
        self.assertEqual(extracted_png.billing_unit, "4567")
        self.assertEqual(len(extracted_png.records), 1)
        self.assertEqual(extracted_png.records[0].consumption_kwh, 167.0)

        # Test JPEG
        jpeg_bytes = create_synthetic_image_file(bill_text, "jpeg")
        extracted_jpeg = self.parser.parse_bill_file(jpeg_bytes, filename="bill.jpg")
        self.assertEqual(extracted_jpeg.extraction_source, "image")
        self.assertEqual(extracted_jpeg.consumer_number, "012345678901")
        self.assertEqual(len(extracted_jpeg.records), 1)

    def test_task26b_8_malformed_unsupported_and_oversized_upload_rejection(self):
        """
        8. Malformed, unsupported and oversized upload rejection:
        Verify parse_bill_file safely raises ValueError for bad extensions, empty files,
        oversized files, and corrupted raster bytes.
        """
        # Unsupported extension
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_file(b"some content", filename="bill.docx")
        self.assertIn("unsupported file format", str(ctx.exception).lower())

        # Empty file
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_file(b"", filename="bill.png")
        self.assertIn("empty", str(ctx.exception).lower())

        # Oversized file (> 10MB)
        oversized_bytes = b"x" * (10 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_file(oversized_bytes, filename="bill.png")
        self.assertIn("exceeds maximum allowed limit", str(ctx.exception).lower())

        # Missing magic header
        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_bill_file(b"bad_image_data_not_real", filename="corrupt.png")
        self.assertTrue(
            any(phrase in str(ctx.exception).lower() for phrase in ["missing png or jpeg header", "cannot be opened or is corrupted"]),
            f"Unexpected error message: {ctx.exception}",
        )

        # Corrupted payload with valid PNG header
        corrupted_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 30
        with self.assertRaises(ValueError) as ctx2:
            self.parser.parse_bill_file(corrupted_png, filename="damaged.png")
        self.assertIn("cannot be opened or is corrupted", str(ctx2.exception).lower())

    def test_task26b_9_historical_table_extraction_and_deduplication(self):
        """
        9. Historical table extraction and duplicate/overlapping-period validation:
        Verify extraction, chronological sorting, and deduplication of cycles.
        """
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
            "Historical Consumption Table:\n"
            "Jan 2025   180   1890.00\n"
            "Feb 2025   175   1820.00\n"
            "Mar 2025   190   1950.00\n"
            "Mar 2025   190   1950.00\n"  # Duplicate row
        )
        pdf_bytes = create_synthetic_pdf(bill_text)
        extracted = self.parser.parse_bill_pdf(pdf_bytes, filename="history_dedup.pdf")

        # Verify no duplicate March 2025 records
        dates = [(r.billing_period_start.year, r.billing_period_start.month) for r in extracted.records]
        self.assertEqual(len(dates), len(set(dates)), "Duplicate historical months must be deduplicated")
        # Verify chronological order
        self.assertTrue(all(dates[i] <= dates[i+1] for i in range(len(dates)-1)), "Records must be sorted chronologically")

    def test_task26b_10_explicit_warnings_for_bimonthly_monthly_forecasting_workflow(self):
        """
        10. Explicit warnings where a bi-monthly record cannot safely be represented in the monthly workflow:
        Verify MonthlyForecastService emits clear warning/disclosure when fed bi-monthly input.
        """
        rec = MonthlyConsumptionRecord(
            billing_period_start=date(2024, 1, 1),
            billing_period_end=date(2024, 3, 1),  # 60 days
            consumption_kwh=420.0,
            bill_amount=1650.0,
            period_label="Jan - Mar 2024",
        )
        req = MonthlyForecastRequest(
            records=[rec],
            horizon_months=1,
            provider_id="tangedco",
            is_live=True,
        )
        forecast_res = monthly_forecast_service.generate_forecast(req)
        self.assertTrue(
            any("bi-monthly" in w.lower() for w in forecast_res.warnings),
            "Forecast warnings must explicitly mention bi-monthly record handling",
        )


    def test_task26c_1_ocr_exif_orientation_6_smartphone_jpeg(self):
        """Regression test: verify smartphone camera JPEG with EXIF orientation 6 is normalized and parsed successfully."""
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        jpeg_bytes = create_synthetic_camera_jpeg(bill_text, orientation=6)
        extracted = self.parser.parse_bill_file(jpeg_bytes, filename="1.jpeg")
        self.assertEqual(extracted.extraction_source, "image")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(extracted.billing_unit, "4567")
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].consumption_kwh, 167.0)

    def test_task26c_2_ocr_sideways_image_without_exif_rescued(self):
        """Regression test: verify unoriented sideways image without EXIF tag is rescued by multi-angle fallback."""
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
        ax.axis("off")
        ax.text(0.05, 0.95, bill_text, fontsize=11, verticalalignment="top", fontfamily="sans-serif")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        base = Image.open(buf).convert("RGB")
        sideways = base.transpose(Image.ROTATE_90)
        jpeg_buf = io.BytesIO()
        sideways.save(jpeg_buf, format="JPEG")

        extracted = self.parser.parse_bill_file(jpeg_buf.getvalue(), filename="sideways_bill.jpg")
        self.assertEqual(extracted.extraction_source, "image")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].consumption_kwh, 167.0)

    def test_task26c_3_ocr_transparent_png_extracted_successfully(self):
        """Regression test: verify RGBA PNG with transparent background composites over white and extracts cleanly."""
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        png_bytes = create_synthetic_image_file(bill_text, "png")
        rgba_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        buf = io.BytesIO()
        rgba_img.save(buf, format="PNG")

        extracted = self.parser.parse_bill_file(buf.getvalue(), filename="transparent_bill.png")
        self.assertEqual(extracted.extraction_source, "image")
        self.assertEqual(extracted.consumer_number, "012345678901")
        self.assertEqual(len(extracted.records), 1)
        self.assertEqual(extracted.records[0].consumption_kwh, 167.0)

    def test_task26c_4_ocr_blank_image_returns_ocr_limitation_warning(self):
        """Regression test: verify genuine blank image produces clear OCR limitation warning."""
        blank_img = Image.new("RGB", (600, 800), color="white")
        buf = io.BytesIO()
        blank_img.save(buf, format="JPEG")
        extracted = self.parser.parse_bill_file(buf.getvalue(), filename="blank.jpg")
        self.assertEqual(len(extracted.records), 0)
        self.assertTrue(extracted.raw_metadata.get("ocr_unreadable"))
        self.assertTrue(
            any("optical character recognition (ocr)" in w.lower() for w in extracted.warnings)
        )

    def test_task26c_5_ocr_runtime_failure_is_not_masked_by_scan_quality(self):
        """Regression test: verify runtime/engine initialization failure is not masked by generic scan quality message."""
        from unittest.mock import patch
        blank_img = Image.new("RGB", (200, 200), color="white")
        buf = io.BytesIO()
        blank_img.save(buf, format="JPEG")

        with patch.object(self.parser, "_extract_text_from_image") as mock_extract:
            mock_extract.return_value = (
                "",
                ["Windows Native OCR runtime bindings (winrt) are not available. Image OCR disabled."],
                {"ocr_runtime_available": False, "page_dimensions": [], "characters_per_page": []},
            )
            extracted = self.parser.parse_bill_image(buf.getvalue(), filename="1.jpeg")
            self.assertEqual(len(extracted.records), 0)
            self.assertIn("not available", extracted.warnings[0].lower())
            self.assertFalse(extracted.warnings[0].startswith("Image could not be read clearly"))

    def test_task26c_6_exact_fixture_1_jpeg_diagnostic_rejection(self):
        """Regression test: verify exact fixture 1.jpeg (252x356 px) is tested directly, rejected as unreadable due to low resolution."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "1.jpeg"
        self.assertTrue(fixture_path.exists(), f"Fixture file not found: {fixture_path}")

        file_bytes = fixture_path.read_bytes()
        self.assertEqual(len(file_bytes), 24141)

        # 1. Direct _extract_text_from_image call
        text, warnings, diag = self.parser._extract_text_from_image(file_bytes)
        self.assertEqual(diag.get("original_dimensions"), [252, 356])
        alnum_count = len(re.findall(r"[A-Za-z0-9]", text))
        self.assertLess(alnum_count, 15, f"Expected < 15 characters from 252x356 thumbnail, got {alnum_count}")

        # 2. Direct parse_bill_image call
        extracted = self.parser.parse_bill_image(file_bytes, filename="1.jpeg")
        self.assertEqual(len(extracted.records), 0)
        self.assertTrue(extracted.raw_metadata.get("ocr_unreadable"))
        self.assertEqual(extracted.raw_metadata.get("original_dimensions"), [252, 356])
        self.assertTrue(
            any("252x356" in w and "too low for optical character recognition" in w.lower() for w in extracted.warnings),
            f"Expected resolution diagnostic in warnings, got: {extracted.warnings}",
        )

    def test_task27_exact_fixture_final_image_jpeg_extraction(self):
        """Regression test: verify exact fixture final image.jpeg is parsed directly and extracts valid MSEDCL fields."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "final image.jpeg"
        self.assertTrue(fixture_path.exists(), f"Fixture file not found: {fixture_path}")

        file_bytes = fixture_path.read_bytes()
        self.assertEqual(len(file_bytes), 191464)

        # 1. Direct parse_bill_image call
        extracted = self.parser.parse_bill_image(file_bytes, filename="final image.jpeg")

        # Must extract exactly 1 billing period
        self.assertEqual(len(extracted.records), 1, f"Expected 1 record, got {len(extracted.records)}. Warnings: {extracted.warnings}")
        rec = extracted.records[0]

        # Valid billing dates: 31-05-2024 to 25-06-2024
        self.assertEqual(rec.billing_period_start, date(2024, 5, 31))
        self.assertEqual(rec.billing_period_end, date(2024, 6, 25))
        self.assertEqual(rec.period_label, "Jun 2024")

        # Meter consumption: 2.0 kWh (distinct from 4.00 KW sanctioned load, 3750.00 deposit, 150.00 bill amount)
        self.assertEqual(rec.consumption_kwh, 2.0)

        # Bill amount: 150.0 Rs (distinct from 160.00 delayed payment surcharge and 3750.00 security deposit)
        self.assertEqual(rec.bill_amount, 150.0)

        # Provider and identifiers
        self.assertEqual(extracted.utility_provider, "MSEDCL (Mahavitaran)")
        self.assertEqual(extracted.consumer_number, "160224871721")
        self.assertEqual(extracted.billing_unit, "4599")

        # Confidence and source
        self.assertEqual(extracted.extraction_source, BillExtractionSource.IMAGE)
        self.assertGreater(extracted.extraction_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()


