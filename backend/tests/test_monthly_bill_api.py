"""
API Integration tests for Electricity Bill PDF Upload Endpoint (POST /api/v1/forecasts/monthly/parse-bill).

Covers:
1. Strict HTTP 401 rejection for unauthenticated requests.
2. Successful parsing of synthetic MSEDCL e-bill (multi-period, provider, consumer no, BU).
3. Successful parsing of synthetic generic utility bill (single cycle).
4. Validation and rejection of non-PDF file extensions (HTTP 422).
5. Validation and rejection of empty 0-byte uploads (HTTP 400).
6. Validation and rejection of invalid / non-PDF content missing %PDF- header (HTTP 422).
7. Validation and rejection of oversized files > 10 MB (HTTP 413).
8. Safe rejection of image-only / scanned PDFs with clear OCR notice (HTTP 422).
9. Safe rejection of encrypted / password-protected PDFs (HTTP 422).
10. Safe rejection of non-utility text documents with 0 billing records (HTTP 422).
11. Verification that no data is persisted to the database during parsing.
"""

import io
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.services.record_service import record_service
from backend.tests.test_bill_parser_service import (
    create_synthetic_camera_jpeg,
    create_synthetic_pdf,
    create_synthetic_scanned_pdf,
    create_synthetic_two_page_scanned_pdf,
)


class TestMonthlyBillPdfApi(unittest.TestCase):
    """Integration test suite for POST /api/v1/forecasts/monthly/parse-bill."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX
        cls.parse_url = f"{cls.prefix}/forecasts/monthly/parse-bill"

        # Register and authenticate a test user
        cls.username = f"bill_api_user_{uuid.uuid4().hex[:8]}"
        cls.password = "ValidPassword123!"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": cls.username, "password": cls.password},
        )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": cls.username, "password": cls.password},
        )
        cls.token = login_res.json()["access_token"]
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}

    def test_unauthenticated_request_returns_401(self):
        """Unauthenticated POST /monthly/parse-bill must return HTTP 401."""
        sample_pdf = create_synthetic_pdf("TEST BILL")
        res = self.client.post(
            self.parse_url,
            files={"file": ("bill.pdf", sample_pdf, "application/pdf")},
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_successful_parsing_synthetic_msedcl_pdf(self):
        """Authenticated caller uploading synthetic MSEDCL bill receives structured ExtractedBillData."""
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

        res = self.client.post(
            self.parse_url,
            files={"file": ("msedcl_jan2024.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["extraction_source"], "pdf")
        self.assertEqual(data["utility_provider"], "MSEDCL (Mahavitaran)")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(data["billing_unit"], "4123")
        self.assertEqual(len(data["records"]), 4)

        # Check chronology and values
        self.assertEqual(data["records"][0]["billing_period_start"], "2023-10-01")
        self.assertEqual(data["records"][3]["billing_period_start"], "2024-01-01")
        self.assertEqual(data["records"][3]["consumption_kwh"], 250.0)
        self.assertEqual(data["records"][3]["bill_amount"], 2676.70)
        self.assertGreaterEqual(data["extraction_confidence"], 0.85)

    def test_successful_parsing_generic_utility_pdf(self):
        """Authenticated caller uploading generic utility bill receives structured record."""
        generic_text = (
            "STATE POWER CORPORATION LIMITED\n"
            "ACCOUNT NO: 9876543210\n"
            "BILLING PERIOD: 01/03/2024 to 31/03/2024\n"
            "TOTAL UNITS BILLED: 310.5 kWh\n"
            "AMOUNT PAYABLE: Rs 2950.00\n"
        )
        pdf_bytes = create_synthetic_pdf(generic_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("generic_bill.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["billing_period_start"], "2024-03-01")
        self.assertEqual(data["records"][0]["billing_period_end"], "2024-03-31")
        self.assertEqual(data["records"][0]["consumption_kwh"], 310.5)
        self.assertEqual(data["records"][0]["bill_amount"], 2950.00)
        self.assertGreaterEqual(data["extraction_confidence"], 0.70)

    def test_successful_parsing_scanned_pdf_via_ocr(self):
        """Authenticated caller uploading image-only scanned bill receives structured records via OCR."""
        sample_scanned_text = (
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
        scanned_pdf_bytes = create_synthetic_scanned_pdf(sample_scanned_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("scanned_msedcl.pdf", scanned_pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["utility_provider"], "MSEDCL (Mahavitaran)")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(data["billing_unit"], "4123")
        self.assertEqual(len(data["records"]), 4)
        self.assertEqual(data["records"][3]["consumption_kwh"], 250.0)
        self.assertEqual(data["records"][3]["bill_amount"], 2676.70)
        self.assertIsNotNone(data["extraction_confidence"])
        self.assertLessEqual(data["extraction_confidence"], 0.90)
        self.assertGreaterEqual(data["extraction_confidence"], 0.70)
        self.assertTrue(data["raw_metadata"].get("ocr_applied"))
        self.assertEqual(data["raw_metadata"].get("ocr_engine"), "Windows.Media.Ocr")

    def test_invalid_extension_rejected(self):
        """Upload with unsupported non-bill extension must be rejected with HTTP 422."""
        res = self.client.post(
            self.parse_url,
            files={"file": ("bill.csv", b"col1,col2\n1,2", "text/csv")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("unsupported file format", res.json()["detail"].lower())

    def test_empty_file_rejected(self):
        """Upload with 0-byte content must be rejected with HTTP 400."""
        res = self.client.post(
            self.parse_url,
            files={"file": ("empty.pdf", b"", "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_non_pdf_content_rejected(self):
        """Upload lacking %PDF- magic header must be rejected with HTTP 422."""
        res = self.client.post(
            self.parse_url,
            files={"file": ("fake.pdf", b"This is plain text with a .pdf name.", "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("missing %pdf- header", res.json()["detail"].lower())

    def test_oversized_pdf_rejected(self):
        """Upload exceeding 10 MB must be rejected with HTTP 413."""
        oversized = b"%PDF-1.4\n" + (b"0" * (11 * 1024 * 1024))
        res = self.client.post(
            self.parse_url,
            files={"file": ("oversized.pdf", oversized, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds maximum allowed limit", res.json()["detail"].lower())

    def test_scanned_image_only_pdf_returns_clear_error(self):
        """Scanned image-only PDF with no text stream returns HTTP 422 with OCR notice."""
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        blank_pdf_bytes = buf.getvalue()

        res = self.client.post(
            self.parse_url,
            files={"file": ("scanned_scan.pdf", blank_pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertTrue(
            "ocr" in res.json()["detail"].lower()
            or "image-only" in res.json()["detail"].lower(),
            f"Expected OCR notice in error detail: {res.json()['detail']}",
        )

    def test_encrypted_pdf_returns_clear_error(self):
        """Encrypted PDF returns HTTP 422 with clear password / decryption advice."""
        encrypted_pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"3 0 obj\n<< /Filter /Standard /V 2 /R 3 >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R /Encrypt 3 0 R >>\n%%EOF"
        )
        res = self.client.post(
            self.parse_url,
            files={"file": ("locked_bill.pdf", encrypted_pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json()["detail"].lower()
        self.assertTrue(
            "encrypted" in detail or "password" in detail,
            f"Expected encryption notice in error detail: {detail}",
        )

    def test_non_utility_document_rejected(self):
        """Text PDF without utility fields returns HTTP 422 with clear notice."""
        prose_text = (
            "Research Paper on Neural Network Architectures\n"
            "This document presents an overview of transformer attention mechanisms.\n"
            "It does not contain any consumer electricity bills or kWh measurements."
        )
        pdf_bytes = create_synthetic_pdf(prose_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("could not reliably identify", res.json()["detail"].lower())

    def test_no_database_persistence_or_side_effects(self):
        """Parsing a bill must not create any records in the forecast database."""
        initial_records = record_service.list_records(owner_user_id=None)
        initial_count = len(initial_records)

        generic_text = (
            "UTILITY BILL\n"
            "ACCOUNT NO: 5544332211\n"
            "BILLING PERIOD: 01/02/2024 to 29/02/2024\n"
            "TOTAL UNITS: 200.0 kWh\n"
            "AMOUNT PAYABLE: Rs 1800.00\n"
        )
        pdf_bytes = create_synthetic_pdf(generic_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("ephemeral_bill.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200)

        after_records = record_service.list_records(owner_user_id=None)
        self.assertEqual(len(after_records), initial_count, "Database record count must not change on bill parsing")

    def test_successful_parsing_scanned_single_month_april_bill(self):
        """Synthetic scanned single-month April bill parses via OCR and returns 200 with 1 record."""
        single_bill_text = (
            "MAHAVITARAN - MSEDCL ELECTRICITY BILL\n"
            "CONSUMER NO: 012345678901   BILLING UNIT: 4123\n"
            "BILL MONTH: APR-24\n"
            "PREV READING DATE: 01/03/24\n"
            "READING DATE: 01/04/24\n"
            "UNITS BILLED: 250.00\n"
            "NET AMOUNT: Rs. 2676.70\n"
        )
        pdf_bytes = create_synthetic_scanned_pdf(single_bill_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("Electricity Bill April.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["utility_provider"], "MSEDCL (Mahavitaran)")
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["consumption_kwh"], 250.0)
        self.assertEqual(data["records"][0]["billing_period_start"], "2024-03-01")
        self.assertEqual(data["records"][0]["billing_period_end"], "2024-04-01")

    def test_unreadable_scan_returns_actionable_error_detail(self):
        """Scanned document with unreadable non-utility text returns 422 with actionable failure detail, not OCR info."""
        prose_text = (
            "Abstract algebra studies algebraic structures such as groups, rings, and fields.\n"
            "This scanned document has no meter readings or electricity utility data."
        )
        pdf_bytes = create_synthetic_scanned_pdf(prose_text)

        res = self.client.post(
            self.parse_url,
            files={"file": ("scanned_paper.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json()["detail"]
        # Detail must NOT be the informational OCR fallback note
        self.assertFalse(detail.lower().startswith("text extracted using windows native ocr"))
        self.assertTrue(
            "could not reliably identify" in detail.lower()
            or "unreadable" in detail.lower()
            or "no billing records" in detail.lower(),
            f"Expected actionable error detail, got: {detail}",
        )
    def test_successful_parsing_scanned_two_page_msedcl_bill(self):
        """Synthetic scanned 2-page MSEDCL bill parses via OCR and returns 200 with 6 records."""
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

        res = self.client.post(
            self.parse_url,
            files={"file": ("scanned_msedcl_2page.pdf", pdf_bytes, "application/pdf")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["utility_provider"], "MSEDCL (Mahavitaran)")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(data["billing_unit"], "4123")
        self.assertEqual(len(data["records"]), 6)
        # Check first (Nov 2024) and last (Apr 2025) records
        self.assertEqual(data["records"][0]["billing_period_start"], "2024-11-01")
        self.assertEqual(data["records"][0]["consumption_kwh"], 220.0)
        self.assertEqual(data["records"][5]["billing_period_start"], "2025-04-01")
        self.assertEqual(data["records"][5]["consumption_kwh"], 250.0)
        self.assertEqual(data["records"][5]["bill_amount"], 2630.0)

    def test_direct_png_bill_upload_successful(self):
        """Authenticated caller uploading a genuine PNG image bill extracts structured records."""
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

        res = self.client.post(
            self.parse_url,
            files={"file": ("bill_scan.png", buf.getvalue(), "image/png")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(data["billing_unit"], "4567")
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["consumption_kwh"], 167.0)

    def test_direct_jpeg_bill_upload_successful(self):
        """Authenticated caller uploading a genuine JPEG image bill extracts structured records."""
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
        fig.savefig(buf, format="jpeg", bbox_inches="tight")
        plt.close(fig)

        res = self.client.post(
            self.parse_url,
            files={"file": ("bill_scan.jpg", buf.getvalue(), "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["consumption_kwh"], 167.0)

    def test_disguised_file_invalid_magic_bytes_rejected(self):
        """Upload with .png extension but plain text content is rejected for invalid magic bytes."""
        res = self.client.post(
            self.parse_url,
            files={"file": ("fake_image.png", b"Not a real PNG header content", "image/png")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("invalid magic bytes", res.json()["detail"].lower())

    def test_corrupted_image_rejected(self):
        """Upload with PNG header but corrupted body payload is rejected as malformed."""
        corrupted_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # Valid header, broken payload
        res = self.client.post(
            self.parse_url,
            files={"file": ("broken.png", corrupted_png, "image/png")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("corrupted or malformed", res.json()["detail"].lower())


    def test_upload_smartphone_jpeg_with_exif_orientation_succeeds(self):
        """Regression test: verify upload of 1.jpeg captured with smartphone camera EXIF orientation 6 succeeds with HTTP 200."""
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        jpeg_bytes = create_synthetic_camera_jpeg(bill_text, orientation=6)
        res = self.client.post(
            self.parse_url,
            files={"file": ("1.jpeg", jpeg_bytes, "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(data["billing_unit"], "4567")
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["consumption_kwh"], 167.0)

    def test_upload_blank_jpeg_returns_422_with_clear_ocr_notice(self):
        """Regression test: verify upload of a blank 1.jpeg returns HTTP 422 with clear OCR notice."""
        from PIL import Image
        blank = Image.new("RGB", (600, 800), color="white")
        buf = io.BytesIO()
        blank.save(buf, format="JPEG")
        res = self.client.post(
            self.parse_url,
            files={"file": ("1.jpeg", buf.getvalue(), "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        self.assertTrue(
            "optical character recognition (ocr)" in res.json()["detail"].lower()
            or "image could not be read clearly" in res.json()["detail"].lower()
        )

    def test_upload_jpeg_with_ocr_runtime_failure_returns_specific_diagnostic_422(self):
        """Regression test: verify OCR runtime failure returns specific diagnostic detail, not generic scan quality text."""
        from unittest.mock import patch
        from backend.app.routers.forecasts import bill_parser_service
        from PIL import Image
        dummy = Image.new("RGB", (200, 200), color="white")
        buf = io.BytesIO()
        dummy.save(buf, format="JPEG")

        with patch.object(bill_parser_service, "_extract_text_from_image") as mock_extract:
            mock_extract.return_value = (
                "",
                ["Windows Native OCR runtime bindings (winrt) are not available. Image OCR disabled."],
                {"ocr_runtime_available": False, "page_dimensions": [], "characters_per_page": []},
            )
            res = self.client.post(
                self.parse_url,
                files={"file": ("1.jpeg", buf.getvalue(), "image/jpeg")},
                headers=self.auth_headers,
            )
            self.assertEqual(res.status_code, 422)
            self.assertIn("not available", res.json()["detail"].lower())
            self.assertFalse(res.json()["detail"].startswith("Image could not be read clearly"))

    def test_upload_exact_fixture_1_jpeg_returns_422_with_resolution_diagnostic(self):
        """Regression test: verify uploading the exact 1.jpeg fixture file returns HTTP 422 with resolution diagnostic detail."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "1.jpeg"
        self.assertTrue(fixture_path.exists(), f"Fixture file not found: {fixture_path}")

        with open(fixture_path, "rb") as f:
            file_bytes = f.read()

        res = self.client.post(
            self.parse_url,
            files={"file": ("1.jpeg", file_bytes, "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 422)
        detail = res.json()["detail"]
        self.assertIn("252x356", detail)
        self.assertIn("too low for optical character recognition", detail.lower())

    def test_upload_high_res_crisp_image_succeeds_extraction(self):
        """Regression test: verify uploading a high-resolution, legible bill image succeeds with HTTP 200."""
        bill_text = (
            "MAHARASHTRA STATE ELECTRICITY DISTRIBUTION CO. LTD.\n"
            "Bill of supply For: APR-2025\n"
            "Consumer No: 012345678901   Billing Unit: 4567\n"
            "Units: 167\n"
            "Bill Amount Rs: 1760.00\n"
        )
        jpeg_bytes = create_synthetic_camera_jpeg(bill_text, orientation=1)
        res = self.client.post(
            self.parse_url,
            files={"file": ("crisp_bill.jpeg", jpeg_bytes, "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["consumer_number"], "012345678901")
        self.assertEqual(len(data["records"]), 1)
        self.assertEqual(data["records"][0]["consumption_kwh"], 167.0)

    def test_upload_exact_fixture_final_image_jpeg_succeeds_extraction(self):
        """Regression test: verify uploading exact fixture final image.jpeg succeeds with HTTP 200 and extracts valid MSEDCL records."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "final image.jpeg"
        self.assertTrue(fixture_path.exists(), f"Fixture file not found: {fixture_path}")

        file_bytes = fixture_path.read_bytes()
        res = self.client.post(
            self.parse_url,
            files={"file": ("final image.jpeg", file_bytes, "image/jpeg")},
            headers=self.auth_headers,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["extraction_source"], "image")
        self.assertEqual(data["utility_provider"], "MSEDCL (Mahavitaran)")
        self.assertEqual(data["consumer_number"], "160224871721")
        self.assertEqual(data["billing_unit"], "4599")
        self.assertEqual(len(data["records"]), 1)

        rec = data["records"][0]
        self.assertEqual(rec["billing_period_start"], "2024-05-31")
        self.assertEqual(rec["billing_period_end"], "2024-06-25")
        self.assertEqual(rec["period_label"], "Jun 2024")
        self.assertEqual(rec["consumption_kwh"], 2.0)
        self.assertEqual(rec["bill_amount"], 150.0)


if __name__ == "__main__":
    unittest.main()

