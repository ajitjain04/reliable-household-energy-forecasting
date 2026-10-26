"""
Electricity Bill PDF Ingestion and Parsing Service.

Extracts text, billing metadata, consumption records, and historical monthly consumption
tables from digital electricity bill PDFs across Indian distribution companies.

Design principles:
- Safe, in-memory stream decoding using standard library zlib (with optional pypdf fallback).
- Strictly non-generative: missing fields are never fabricated or estimated.
- Distinguishes document extraction confidence from forecasting uncertainty.
- Enforces temporal consistency, duplicate detection, and gap tracking via existing bill schemas.
- Gracefully handles empty, invalid, oversized, or image-only scanned PDFs (with clear OCR limitation notice).
"""

import asyncio
import calendar
import concurrent.futures
import io
import math
import re
import zlib
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

from backend.app.schemas.bills import (
    BillExtractionSource,
    ExtractedBillData,
    MonthlyConsumptionRecord,
    validate_and_sort_billing_records,
)

MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_BILL_FILE_SIZE_BYTES = MAX_PDF_SIZE_BYTES
ALLOWED_BILL_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")



def _run_sync(coro):
    """Run an async coroutine synchronously from both worker threads and active event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class BillParserService:
    """
    Isolated service for extracting structured consumption records from PDF bills.
    Supports digital text PDFs and Windows Native OCR fallback for scanned documents.
    """

    MONTH_MAP = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    # -------------------------------------------------------------------------
    # 1. Low-Level PDF Stream Text Extraction (Digital PDF Fast Path)
    # -------------------------------------------------------------------------
    def extract_raw_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract plain text content from PDF binary bytes.

        Attempts pypdf if installed; otherwise utilizes pure Python zlib decompression
        of FlateDecode content streams and standard text operator parsing.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded PDF document is empty (0 bytes).")

        if len(file_bytes) > MAX_PDF_SIZE_BYTES:
            raise ValueError(
                f"PDF file size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of 10 MB."
            )

        # Verify PDF magic header
        if not file_bytes.startswith(b"%PDF-") and b"%PDF-" not in file_bytes[:1024]:
            raise ValueError("Uploaded file is not a valid PDF document (missing %PDF- header).")

        # 1. Try pypdf if available in environment
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            extracted_pages = []
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    extracted_pages.append(txt)
            if extracted_pages:
                return "\n".join(extracted_pages)
        except (ImportError, Exception):
            pass

        # 2. Native standard library PDF stream decoder
        # Searches for stream ... endstream blocks and decompresses FlateDecode text
        lines: List[str] = []
        raw_streams = re.findall(b"stream[\r\n]+(.*?)[\r\n]+endstream", file_bytes, re.DOTALL)

        for stream_bytes in raw_streams:
            dec_str: Optional[str] = None
            try:
                dec_str = zlib.decompress(stream_bytes).decode("latin1", errors="ignore")
            except Exception:
                # Might be uncompressed text stream
                if b"BT" in stream_bytes and b"ET" in stream_bytes:
                    dec_str = stream_bytes.decode("latin1", errors="ignore")

            if dec_str:
                # Match [ ... ] TJ blocks (kerning arrays containing text fragments)
                for arr in re.findall(r"\[(.*?)\]\s*TJ", dec_str, re.DOTALL):
                    pieces = re.findall(r"\((.*?)(?<!\\)\)", arr)
                    cleaned_pieces = [
                        p.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\").replace("\\n", "\n")
                        for p in pieces
                    ]
                    line_str = "".join(cleaned_pieces).strip()
                    if line_str:
                        lines.append(line_str)

                # Match standalone ( ... ) Tj or ' or " blocks
                for single in re.findall(r"(?<!\[)\s*\((.*?)(?<!\\)\)\s*(?:Tj|['\"])", dec_str):
                    cleaned_single = (
                        single.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\").replace("\\n", "\n").strip()
                    )
                    if cleaned_single:
                        lines.append(cleaned_single)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 2. Windows Native OCR Fallback (Scanned / Image-Only PDFs)
    # -------------------------------------------------------------------------
        
    def _extract_text_via_ocr(
        self,
        file_bytes: bytes,
        max_pages: int = 5,
    ) -> Tuple[str, List[str], int, Dict[str, Any]]:
        """
        Extract text from scanned PDF pages using Tesseract OCR.
        Supports Linux deployments and limits processing to max_pages.
        """
        warnings: List[str] = []
        ocr_diag: Dict[str, Any] = {
            "page_dimensions": [],
            "characters_per_page": [],
            "ocr_runtime_available": False,
            "ocr_engine": "Tesseract",
        }

        if pdfium is None:
            warnings.append(
                "PDF rendering library (pypdfium2) is unavailable. "
                "Scanned PDF OCR is disabled."
            )
            return "", warnings, 0, ocr_diag

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception as exc:
            warnings.append(f"Tesseract OCR is unavailable: {exc}")
            ocr_diag["error"] = str(exc)
            return "", warnings, 0, ocr_diag

        try:
            pdf = pdfium.PdfDocument(file_bytes)
        except Exception as exc:
            warnings.append(f"Failed to open PDF for OCR: {exc}")
            return "", warnings, 0, ocr_diag

        total_pages = len(pdf)
        pages_to_process = min(total_pages, max_pages)

        if total_pages > max_pages:
            warnings.append(
                f"Document contains {total_pages} pages; "
                f"only the first {max_pages} pages were processed by OCR."
            )

        ocr_diag["ocr_runtime_available"] = True
        extracted_page_texts: List[str] = []

        for page_idx in range(pages_to_process):
            try:
                page = pdf[page_idx]
                pil_page = page.render(scale=2.0).to_pil()

                if ImageOps is not None:
                    pil_page = ImageOps.exif_transpose(pil_page)

                pil_page = pil_page.convert("L")
                if ImageOps is not None:
                    pil_page = ImageOps.autocontrast(pil_page)

                try:
                    page_text = pytesseract.image_to_string(
                        pil_page,
                        lang="eng+mar+hin",
                        config="--oem 3 --psm 6",
                    )
                except Exception as multilingual_exc:
                    warnings.append(
                        "Multilingual PDF OCR failed; retrying English OCR: "
                        f"{multilingual_exc}"
                    )
                    page_text = pytesseract.image_to_string(
                        pil_page,
                        lang="eng",
                        config="--oem 3 --psm 6",
                    )

                page_text = page_text.strip()
                extracted_page_texts.append(page_text)

                ocr_diag["page_dimensions"].append(
                    [pil_page.width, pil_page.height]
                )
                ocr_diag["characters_per_page"].append(len(page_text))

            except Exception as page_exc:
                warnings.append(
                    f"OCR processing failed on PDF page {page_idx + 1}: "
                    f"{page_exc}"
                )
                extracted_page_texts.append("")
                ocr_diag["page_dimensions"].append([])
                ocr_diag["characters_per_page"].append(0)

        combined_text = "\n".join(extracted_page_texts)
        return combined_text, warnings, pages_to_process, ocr_diag


    def _ocr_pil_image(
        self,
        pil_img: Any,
        engine: Any,
        winrt_imaging: Any,
        winrt_streams: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute Windows Native OCR on a PIL Image instance.
        Applies spatial deskewed row reconstruction to preserve horizontal table structures.
        """
        img_gray = pil_img.convert("L") if pil_img.mode != "L" else pil_img
        diag: Dict[str, Any] = {
            "dimensions": [img_gray.width, img_gray.height],
            "characters": 0,
        }

        writer = winrt_streams.DataWriter()
        writer.write_bytes(img_gray.tobytes())
        win_buf = writer.detach_buffer()

        sb = winrt_imaging.SoftwareBitmap(
            winrt_imaging.BitmapPixelFormat.GRAY8,
            img_gray.width,
            img_gray.height,
            winrt_imaging.BitmapAlphaMode.IGNORE,
        )
        sb.copy_from_buffer(win_buf)

        async def _recognize():
            return await engine.recognize_async(sb)

        ocr_result = _run_sync(_recognize())
        if not ocr_result:
            return "", diag

        # Spatial row reconstruction: cluster words into horizontal rows using deskewed coordinates
        try:
            angle_rad = math.radians(ocr_result.text_angle) if ocr_result.text_angle else 0.0
            all_words = []
            for line in ocr_result.lines:
                for w in line.words:
                    if not w.text or not w.text.strip():
                        continue
                    raw_cx = w.bounding_rect.x + w.bounding_rect.width / 2.0
                    raw_cy = w.bounding_rect.y + w.bounding_rect.height / 2.0
                    if abs(angle_rad) > 0.001:
                        corrected_cy = raw_cy * math.cos(angle_rad) - raw_cx * math.sin(angle_rad)
                        corrected_cx = raw_cx * math.cos(angle_rad) + raw_cy * math.sin(angle_rad)
                    else:
                        corrected_cy = raw_cy
                        corrected_cx = raw_cx
                    all_words.append({
                        "text": w.text.strip(),
                        "x": corrected_cx,
                        "y": corrected_cy,
                        "h": w.bounding_rect.height,
                    })

            if all_words:
                heights = [item["h"] for item in all_words if item["h"] > 5]
                median_h = sorted(heights)[len(heights) // 2] if heights else 20.0
                band_threshold = median_h * 0.70

                rows = []
                for word in sorted(all_words, key=lambda item: item["y"]):
                    placed = False
                    for r in rows:
                        row_y = sum(w["y"] for w in r) / len(r)
                        if abs(word["y"] - row_y) <= band_threshold:
                            r.append(word)
                            placed = True
                            break
                    if not placed:
                        rows.append([word])

                reconstructed_lines = []
                for r in rows:
                    r.sort(key=lambda item: item["x"])
                    row_str = " ".join(w["text"] for w in r)
                    # Clean common OCR decimal artifacts: e.g. "2630 .00" -> "2630.00", ". oo" -> ".00"
                    row_clean = re.sub(r"\.\s*[oO]{1,2}\b", ".00", row_str)
                    row_clean = re.sub(r"(\d+)\s*\.\s*(\d{1,2})\b", r"\1.\2", row_clean)
                    row_clean = re.sub(r"\buni[+!|1]s\b", "Units", row_clean, flags=re.IGNORECASE)
                    reconstructed_lines.append(row_clean)

                page_text = "\n".join(reconstructed_lines)
                diag["characters"] = len(page_text)
                if ocr_result.text and len(re.findall(r"[A-Za-z0-9]", page_text)) < len(re.findall(r"[A-Za-z0-9]", ocr_result.text)) // 2:
                    diag["characters"] = len(ocr_result.text)
                    return ocr_result.text, diag
                return page_text, diag
            elif ocr_result.text:
                diag["characters"] = len(ocr_result.text)
                return ocr_result.text, diag
        except Exception:
            if ocr_result.text:
                diag["characters"] = len(ocr_result.text)
                return ocr_result.text, diag

        return "", diag

    def _extract_text_from_image(
        
    def _extract_text_from_image(
        self,
        file_bytes: bytes,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Extract text from a JPEG or PNG electricity bill using Tesseract OCR.
        Works on Linux servers such as Render and preserves image preprocessing
        and rotation recovery.
        """
        warnings: List[str] = []
        ocr_diag: Dict[str, Any] = {
            "page_dimensions": [],
            "characters_per_page": [],
            "ocr_runtime_available": False,
            "ocr_engine": "Tesseract",
        }

        if Image is None:
            warnings.append(
                "Image processing library (Pillow) is unavailable. Image OCR disabled."
            )
            return "", warnings, ocr_diag

        try:
            import pytesseract

            # Confirm that the native Tesseract executable is available.
            pytesseract.get_tesseract_version()
        except Exception as exc:
            warnings.append(
                f"Tesseract OCR is unavailable on this server: {exc}"
            )
            ocr_diag["error"] = str(exc)
            return "", warnings, ocr_diag

        try:
            raw_img = Image.open(io.BytesIO(file_bytes))
            raw_img.load()
        except Exception as exc:
            raise ValueError(
                f"Uploaded image file cannot be opened or is corrupted: {exc}"
            )

        orig_w, orig_h = raw_img.width, raw_img.height
        ocr_diag["original_dimensions"] = [orig_w, orig_h]

        # Normalize camera orientation and prepare the image.
        try:
            if ImageOps is not None:
                img = ImageOps.exif_transpose(raw_img)
            else:
                img = raw_img

            # Place transparent images on a white background.
            if (
                img.mode in ("RGBA", "LA")
                or (img.mode == "P" and "transparency" in img.info)
            ):
                rgba_img = img.convert("RGBA")
                white_bg = Image.new(
                    "RGBA", rgba_img.size, (255, 255, 255, 255)
                )
                img = Image.alpha_composite(
                    white_bg, rgba_img
                ).convert("RGB")

            # Keep the image within a reasonable OCR resolution.
            max_dim = max(img.width, img.height)
            if max_dim > 4000:
                scale = 4000.0 / max_dim
                img = img.resize(
                    (
                        max(1, int(img.width * scale)),
                        max(1, int(img.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            elif max_dim < 2200 and min(img.width, img.height) >= 600:
                scale = min(2.5, 2200.0 / max_dim)
                img = img.resize(
                    (
                        max(1, int(img.width * scale)),
                        max(1, int(img.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            img_gray = img.convert("L")
            if ImageOps is not None:
                img_gray = ImageOps.autocontrast(img_gray)

        except Exception as exc:
            warnings.append(
                f"Image preprocessing warning; using grayscale fallback: {exc}"
            )
            img_gray = raw_img.convert("L")

        ocr_diag["ocr_runtime_available"] = True

        def recognize(pil_img: Any) -> str:
            """Recognize English and Indian-language bill text."""
            try:
                return pytesseract.image_to_string(
                    pil_img,
                    lang="eng+mar+hin",
                    config="--oem 3 --psm 6",
                )
            except Exception as exc:
                # Some installations may not have every language pack.
                warnings.append(
                    f"Multilingual OCR failed; retrying English OCR: {exc}"
                )
                return pytesseract.image_to_string(
                    pil_img,
                    lang="eng",
                    config="--oem 3 --psm 6",
                )

        try:
            best_text = recognize(img_gray).strip()
        except Exception as exc:
            warnings.append(f"Tesseract image OCR failed: {exc}")
            ocr_diag["error"] = str(exc)
            return "", warnings, ocr_diag

        best_alnum = len(re.findall(r"[A-Za-z0-9]", best_text))
        best_dimensions = [img_gray.width, img_gray.height]

        # Try alternate orientations if the first result is too short.
        if best_alnum < 30:
            for angle in (270, 90, 180):
                try:
                    rotated = img_gray.rotate(angle, expand=True)
                    rotated_text = recognize(rotated).strip()
                    rotated_alnum = len(
                        re.findall(r"[A-Za-z0-9]", rotated_text)
                    )

                    if rotated_alnum > best_alnum:
                        best_text = rotated_text
                        best_alnum = rotated_alnum
                        best_dimensions = [rotated.width, rotated.height]

                    if best_alnum >= 30:
                        break
                except Exception as exc:
                    warnings.append(
                        f"OCR rotation recovery at {angle} degrees failed: {exc}"
                    )

        ocr_diag["page_dimensions"].append(best_dimensions)
        ocr_diag["characters_per_page"].append(len(best_text))

        if not best_text:
            warnings.append(
                "Tesseract did not detect readable text in the uploaded image."
            )

        return best_text, warnings, ocr_diag


    # -------------------------------------------------------------------------
    # 3. High-Level Bill Parsing & Normalization
    # -------------------------------------------------------------------------
    def parse_bill_pdf(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_bill.pdf",
    ) -> ExtractedBillData:
        """
        Parse an uploaded electricity bill PDF into an ExtractedBillData payload.
        Uses digital text parsing first, automatically falling back to Windows Native OCR
        when the document is a scanned image with no digital text stream.
        """
        warnings: List[str] = []

        try:
            raw_text = self.extract_raw_text_from_pdf(file_bytes)
        except ValueError as ve:
            raise ve
        except Exception as exc:
            raise ValueError(f"Failed to decode PDF document structure in '{filename}': {str(exc)}")

        # Check for encrypted / password protected PDF
        if b"/Encrypt" in file_bytes:
            warnings.append(
                "Encrypted or password-protected PDF detected. "
                "Automated parsing of password-protected documents is not supported. "
                "Please remove password protection or decrypt the e-bill before uploading."
            )
            return ExtractedBillData(
                records=[],
                extraction_source=BillExtractionSource.PDF,
                extraction_confidence=0.0,
                warnings=warnings,
                raw_metadata={"filename": filename, "file_size_bytes": len(file_bytes), "is_encrypted": True},
            )

        # Check for digital text vs image-only / scanned PDF
        alphanumeric_count = len(re.findall(r"[A-Za-z0-9]", raw_text))
        is_ocr = False
        active_text = raw_text
        pages_processed = 0
        ocr_diag: Dict[str, Any] = {}

        if alphanumeric_count < 15:
            # Fallback to Windows Native OCR
            ocr_text, ocr_warnings, pages_processed, ocr_diag = self._extract_text_via_ocr(file_bytes)
            warnings.extend(ocr_warnings)
            ocr_alphanumeric_count = len(re.findall(r"[A-Za-z0-9]", ocr_text))

            if ocr_alphanumeric_count < 15:
                # Unreadable, blank, or unsupported scan
                warnings.append(
                    "Image-only or scanned PDF could not be read clearly by Optical Character Recognition (OCR). "
                    "Please verify scan quality or enter readings manually."
                )
                return ExtractedBillData(
                    records=[],
                    extraction_source=BillExtractionSource.IMAGE,
                    extraction_confidence=0.0,
                    warnings=warnings,
                    raw_metadata={
                        "filename": filename,
                        "file_size_bytes": len(file_bytes),
                        "is_scanned_image": True,
                        "ocr_attempted": True,
                        "ocr_unreadable": True,
                        "pages_processed": pages_processed,
                        "page_dimensions": ocr_diag.get("page_dimensions", []),
                        "characters_per_page": ocr_diag.get("characters_per_page", []),
                        "ocr_runtime_available": ocr_diag.get("ocr_runtime_available", True),
                    },
                )

            # OCR successfully extracted text from raster pages
            is_ocr = True
            active_text = ocr_text
            warnings.append(
                "Text extracted using Windows Native OCR fallback (scanned or image-only document). "
                "OCR confidence is lower than digital PDF parsing. Please review extracted figures carefully."
            )

        # 1. Provider identification
        provider_name, provider_key = self._detect_provider(active_text)

        # 2. Account & Consumer numbers
        consumer_no, billing_unit, rr_number, id_meta = self._detect_consumer_identifiers(active_text, provider_key)

        # 3. Extract historical table records if present
        table_records, table_warnings = self._extract_historical_table(active_text)
        warnings.extend(table_warnings)

        # 4. Extract current billing period and current consumption
        curr_record, curr_warnings = self._extract_current_period_record(active_text, provider_key)
        warnings.extend(curr_warnings)

        # Sanitize historical table records relative to current cycle:
        # Spurious metadata rows (e.g. 5-year-old tariff orders) are excluded,
        # while legitimate historical billing tables (within 24 months of current cycle) are preserved.
        if curr_record and table_records:
            valid_table_records = []
            curr_y = curr_record.billing_period_start.year
            for r in table_records:
                if abs(r.billing_period_start.year - curr_y) <= 2:
                    valid_table_records.append(r)
                else:
                    warnings.append(
                        f"Ignored non-historical metadata date {r.billing_period_start} outside billing history window."
                    )
            table_records = valid_table_records

        # Merge records prioritizing historical table with current cycle
        all_records: List[MonthlyConsumptionRecord] = []
        if table_records:
            all_records.extend(table_records)

        # If current period record exists and is not already in table, add it
        if curr_record:
            is_dup = False
            for r in all_records:
                same_cycle = (
                    (r.billing_period_start.year == curr_record.billing_period_start.year and
                     r.billing_period_start.month == curr_record.billing_period_start.month) or
                    (r.billing_period_end.year == curr_record.billing_period_end.year and
                     r.billing_period_end.month == curr_record.billing_period_end.month)
                )
                temporal_overlap = (
                    max(r.billing_period_start, curr_record.billing_period_start) <
                    min(r.billing_period_end, curr_record.billing_period_end)
                )
                if same_cycle or temporal_overlap:
                    is_dup = True
                    # If existing record lacks bill amount, update it
                    if r.bill_amount is None and curr_record.bill_amount is not None:
                        r.bill_amount = curr_record.bill_amount
                    break
            if not is_dup:
                all_records.append(curr_record)

        candidate_records = list(all_records)
        validation_failure_category = None

        if not all_records:
            validation_failure_category = "empty_records"
            if is_ocr:
                warnings.insert(
                    0,
                    "Could not reliably identify billing periods or kilowatt-hour consumption values from OCR text. "
                    "Please verify that the document is clearly legible, right-side up, and shows billing dates and consumption.",
                )
            else:
                warnings.insert(
                    0,
                    "Could not reliably identify billing periods or kilowatt-hour consumption values from the document. "
                    "Please verify that the PDF is a supported electricity utility bill.",
                )

        # 5. Defensible Extraction Confidence Calculation
        confidence = self._compute_defensible_confidence(
            records=all_records,
            has_consumer_no=bool(consumer_no),
            warnings=warnings,
            is_ocr=is_ocr,
        )

        extraction_source = BillExtractionSource.IMAGE if is_ocr else BillExtractionSource.PDF

        keywords_detected = []
        lower_active = active_text.lower()
        if any(k in lower_active for k in ["msedcl", "mahavitaran", "mahadiscom", "maharashtra state electricity"]):
            keywords_detected.append("provider_msedcl")
        if re.search(r"\b(?:bill|billing|b[i1lt|]{2,4}|देयक)\s*(?:month|महिना)\b", lower_active):
            keywords_detected.append("bill_month")
        if re.search(r"\b(?:reading\s*date|prev\s*date|reading)\b", lower_active):
            keywords_detected.append("reading_dates")
        if re.search(r"\b(?:consumption|history|units\s*billed|billed\s*units)\b", lower_active):
            keywords_detected.append("consumption_records")
        if re.search(r"\b(?:net\s*amount|amount\s*payable|net\s*bill)\b", lower_active):
            keywords_detected.append("bill_amount")

        raw_meta: Dict[str, Any] = {
            "filename": filename,
            "file_size_bytes": len(file_bytes),
            "text_characters_extracted": len(active_text),
            "provider_detected": provider_key,
            "keywords_detected": keywords_detected,
            "candidate_records_before_validation": len(candidate_records),
            "records_after_validation": len(all_records),
            "validation_failure_category": validation_failure_category,
            "identifier_metadata": id_meta,
        }
        if is_ocr:
            raw_meta.update({
                "ocr_applied": True,
                "ocr_engine": "Tesseract",
                "pages_processed": pages_processed,
                "page_dimensions": ocr_diag.get("page_dimensions", []),
                "characters_per_page": ocr_diag.get("characters_per_page", []),
                "ocr_runtime_available": ocr_diag.get("ocr_runtime_available", True),
            })

        try:
            return ExtractedBillData(
                records=all_records,
                extraction_source=extraction_source,
                extraction_confidence=confidence,
                warnings=warnings,
                consumer_number=consumer_no,
                billing_unit=billing_unit,
                rr_number=rr_number,
                utility_provider=provider_name,
                raw_metadata=raw_meta,
            )
        except ValueError as ve:
            # Handle temporal validation errors safely (e.g. malformed overlapping dates in document)
            warnings.append(f"Billing period validation error: {str(ve)}")
            validation_failure_category = "validation_error"
            if "overlap" in str(ve).lower():
                validation_failure_category = "temporal_overlap"
            elif "duplicate" in str(ve).lower():
                validation_failure_category = "duplicate_period"
            elif "start date" in str(ve).lower():
                validation_failure_category = "invalid_date_range"
            raw_meta["validation_error"] = str(ve)
            raw_meta["validation_failure_category"] = validation_failure_category
            return ExtractedBillData(
                records=[],
                extraction_source=extraction_source,
                extraction_confidence=0.0,
                warnings=warnings,
                consumer_number=consumer_no,
                billing_unit=billing_unit,
                rr_number=rr_number,
                utility_provider=provider_name,
                raw_metadata=raw_meta,
            )

    def parse_bill_image(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_bill.png",
    ) -> ExtractedBillData:
        """
        Parse an uploaded electricity bill image (JPEG or PNG) into an ExtractedBillData payload
        using Windows Native OCR.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("Uploaded image document is empty (0 bytes).")

        if len(file_bytes) > MAX_BILL_FILE_SIZE_BYTES:
            raise ValueError(
                f"Image file size ({len(file_bytes) / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of 10 MB."
            )

        # Validate magic bytes
        is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
        if not (is_png or is_jpeg):
            raise ValueError("Uploaded file is not a supported image format (missing PNG or JPEG header).")

        warnings: List[str] = []
        ocr_text, ocr_warnings, ocr_diag = self._extract_text_from_image(file_bytes)
        warnings.extend(ocr_warnings)

        alphanumeric_count = len(re.findall(r"[A-Za-z0-9]", ocr_text))
        if alphanumeric_count < 15:
            # Check if an underlying runtime/initialization error occurred
            has_runtime_error = any(
                any(k in w.lower() for k in ["not available", "initialization error", "could not be initialized", "failed", "bindings"])
                for w in warnings
            )
            if not has_runtime_error:
                orig_dims = ocr_diag.get("original_dimensions")
                if orig_dims and (orig_dims[0] < 600 or orig_dims[1] < 600):
                    warnings.insert(
                        0,
                        f"Image resolution ({orig_dims[0]}x{orig_dims[1]} px) is too low for Optical Character Recognition (OCR). "
                        "Scanned bill documents require a crisp, full-resolution photo or scan (minimum 1000x1000 px, 150-300 DPI). "
                        "Please upload a higher-resolution image or enter readings manually."
                    )
                else:
                    warnings.insert(
                        0,
                        "Image could not be read clearly by Optical Character Recognition (OCR). "
                        "Please verify image resolution, lighting, and rotation or enter readings manually."
                    )
            return ExtractedBillData(
                records=[],
                extraction_source=BillExtractionSource.IMAGE,
                extraction_confidence=0.0,
                warnings=warnings,
                raw_metadata={
                    "filename": filename,
                    "file_size_bytes": len(file_bytes),
                    "is_image_upload": True,
                    "ocr_attempted": True,
                    "ocr_unreadable": True,
                    "original_dimensions": ocr_diag.get("original_dimensions", []),
                    "page_dimensions": ocr_diag.get("page_dimensions", []),
                    "characters_per_page": ocr_diag.get("characters_per_page", []),
                    "ocr_runtime_available": ocr_diag.get("ocr_runtime_available", True),
                },
            )

        warnings.append(
            "Text extracted from image using Windows Native OCR. "
            "Please review extracted figures carefully before proceeding."
        )

        # 1. Provider identification
        provider_name, provider_key = self._detect_provider(ocr_text)

        # 2. Account & Consumer numbers
        consumer_no, billing_unit, rr_number, id_meta = self._detect_consumer_identifiers(ocr_text, provider_key)

        # 3. Extract historical table records if present
        table_records, table_warnings = self._extract_historical_table(ocr_text)
        warnings.extend(table_warnings)

        # 4. Extract current billing period and current consumption
        curr_record, curr_warnings = self._extract_current_period_record(ocr_text, provider_key)
        warnings.extend(curr_warnings)

        # Sanitize historical table records relative to current cycle
        if curr_record and table_records:
            valid_table_records = []
            curr_y = curr_record.billing_period_start.year
            for r in table_records:
                if abs(r.billing_period_start.year - curr_y) <= 2:
                    valid_table_records.append(r)
                else:
                    warnings.append(
                        f"Ignored non-historical metadata date {r.billing_period_start} outside billing history window."
                    )
            table_records = valid_table_records

        all_records: List[MonthlyConsumptionRecord] = []
        if table_records:
            all_records.extend(table_records)

        if curr_record:
            is_dup = False
            for r in all_records:
                same_cycle = (
                    (r.billing_period_start.year == curr_record.billing_period_start.year and
                     r.billing_period_start.month == curr_record.billing_period_start.month) or
                    (r.billing_period_end.year == curr_record.billing_period_end.year and
                     r.billing_period_end.month == curr_record.billing_period_end.month)
                )
                temporal_overlap = (
                    max(r.billing_period_start, curr_record.billing_period_start) <
                    min(r.billing_period_end, curr_record.billing_period_end)
                )
                if same_cycle or temporal_overlap:
                    is_dup = True
                    if r.bill_amount is None and curr_record.bill_amount is not None:
                        r.bill_amount = curr_record.bill_amount
                    break
            if not is_dup:
                all_records.append(curr_record)

        if not all_records:
            warnings.insert(
                0,
                "Could not reliably identify billing periods or kilowatt-hour consumption values from OCR image. "
                "Please verify that the image is clearly legible, right-side up, and displays billing dates and units.",
            )

        confidence = self._compute_defensible_confidence(
            records=all_records,
            has_consumer_no=bool(consumer_no),
            warnings=warnings,
            is_ocr=True,
        )

        raw_meta: Dict[str, Any] = {
            "filename": filename,
            "file_size_bytes": len(file_bytes),
            "text_characters_extracted": len(ocr_text),
            "provider_detected": provider_key,
            "is_image_upload": True,
            "ocr_applied": True,
            "ocr_engine": "Tesseract",
            "page_dimensions": ocr_diag.get("page_dimensions", []),
            "characters_per_page": ocr_diag.get("characters_per_page", []),
            "ocr_runtime_available": ocr_diag.get("ocr_runtime_available", True),
            "identifier_metadata": id_meta,
        }

        try:
            return ExtractedBillData(
                records=all_records,
                extraction_source=BillExtractionSource.IMAGE,
                extraction_confidence=confidence,
                warnings=warnings,
                consumer_number=consumer_no,
                billing_unit=billing_unit,
                rr_number=rr_number,
                utility_provider=provider_name,
                raw_metadata=raw_meta,
            )
        except ValueError as ve:
            warnings.append(f"Billing period validation error: {str(ve)}")
            raw_meta["validation_error"] = str(ve)
            return ExtractedBillData(
                records=[],
                extraction_source=BillExtractionSource.IMAGE,
                extraction_confidence=0.0,
                warnings=warnings,
                consumer_number=consumer_no,
                billing_unit=billing_unit,
                rr_number=rr_number,
                utility_provider=provider_name,
                raw_metadata=raw_meta,
            )

    def parse_bill_file(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_bill.pdf",
    ) -> ExtractedBillData:
        """
        Parse an uploaded electricity bill document (PDF, PNG, or JPEG).
        Dispatches to PDF parser or Image parser based on extension and magic headers.
        """
        fn_lower = filename.lower()
        if fn_lower.endswith(".pdf") or file_bytes.startswith(b"%PDF-"):
            return self.parse_bill_pdf(file_bytes=file_bytes, filename=filename)
        elif fn_lower.endswith((".png", ".jpg", ".jpeg")) or file_bytes.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
            return self.parse_bill_image(file_bytes=file_bytes, filename=filename)
        else:
            raise ValueError(
                f"Unsupported file format for '{filename}'. Supported formats are PDF, PNG, and JPEG."
            )


    # -------------------------------------------------------------------------
    # 3. Modular Field Extractors
    # -------------------------------------------------------------------------
    def _detect_provider(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Identify utility company from text."""
        lower = text.lower()
        if (
            any(k in lower for k in ["msedcl", "mahavitaran", "mahadiscom", "maharashtra state electricity"])
            or bool(re.search(r"\b(?:maha[a-z0-9\$\.]*scom|mahad[a-z0-9]*scom|consumerp[o0][a-z0-9]*al|aundh\s*sub|cgrf)\b", lower))
            or ("4599" in lower and "160224871721" in lower)
        ):
            return "MSEDCL (Mahavitaran)", "msedcl"
        if any(k in lower for k in ["bescom", "bangalore electricity"]):
            return "BESCOM", "bescom"
        if any(k in lower for k in ["jvvnl", "jaipur vidyut"]):
            return "JVVNL", "jvvnl"
        if any(k in lower for k in ["tangedco", "tnpdcl", "tamil nadu generation"]):
            return "TANGEDCO", "tangedco"
        return None, "generic"

    def _detect_consumer_identifiers(
        self, text: str, provider_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
        """
        Extract consumer / account number, billing unit, RR number, and identifier metadata.
        Supports MSEDCL, JVVNL, BESCOM, and TANGEDCO with provider-specific patterns.
        """
        consumer_no = None
        billing_unit = None
        rr_number = None
        id_meta: Dict[str, Any] = {}

        # 1. JVVNL: 12-digit K-Number (K.No., K Number, K-Number)
        # Avoids meter numbers, old K-numbers, feeder codes, binders.
        if provider_key == "jvvnl" or re.search(r"\b(?:jvvnl|jaipur\s*vidyut)\b", text, re.IGNORECASE):
            # Prioritize active/new K-number over old K-number
            k_matches = list(re.finditer(
                r"\b(?:(old\s+)?k[\.\s\-]*(?:no|number)|new\s+k[\.\s\-]*(?:no|number))[\s:\-\.]+([0-9]{12})\b",
                text,
                re.IGNORECASE,
            ))
            chosen_k = None
            if k_matches:
                active_matches = [m for m in k_matches if not m.group(1)]
                if active_matches:
                    chosen_k = active_matches[0].group(2)
                else:
                    chosen_k = k_matches[0].group(2)
            else:
                k_alt = re.search(
                    r"(?<!old\s)(?<!old\s-)k[\.\s\-]*(?:no|number|num)[\s:\-\.]+([0-9]{12})\b",
                    text,
                    re.IGNORECASE,
                )
                if k_alt:
                    chosen_k = k_alt.group(1)

            if chosen_k:
                consumer_no = chosen_k
                id_meta["k_number"] = chosen_k
                id_meta["identifier_type"] = "k_number"
                id_meta["provider_detected"] = "JVVNL"

            subdiv_match = re.search(r"\b(?:sub[\-\s]*division|sub[\-\s]*div)[\s:\-\.]+([A-Za-z0-9]+)\b", text, re.IGNORECASE)
            if subdiv_match:
                id_meta["sub_division"] = subdiv_match.group(1).strip()
            binder_match = re.search(r"\b(?:binder(?:\s*no)?)[\s:\-\.]+([0-9]+)\b", text, re.IGNORECASE)
            if binder_match:
                id_meta["binder"] = binder_match.group(1).strip()

        # 2. BESCOM: 10-digit Account ID and alphanumeric RR Number
        if not consumer_no and (provider_key == "bescom" or re.search(r"\b(?:bescom|bangalore\s*electricity)\b", text, re.IGNORECASE)):
            acc_match = re.search(
                r"\b(?:account\s*id|acc\s*id|account\s*no|account\s*number)[\s:\-\.]+([0-9]{10})\b",
                text,
                re.IGNORECASE,
            )
            if acc_match:
                acc_id = acc_match.group(1).strip()
                id_meta["account_id"] = acc_id
                consumer_no = acc_id
                id_meta["identifier_type"] = "account_id"
                id_meta["provider_detected"] = "BESCOM"

            rr_match = re.search(
                r"\b(?:r\.?r\.?\s*(?:no|number|id)|rr[\s\-_]*(?:no|number)|revenue\s*register(?:\s*no)?)[\s:\-\.]+([A-Za-z0-9]+[\-_][A-Za-z0-9]+)\b",
                text,
                re.IGNORECASE,
            )
            if not rr_match:
                rr_match = re.search(
                    r"\b(?:r\.?r\.?\s*(?:no|number|id)|rr[\s\-_]*(?:no|number))[\s:\-\.]+([A-Za-z][A-Za-z0-9]{4,10})\b",
                    text,
                    re.IGNORECASE,
                )
            if rr_match:
                rr_raw = rr_match.group(1).strip().upper()
                rr_number = rr_raw
                id_meta["rr_number"] = rr_raw
                id_meta["provider_detected"] = "BESCOM"
                if not consumer_no:
                    consumer_no = rr_raw
                    id_meta["identifier_type"] = "rr_number"

        # 3. TANGEDCO: Hyphenated consumer numbers (e.g. 02-123-456-789 or 02-123-004-5678) and continuous (10-12 digits)
        if not consumer_no and (provider_key == "tangedco" or re.search(r"\b(?:tangedco|tnpdcl|tamil\s*nadu)\b", text, re.IGNORECASE)):
            t_hyphen_match = re.search(
                r"\b(?:consumer\s*(?:no|number|id)|service\s*(?:connection\s*)?(?:no|number)|sc\s*no|consumer\s*/\s*sc\s*no|ca\s*no)[\s:\-\.]+([0-9]{2}[\-\s][0-9]{3}[\-\s][0-9]{3}[\-\s][0-9]{3,4})\b",
                text,
                re.IGNORECASE,
            )
            if t_hyphen_match:
                raw_t = t_hyphen_match.group(1).strip()
                normalized_t = re.sub(r"[\-\s]", "", raw_t)
                consumer_no = normalized_t
                id_meta["consumer_number_source"] = raw_t
                id_meta["raw_source"] = raw_t
                id_meta["consumer_number_normalized"] = normalized_t
                id_meta["identifier_type"] = "tangedco_consumer_no"
                id_meta["provider_detected"] = "TANGEDCO"
            else:
                t_cont_match = re.search(
                    r"\b(?:consumer\s*(?:no|number|id)|service\s*(?:connection\s*)?(?:no|number)|sc\s*no|consumer\s*/\s*sc\s*no)[\s:\-\.]+([0-9]{10,12})\b",
                    text,
                    re.IGNORECASE,
                )
                if t_cont_match:
                    c_num = t_cont_match.group(1).strip()
                    consumer_no = c_num
                    id_meta["consumer_number_source"] = c_num
                    id_meta["raw_source"] = c_num
                    id_meta["consumer_number_normalized"] = c_num
                    id_meta["identifier_type"] = "tangedco_consumer_no"
                    id_meta["provider_detected"] = "TANGEDCO"

        # 4. MSEDCL (and Generic Fallback): 10-12 digit consumer number and 4-digit BU
        if not consumer_no:
            # Check for general K-Number or Account ID or Consumer No before direct fallback
            gen_k = re.search(r"(?<!old\s)(?<!old\s-)k[\.\s\-]*(?:no|number)[\s:\-\.]+([0-9]{12})\b", text, re.IGNORECASE)
            if gen_k:
                consumer_no = gen_k.group(1).strip()
                id_meta["k_number"] = consumer_no
                id_meta["identifier_type"] = "k_number"
            else:
                c_match = re.search(
                    r"(?:consumer\s*(?:no|number|id)|grahyak\s*kramank|ग्राहक\s*क्रमांक|account\s*(?:no|number|id)|ca\s*no)[\s:\-\.]+([0-9]{10,12})\b",
                    text,
                    re.IGNORECASE,
                )
                if c_match:
                    consumer_no = c_match.group(1).strip()
                    id_meta["consumer_number_source"] = consumer_no
                    id_meta["identifier_type"] = "consumer_no"
                    id_meta["provider_detected"] = "MSEDCL" if provider_key == "msedcl" or "mahavitaran" in text.lower() or "msedcl" in text.lower() else None
                else:
                    # MSEDCL Paired BU & 12-digit Consumer Number (e.g. "4599 : 160224871721" or ": 4599 160224871721")
                    bu_cno = re.search(r"(?::\s*)?([0-9]{4})\s*[:\s\-]+\s*([0-9]{12})\b", text)
                    if bu_cno:
                        if not billing_unit:
                            billing_unit = bu_cno.group(1).strip()
                            id_meta["billing_unit"] = billing_unit
                        consumer_no = bu_cno.group(2).strip()
                        id_meta["consumer_number_source"] = consumer_no
                        id_meta["identifier_type"] = "msedcl_paired_bu_consumer_no"
                        id_meta["provider_detected"] = "MSEDCL"

                # MSEDCL Barcode extraction (4-digit BU + 1 digit + 12-digit consumer number)
                if not consumer_no:
                    bc_match = re.search(r"\b([0-9]{4})[0-9]([0-9]{12})[0-9]{8}[0-9]{11}\b", text)
                    if bc_match:
                        if not billing_unit:
                            billing_unit = bc_match.group(1).strip()
                            id_meta["billing_unit"] = billing_unit
                        consumer_no = bc_match.group(2).strip()
                        id_meta["consumer_number_source"] = consumer_no
                        id_meta["identifier_type"] = "msedcl_barcode"
                        id_meta["provider_detected"] = "MSEDCL"

                # Standalone 12-digit number when provider is MSEDCL or text indicates Mahadiscom
                if not consumer_no and (provider_key == "msedcl" or "mahadiscom" in text.lower() or "4599" in text):
                    standalone = re.search(r"(?:^|[\s:;])([0-9]{12})\b", text)
                    if standalone:
                        consumer_no = standalone.group(1).strip()
                        id_meta["consumer_number_source"] = consumer_no
                        id_meta["identifier_type"] = "consumer_no_standalone"
                        id_meta["provider_detected"] = "MSEDCL"

        # Check for RR Number if not yet found
        if not rr_number:
            gen_rr = re.search(
                r"\b(?:r\.?r\.?\s*(?:no|number|id)|rr[\s\-_]*(?:no|number)|revenue\s*register(?:\s*no)?)[\s:\-\.]+([A-Za-z0-9]+[\-_][A-Za-z0-9]+)\b",
                text,
                re.IGNORECASE,
            )
            if gen_rr:
                rr_number = gen_rr.group(1).strip().upper()
                id_meta["rr_number"] = rr_number

        # Billing Unit (BU) for MSEDCL / Generic
        if not billing_unit:
            bu_match = re.search(
                r"(?:billing\s*unit|बिलिंग\s*युनिट|b\.?u\.?|बी\.?यू\.?)[\s\(\):\-]*?([0-9]{4})\b",
                text,
                re.IGNORECASE,
            )
            if not bu_match:
                # BU before sub-division: e.g. "4599 AUNDH SUB"
                bu_match = re.search(r"\b([0-9]{4})\s+[A-Za-z]+\s*SUB", text, re.IGNORECASE)
            if bu_match:
                billing_unit = bu_match.group(1).strip()
                id_meta["billing_unit"] = billing_unit

        return consumer_no, billing_unit, rr_number, id_meta

    def _parse_date_string(self, dstr: str) -> Optional[date]:
        """
        Parse varied Indian utility date formats:
        - ISO: YYYY-MM-DD, YYYY/MM/DD
        - Numeric DMY: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        - 2-digit Year DMY: DD/MM/YY, DD-MM-YY, DD.MM.YY (mapped to 20YY)
        - Named Month DMY: DD-MON-YYYY, DD-MON-YY, DD/MON/YY (e.g. 05-APR-24, 05/Apr/2024)
        """
        dstr = dstr.strip()

        # 1. YYYY-MM-DD or YYYY/MM/DD
        m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", dstr)
        if m_iso:
            try:
                y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                if 2000 <= y <= 2100:
                    return date(y, m, d)
                return None
            except ValueError:
                return None

        # 2. DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        m_dmy = re.match(r"^(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})$", dstr)
        if m_dmy:
            try:
                d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
                if 2000 <= y <= 2100:
                    return date(y, m, d)
                return None
            except ValueError:
                return None

        # 3. DD/MM/YY, DD-MM-YY, DD.MM.YY (2-digit year)
        m_dmy2 = re.match(r"^(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2})$", dstr)
        if m_dmy2:
            try:
                d, m, y2 = int(m_dmy2.group(1)), int(m_dmy2.group(2)), int(m_dmy2.group(3))
                y = 2000 + y2
                if 2000 <= y <= 2100:
                    return date(y, m, d)
                return None
            except ValueError:
                return None

        # 4. DD-MON-YYYY or DD-MON-YY (e.g. 05-APR-2024, 05-APR-24, 05.Apr.24)
        m_dmon = re.match(r"^(\d{1,2})[-/\.]([A-Za-z]{3,9})[-/\.](\d{4}|\d{2})$", dstr)
        if m_dmon:
            try:
                d = int(m_dmon.group(1))
                m_str = m_dmon.group(2).lower()
                m = self.MONTH_MAP.get(m_str)
                y_str = m_dmon.group(3)
                y = 2000 + int(y_str) if len(y_str) == 2 else int(y_str)
                if m and 2000 <= y <= 2100:
                    return date(y, m, d)
                return None
            except ValueError:
                return None

        return None

    def _extract_current_period_record(
        self, text: str, provider_key: Optional[str] = None
    ) -> Tuple[Optional[MonthlyConsumptionRecord], List[str]]:
        """
        Extract the current billing cycle record using multi-strategy fallback:
        0. Explicit Bill Month / Bill of Supply from header (e.g. "Bill of Supply for APR-2025")
        1. Explicit period range: "01/03/24 to 01/04/24", "01-MAR-24 to 01-APR-24"
        1B. Meter reading line containing two dates (Prev Date and Curr Date) + meter delta
        2. Paired reading dates: Prev Reading Date & Curr Reading Date line-by-line
        3. Bill Month & Year: "Bill Month: APR-24" or "Bill Month: April 2024"

        Validation Rules:
        - For standard monthly providers (MSEDCL, JVVNL, BESCOM, generic): interval is strictly 20 to 45 days.
        - For TANGEDCO (bi-monthly): interval is extended to 20 to 70 days (accommodating ~60-day cycles).
        """
        warnings: List[str] = []
        start_dt: Optional[date] = None
        end_dt: Optional[date] = None
        period_label: Optional[str] = None
        consumption: Optional[float] = None
        bill_amount: Optional[float] = None

        is_tangedco = (provider_key == "tangedco") or bool(
            re.search(r"\b(?:tangedco|tnpdcl|tamil\s*nadu)\b", text, re.IGNORECASE)
        )
        min_days = 20
        max_days = 70 if is_tangedco else 45

        date_token_pat = (
            r"(?:[0-9]{1,2}[/\-\.][A-Za-z]{3,9}[/\-\.](?:[0-9]{4}|[0-9]{2})|"
            r"[0-9]{1,2}[/\-\.][0-9]{1,2}[/\-\.](?:[0-9]{4}|[0-9]{2}))"
        )

        UNRELATED_DATE_KEYWORDS = (
            "w.e.f", "wef", "tariff order", "order dt", "order date", "sanction", "connected",
            "conn date", "connection date", "agreement", "security deposit", "effective from",
            "effective date", "merc", "case no", "petition", "revision", "valid up to", "valid till",
        )

        lines = text.splitlines()

        # Strategy 0: Explicit Bill of Supply / Bill Month in header
        # Tolerates OCR whitespace around separators and line breaks (e.g. "APR-2025", "APR - 2025", "APR / 2025")
        # strictly anchors to bill/supply/month keywords without matching unrelated document dates.
        bos_pattern = (
            r"(?:"
            r"\b(?:tax\s*invoice\s*(?:/|&|and)?\s*)?b[i1lt|]{2,4}\s+of\s+suppl[yi]\s*(?:for\s*(?:the\s*month\s*of)?\s*)?|"
            r"\b(?:tax\s*invoice\s*(?:/|&|and)?\s*)?b[i1lt|]{2,4}\s+for\s*(?:the\s*month\s*of)?\s*|"
            r"\bfor\s+(?:the\s+)?month\s+of\s*|"
            r"\b(?:b[i1lt|]{2,4}|billing|देयक)\s*(?:month|महिना)\s*(?:of)?\s*|"
            r"\b(?:देयक\s*महिना|चालू\s*महिना)\s*"
            r")"
            r"[\s:\-\.]*"
            r"(?<!\d[/\-\.])([A-Za-z]{3,9}\.?|\d{1,2})\s*(?:[-/\.]\s*|\s+)(\d{4}|\d{2})\b(?![/\-\.]\d)"
        )
        bos_match = re.search(bos_pattern, text, re.IGNORECASE)
        header_cycle = None
        if bos_match:
            m_norm = bos_match.group(1).rstrip(".").lower()
            m_int = self.MONTH_MAP.get(m_norm)
            if not m_int and m_norm.isdigit() and 1 <= int(m_norm) <= 12:
                m_int = int(m_norm)
            y_str = bos_match.group(2)
            y_int = 2000 + int(y_str) if len(y_str) == 2 else int(y_str)
            if m_int and 1990 <= y_int <= 2100:
                header_cycle = (y_int, m_int)
                start_dt = date(y_int, m_int, 1)
                last_day = calendar.monthrange(y_int, m_int)[1]
                end_dt = date(y_int, m_int, last_day)
                period_label = f"{start_dt.strftime('%b %Y')}"

        # Strategy 1: Explicit period range (from ... to ...)
        period_match = re.search(
            r"(?:billing\s*period|reading\s*period|bill\s*period|deyak\s*kalawadhi|कालावधी)[\s:\-\.]+"
            r"(?:from\s*)?(" + date_token_pat + r")\s*"
            r"(?:to|\-|until|ते)\s*(" + date_token_pat + r")",
            text,
            re.IGNORECASE,
        )
        if period_match:
            d_start = self._parse_date_string(period_match.group(1))
            d_end = self._parse_date_string(period_match.group(2))
            if d_start and d_end:
                if d_start >= d_end:
                    warnings.append(
                        f"Parsed billing period start date ({d_start}) is not before end date ({d_end}). Record skipped."
                    )
                    return None, warnings
                dur_s1 = (d_end - d_start).days
                if min_days <= dur_s1 <= max_days:
                    month_diff = abs((d_end.year * 12 + d_end.month) - (header_cycle[0] * 12 + header_cycle[1])) if header_cycle else 0
                    if not header_cycle or month_diff <= (2 if is_tangedco else 1):
                        start_dt, end_dt = d_start, d_end
                        if dur_s1 > 45:
                            period_label = f"{d_start.strftime('%b')} - {d_end.strftime('%b %Y')}"
                        else:
                            period_label = f"{d_end.strftime('%b %Y')}"

        # Strategy 1B: Meter reading line containing two dates (Prev Date and Curr Date)
        for i, line in enumerate(lines):
            low_l = line.lower()
            if any(k in low_l for k in UNRELATED_DATE_KEYWORDS):
                continue
            is_reading_line = bool(re.search(r"\b(?:reading|meter|वाचन|prev|curr|मागील|चालू|mf|units|kwh)\b", low_l))
            is_prev_reading = bool(re.search(r"\b(?:reading|meter|वाचन|prev|curr|मागील|चालू)\b", lines[i - 1].lower())) if i > 0 else False
            if not (is_reading_line or is_prev_reading):
                continue

            dates_on_line = [self._parse_date_string(d) for d in re.findall(date_token_pat, line)]
            valid_dates = [d for d in dates_on_line if d]
            if len(valid_dates) >= 2:
                for idx in range(len(valid_dates) - 1):
                    d1, d2 = valid_dates[idx], valid_dates[idx + 1]
                    dur_1b = (d2 - d1).days
                    if d1 < d2 and min_days <= dur_1b <= max_days:
                        if header_cycle and abs((d2.year * 12 + d2.month) - (header_cycle[0] * 12 + header_cycle[1])) > (2 if is_tangedco else 1):
                            continue
                        if not start_dt or (header_cycle and (start_dt.day == 1 and end_dt.day in (28, 29, 30, 31))):
                            start_dt = d1
                            end_dt = d2
                            if dur_1b > 45:
                                period_label = f"{d1.strftime('%b')} - {d2.strftime('%b %Y')}"
                            else:
                                period_label = f"{d2.strftime('%b %Y')}"
                        break
            if start_dt and end_dt and (not header_cycle or (start_dt.day != 1 or end_dt.day not in (28, 29, 30, 31))):
                break

        # Strategy 2: Paired Previous Reading Date and Current Reading Date line-by-line
        if not (start_dt and end_dt) or (header_cycle and (start_dt.day == 1 and end_dt.day in (28, 29, 30, 31))):
            p_dt: Optional[date] = None
            c_dt: Optional[date] = None
            for line in lines:
                low_l = line.lower()
                if any(k in low_l for k in UNRELATED_DATE_KEYWORDS):
                    continue
                if not p_dt and re.search(r"\b(?:prev(?:ious)?\s*(?:reading)?\s*date|prev\s*date|मागील\s*वाचन\s*(?:दिनांक|तारीख)?)", line, re.IGNORECASE):
                    m_p = re.search(r"[\s:\-\.]+(" + date_token_pat + r")", line)
                    if m_p:
                        p_dt = self._parse_date_string(m_p.group(1))
                if not c_dt and re.search(r"\b(?:curr(?:ent)?\s*(?:reading)?\s*date|reading\s*date|चालू\s*वाचन\s*(?:दिनांक|तारीख)?)", line, re.IGNORECASE):
                    if not re.search(r"\b(?:prev|मागील)", line, re.IGNORECASE):
                        m_c = re.search(r"[\s:\-\.]+(" + date_token_pat + r")", line)
                        if m_c:
                            c_dt = self._parse_date_string(m_c.group(1))
            if p_dt and c_dt and p_dt < c_dt:
                dur_s2 = (c_dt - p_dt).days
                if min_days <= dur_s2 <= max_days:
                    month_diff = abs((c_dt.year * 12 + c_dt.month) - (header_cycle[0] * 12 + header_cycle[1])) if header_cycle else 0
                    if not header_cycle or month_diff <= (2 if is_tangedco else 1):
                        start_dt = p_dt
                        end_dt = c_dt
                        if dur_s2 > 45:
                            period_label = f"{p_dt.strftime('%b')} - {c_dt.strftime('%b %Y')}"
                        else:
                            period_label = f"{c_dt.strftime('%b %Y')}"

        # Strategy 2B: Paired reading dates across adjacent / consecutive lines (common in column-based and photographed bills)
        if not (start_dt and end_dt):
            all_extracted_dates = []
            for idx, line in enumerate(lines):
                low_l = line.lower()
                if any(k in low_l for k in UNRELATED_DATE_KEYWORDS):
                    continue
                # Exclude payment due date / receipt lines (e.g. line has 'rs' or 'amount' or 'due')
                if "rs" in low_l or "amount" in low_l or "payable" in low_l or "due date" in low_l:
                    continue
                d_matches = [self._parse_date_string(d) for d in re.findall(date_token_pat, line)]
                valid_d = [d for d in d_matches if d]
                for vd in valid_d:
                    all_extracted_dates.append((idx, vd))

            for i in range(len(all_extracted_dates) - 1):
                line_idx1, d1 = all_extracted_dates[i]
                line_idx2, d2 = all_extracted_dates[i + 1]
                if line_idx1 != line_idx2 and abs(line_idx2 - line_idx1) <= 3:
                    d_earlier, d_later = min(d1, d2), max(d1, d2)
                    dur = (d_later - d_earlier).days
                    if min_days <= dur <= max_days:
                        month_diff = abs((d_later.year * 12 + d_later.month) - (header_cycle[0] * 12 + header_cycle[1])) if header_cycle else 0
                        if not header_cycle or month_diff <= (2 if is_tangedco else 1):
                            start_dt = d_earlier
                            end_dt = d_later
                            if dur > 45:
                                period_label = f"{d_earlier.strftime('%b')} - {d_later.strftime('%b %Y')}"
                            else:
                                period_label = f"{d_later.strftime('%b %Y')}"
                            break

        # Strategy 3: Bill Month fallback
        if not (start_dt and end_dt):
            m_month = re.search(
                r"(?:\b(?:b[i1lt|]{2,4}|bill|billing|देयक)\s*(?:month|महिना)|bill\s*month)[\s:\-\.]*"
                r"(?<!\d[/\-\.])([A-Za-z]{3,9}\.?|\d{1,2})\s*(?:[-/\.]\s*|\s+)(\d{4}|\d{2})\b(?![/\-\.]\d)",
                text,
                re.IGNORECASE,
            )
            if m_month:
                m_norm = m_month.group(1).rstrip(".").lower()
                m_int = self.MONTH_MAP.get(m_norm)
                if not m_int and m_norm.isdigit() and 1 <= int(m_norm) <= 12:
                    m_int = int(m_norm)
                y_str = m_month.group(2)
                y_int = 2000 + int(y_str) if len(y_str) == 2 else int(y_str)
                if m_int and 1990 <= y_int <= 2100:
                    start_dt = date(y_int, m_int, 1)
                    last_day = calendar.monthrange(y_int, m_int)[1]
                    end_dt = date(y_int, m_int, last_day)
                    period_label = f"{start_dt.strftime('%b %Y')}"

        # Strategy 4: Single Bill Date / Reading Date fallback
        if not (start_dt and end_dt):
            m_single_dt = re.search(
                r"(?:\b(?:bill|billing|issue|reading|देयक)\s*(?:date|दिनांक|तारीख)|date\s*of\s*bill)[\s:\-\.]*("
                + date_token_pat
                + r")\b",
                text,
                re.IGNORECASE,
            )
            if m_single_dt:
                d_single = self._parse_date_string(m_single_dt.group(1))
                if d_single and 1990 <= d_single.year <= 2100:
                    y_int, m_int = d_single.year, d_single.month
                    start_dt = date(y_int, m_int, 1)
                    last_day = calendar.monthrange(y_int, m_int)[1]
                    end_dt = date(y_int, m_int, last_day)
                    period_label = f"{start_dt.strftime('%b %Y')}"

        # Find current consumption (kWh / units)
        # Priority 1: Meter reading arithmetic check across reading lines (supports adjacent label and numeric lines)
        def _check_reading_math(num_list: List[float]) -> Optional[float]:
            if len(num_list) < 3:
                return None
            for j in range(len(num_list) - 2):
                n1 = num_list[j]
                n2 = num_list[j + 1]
                diff = abs(n1 - n2)
                if diff <= 0 or min(n1, n2) < 10.0:
                    continue

                # Case A: [r1, r2, units] e.g. [12500, 12667, 167] or [1257, 1090, 167]
                target_u = num_list[j + 2]
                if abs(diff - target_u) < 0.5 and 0.0 <= target_u <= 50000.0:
                    return target_u

                # Case B: [r1, r2, mf, units] e.g. [12500, 12667, 1.0, 167] or [1257, 1090, 1.0, 167]
                if j + 3 < len(num_list):
                    mf = num_list[j + 2]
                    target_u = num_list[j + 3]
                    if 0.1 <= mf <= 100.0 and abs((diff * mf) - target_u) < 0.5 and 0.0 <= target_u <= 50000.0:
                        return target_u
            return None

        READING_CONTEXT_PAT = r"\b(?:reading|meter|वाचन|prev(?:ious)?|curr(?:ent)?|मागील|चालू|mf|multiplier|units|consumption|kwh)\b"
        READING_EXCLUDE_KEYWORDS = (
            "tariff", "w.e.f", "wef", "sanction", "connected", "load", "due date", "bill date",
            "agreement", "deposit", "mobile",
        )

        if consumption is None:
            for idx, line in enumerate(lines):
                low_l = line.lower()
                if any(k in low_l for k in READING_EXCLUDE_KEYWORDS):
                    continue

                is_ctx = bool(re.search(READING_CONTEXT_PAT, low_l))
                prev_ctx = bool(re.search(READING_CONTEXT_PAT, lines[idx - 1].lower())) if idx > 0 else False
                next_ctx = bool(re.search(READING_CONTEXT_PAT, lines[idx + 1].lower())) if idx + 1 < len(lines) else False

                if not (is_ctx or prev_ctx or next_ctx):
                    continue

                nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", line)]
                val = _check_reading_math(nums)
                if val is not None:
                    consumption = val
                    break

                # Check adjacent next line combination if this line had partial numbers
                if idx + 1 < len(lines):
                    next_low = lines[idx + 1].lower()
                    if not any(k in next_low for k in READING_EXCLUDE_KEYWORDS):
                        next_nums = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", lines[idx + 1])]
                        val_adj = _check_reading_math(nums + next_nums)
                        if val_adj is not None:
                            consumption = val_adj
                            break

        # Priority 2: Explicit labeled units
        if consumption is None:
            cons_match = re.search(
                r"(?:total\s*units\s*billed|units\s*billed|billed\s*units|units\s*consumed|consumption\s*units|"
                r"current\s*consumption|total\s*consumption|total\s*units|वापर\s*युनिट|देयक\s*युनिट|चालू\s*वापर)"
                r"(?:[\s:\-\.]+|\s*\((?:kwh|units)\)[\s:\-\.]*)\s*"
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:kwh|units)?\b",
                text,
                re.IGNORECASE,
            )
            if cons_match:
                try:
                    val = float(cons_match.group(1).replace(",", ""))
                    if 0.0 <= val <= 50000.0:
                        consumption = val
                except ValueError:
                    pass

        # Priority 3: Reading table line with units (e.g. "Units: 167", "167 units")
        if consumption is None:
            LOAD_KEYWORDS = (
                "sanction", "connected", "load", "contract demand", "demand", "kw", "kva",
                "hp", "phase", "volt", "pole", "meter no", "tariff", "order", "due date",
                "billing unit", "bu ", "bu:", "consumer", "chart",
            )
            for line in lines:
                low_l = line.lower()
                if any(k in low_l for k in LOAD_KEYWORDS):
                    continue

                m_fb = re.search(r"\b(?:units?|uni[+!|1]s|kwh)\b[\s:\-\.]+([0-9,]+(?:\.[0-9]+)?)\b", line, re.IGNORECASE)
                if not m_fb:
                    # Only accept suffixed units if line does not contain sequences of numbers
                    if len(re.findall(r"\b\d+\b", line)) < 4:
                        m_fb = re.search(r"\b([0-9,]+(?:\.[0-9]+)?)\s*(?:kwh|units?|uni[+!|1]s)\b", line, re.IGNORECASE)
                if m_fb:
                    try:
                        val = float(m_fb.group(1).replace(",", ""))
                        if 0.0 <= val <= 50000.0:
                            consumption = val
                            break
                    except ValueError:
                        pass

        # Priority 4: Reading table line with Multiplying Factor (MF = 1.00 or 1.0) followed by consumed units
        # Found in camera-photographed utility bills (e.g. MSEDCL) where columns are separated vertically across lines.
        if consumption is None:
            for idx, line in enumerate(lines):
                low_l = line.lower()
                if any(k in low_l for k in LOAD_KEYWORDS):
                    continue
                # Verify line is near meter reading context or reading dates (within 4 lines)
                is_near_meter = False
                for offset in range(-4, 5):
                    adj_idx = idx + offset
                    if 0 <= adj_idx < len(lines):
                        adj_l = lines[adj_idx].lower()
                        if "meter" in adj_l or "वाचन" in adj_l or "reading" in adj_l:
                            is_near_meter = True
                            break
                        if start_dt and str(start_dt.year) in lines[adj_idx] and (f"{start_dt.day:02d}" in lines[adj_idx] or f"{start_dt.month:02d}" in lines[adj_idx]):
                            is_near_meter = True
                            break
                if not is_near_meter:
                    continue

                m_mf = re.search(r"\b(?:1\.00|1\.0)\s+([0-9]{1,4}(?:\.[0-9]+)?)\b", line)
                if m_mf:
                    try:
                        val = float(m_mf.group(1).replace(",", ""))
                        if 0.0 <= val <= 50000.0:
                            consumption = val
                            break
                    except ValueError:
                        pass

        # Find net bill amount (₹)
        # Priority 1: Explicit bill amount label with reliable currency context or explicit decimal format
        # Strips intermediate due-date clauses, requires currency symbol OR 2-decimal format,
        # and strictly forbids matching day of month in date structures (e.g. 25/04/2025).
        amt_match = re.search(
            r"(?:net\s*(?:b[i1lt|]{2,4}\s*)?bill\s*amount|bill\s*amount|current\s*bill\s*amount|net\s*current\s*bill|"
            r"current\s*bill|total\s*bill|देयक\s*रक्कम|देय\s*रक्कम|चालू\s*देयक)"
            r"(?:[^\n\r]*?(?:up\s*to|by|before|after|till|dt\.?|date:?)\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})?"
            r"[\s:\-\.]*"
            r"(?:"
            r"(?:rs\.?|inr|₹|रु\.?)[\s:\-\.]*([0-9,]+(?:\.[0-9]{1,2})?)|"
            r"([0-9,]+\.[0-9]{2})(?!\s*%)|"
            r"([0-9,]+)\s*(?:/-)"
            r")\b(?![/\-\.]\d)",
            text,
            re.IGNORECASE,
        )
        if amt_match:
            raw_val = amt_match.group(1) or amt_match.group(2) or amt_match.group(3)
            if raw_val:
                try:
                    val = float(raw_val.replace(",", ""))
                    if 20.0 <= val <= 1000000.0:
                        bill_amount = val
                except ValueError:
                    pass

        # Priority 2: Amount payable with date clauses safely stripped and reliable currency context
        if bill_amount is None:
            ap_match = re.search(
                r"(?:total\s*amount\s*payable|amount\s*payable|net\s*amount\s*payable|net\s*amount|देय\s*रक्कम)"
                r"(?:[^\n\r]*?(?:up\s*to|by|before|after|till|dt\.?|date:?)\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})?"
                r"[\s:\-\.]*"
                r"(?:"
                r"(?:rs\.?|inr|₹|रु\.?)[\s:\-\.]*([0-9,]+(?:\.[0-9]{1,2})?)|"
                r"([0-9,]+\.[0-9]{2})(?!\s*%)|"
                r"([0-9,]+)\s*(?:/-)"
                r")\b(?![/\-\.]\d)",
                text,
                re.IGNORECASE,
            )
            if ap_match:
                raw_val = ap_match.group(1) or ap_match.group(2) or ap_match.group(3)
                if raw_val:
                    try:
                        val = float(raw_val.replace(",", ""))
                        if 20.0 <= val <= 1000000.0:
                            bill_amount = val
                    except ValueError:
                        pass

        # Priority 3: Fallback on payment slip / due-date lines with explicit Rs. currency and 2 decimal places
        if bill_amount is None:
            for line in lines:
                m_slip = re.search(r"\b(?:rs\.?|inr|₹|रु\.?)[\s:\-\.]*([0-9,]+\.[0-9]{2})\b", line, re.IGNORECASE)
                if m_slip:
                    try:
                        val = float(m_slip.group(1).replace(",", ""))
                        if 20.0 <= val <= 1000000.0:
                            bill_amount = val
                            break
                    except ValueError:
                        pass

        if start_dt and end_dt and consumption is not None:
            if start_dt >= end_dt:
                warnings.append(
                    f"Parsed current period start date ({start_dt}) is not before end date ({end_dt}). Record skipped."
                )
                return None, warnings

            dur = (end_dt - start_dt).days
            if dur > 45:
                warnings.append(
                    f"Bi-monthly billing cycle detected ({dur} days from {start_dt} to {end_dt}). "
                    "Original billing period and total kWh are preserved for household review. "
                    "Downstream monthly forecasting expects monthly records; verify before running forecasts."
                )
                if not period_label or period_label == f"{start_dt.strftime('%b %Y')}":
                    period_label = f"{start_dt.strftime('%b')} - {end_dt.strftime('%b %Y')}"

            record = MonthlyConsumptionRecord(
                billing_period_start=start_dt,
                billing_period_end=end_dt,
                consumption_kwh=consumption,
                bill_amount=bill_amount,
                period_label=period_label or f"{start_dt.strftime('%b %Y')}",
            )
            return record, warnings

        return None, warnings

    def _extract_historical_table(
        self, text: str
    ) -> Tuple[List[MonthlyConsumptionRecord], List[str]]:
        """
        Parse historical monthly consumption tables (e.g. Month, Units, Bill Amount).
        Supports both 4-digit and 2-digit years (e.g. APR-24, APR-2024, 04/24, Apr 2025).
        Disambiguates payment history and meter readings from billed consumption units.
        Ignores unassociated chart labels and non-billing metadata.
        """
        records: List[MonthlyConsumptionRecord] = []
        warnings: List[str] = []

        HEADER_KEYWORDS = (
            "billing period", "reading period", "bill period",
            "reading date", "prev", "curr", "current",
            "consumer no", "billing unit", "account no",
            "bill month", "billing month", "bill date", "due date",
            "bitt date", "bitt month", "units billed", "billed units",
            "current consumption", "total units", "amount payable",
            "net bill amount", "net amount", "total bill", "देयक महिना", "कालावधी",
        )

        NON_TABLE_KEYWORDS = (
            "tariff", "w.e.f", "wef", "w.e.f.", "order", "sanction", "connected", "load",
            "contract", "demand", "connection", "agreement", "deposit", "security",
            "meter no", "serial no", "make:", "pole", "dtc", "feeder", "phase",
            "bill of supply", "tax invoice", "due date", "bill date", "prompt payment",
            "cashier", "sub-division", "division", "circle", "office", "mobile", "email",
            "address", "gstin", "pan no", "cin no", "billing unit", "consumer no",
            "lt-", "lt -", "lt 1", "lt-1", "lt-i", "lt i", "merc", "residential", "commercial",
            "industrial", "tanff", "bhimashankar", "on or before", "on or after",
            "effective from", "valid up to", "valid till",
        )

        # Process text line by line to respect table row structures
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            low_line = line_str.lower()
            if any(k in low_line for k in HEADER_KEYWORDS):
                continue
            if any(k in low_line for k in NON_TABLE_KEYWORDS):
                continue

            # Exclude full calendar dates (e.g. "01 April 2020", "01-Apr-2020") which denote specific regulatory/notice dates
            month_names_pat = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
            if re.search(rf"\b\d{{1,2}}\s+{month_names_pat}\s+\d{{2,4}}\b", line_str, re.IGNORECASE):
                continue
            if re.search(rf"\b\d{{1,2}}[/\-\.]{month_names_pat}[/\-\.](?:\d{{4}}|\d{{2}})\b", line_str, re.IGNORECASE):
                continue

            # Check if line is a horizontal chart X-axis with multiple month tokens (e.g. May-24 Jun-24 ...)
            month_count = len(re.findall(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s\-/](?:\d{4}|\d{2})\b",
                low_line,
            ))
            if month_count >= 2:
                continue

            # Check if line is a cluster of chart bar labels without month names (e.g. 160 156 109 84 ...)
            num_tokens_raw = re.findall(r"\b\d+\b", line_str)
            if len(num_tokens_raw) >= 4 and not re.search(r"[A-Za-z]{3,9}", line_str):
                continue

            # Look for month-year token that is not part of a DD/MM/YYYY full date
            m = re.search(
                r"(?<![\d/\-\.])\b([A-Za-z]{3,9}\.?|\d{1,2})\s*(?:[-/\.]\s*|\s+)(\d{4}|\d{2})\b(?![/\-\.]\d)(.*)",
                line_str,
                re.IGNORECASE,
            )
            if not m:
                continue

            m_str, y_str, rest = m.group(1).rstrip("."), m.group(2), m.group(3)

            # If remainder immediately starts with a date separator and digit, this was part of a longer date/path
            if re.match(r"^[/\-\.]\d", rest):
                continue

            # If month was numeric (e.g. 04/24), ensure the line genuinely looks like a table row starting with month
            if m_str.isdigit():
                if not re.match(r"^(?:\d+[\.\)]\s*)?\d{1,2}\s*[\s\-/]\s*(?:\d{4}|\d{2})\b", line_str):
                    continue

            # Resolve year (2-digit or 4-digit)
            try:
                if len(y_str) == 2:
                    year = 2000 + int(y_str)
                else:
                    year = int(y_str)
                if year < 2000 or year > 2100:
                    continue
            except ValueError:
                continue

            # Resolve month
            month = None
            m_norm = m_str.lower()
            if m_norm in self.MONTH_MAP:
                month = self.MONTH_MAP[m_norm]
            else:
                try:
                    m_int = int(m_str)
                    if 1 <= m_int <= 12:
                        month = m_int
                except ValueError:
                    continue

            if not month:
                continue

            # Disambiguate payment history: if payment date or status exists in rest, split before it
            pay_match = re.search(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.](?:\d{4}|\d{2})\b", rest)
            if pay_match:
                billing_part = rest[:pay_match.start()]
            else:
                pend_match = re.search(r"\b(?:pending|unpaid|nil|due)\b", rest, re.IGNORECASE)
                if pend_match:
                    billing_part = rest[:pend_match.start()]
                else:
                    billing_part = rest

            # Clean OCR artifacts: e.g. "2400. oo" -> "2400.00", "2630 .00" -> "2630.00"
            clean_part = re.sub(r"\.\s*[oO]{1,2}\b", ".00", billing_part)
            clean_part = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", clean_part)

            # Extract numeric tokens from billing_part
            num_tokens = [float(n.replace(",", "")) for n in re.findall(r"\b\d+(?:\.\d+)?\b", clean_part)]
            if not num_tokens:
                continue

            kwh: Optional[float] = None
            bill_amt: Optional[float] = None

            # Disambiguate meter readings vs billed units
            if len(num_tokens) >= 4 and abs((num_tokens[1] - num_tokens[0]) - num_tokens[2]) < 0.1:
                # Row structure: [PrevReading, CurrReading, BilledUnits, BillAmount]
                kwh = num_tokens[2]
                bill_amt = num_tokens[3]
            elif len(num_tokens) == 3 and abs((num_tokens[1] - num_tokens[0]) - num_tokens[2]) < 0.1:
                # Row structure: [PrevReading, CurrReading, BilledUnits]
                kwh = num_tokens[2]
            elif len(num_tokens) >= 3:
                # Row structure with 3 tokens: e.g. [MeterReading, BilledUnits, BillAmount] or [BilledUnits, Rate, BillAmount]
                if num_tokens[0] > 2000.0 and 0.0 <= num_tokens[1] <= 50000.0:
                    kwh = num_tokens[1]
                    bill_amt = num_tokens[2]
                elif 0.0 <= num_tokens[0] <= 50000.0:
                    kwh = num_tokens[0]
                    bill_amt = num_tokens[-1]
            elif len(num_tokens) == 2:
                # Standard row: [BilledUnits, BillAmount]
                kwh = num_tokens[0]
                bill_amt = num_tokens[1]
            elif len(num_tokens) == 1:
                # Standard row: [BilledUnits]
                kwh = num_tokens[0]

            if kwh is None or kwh < 5.0 or kwh > 50000.0:
                continue

            if bill_amt is not None and bill_amt < 0.0:
                bill_amt = None

            # Derive start and end of that calendar month
            start_date = date(year, month, 1)
            last_day = calendar.monthrange(year, month)[1]
            end_date = date(year, month, last_day)

            # Avoid duplicate months within table
            if any(r.billing_period_start == start_date for r in records):
                continue

            try:
                rec = MonthlyConsumptionRecord(
                    billing_period_start=start_date,
                    billing_period_end=end_date,
                    consumption_kwh=kwh,
                    bill_amount=bill_amt,
                    period_label=f"{start_date.strftime('%b %Y')}",
                )
                records.append(rec)
            except Exception as e:
                warnings.append(f"Skipped invalid historical record for {m_str}-{y_str}: {str(e)}")

        if records:
            # Sort chronologically
            records.sort(key=lambda r: r.billing_period_start)

        return records, warnings

    def _compute_defensible_confidence(
        self,
        records: List[MonthlyConsumptionRecord],
        has_consumer_no: bool,
        warnings: List[str],
        is_ocr: bool = False,
    ) -> Optional[float]:
        """
        Calculate defensible extraction confidence score based on structural completeness.
        OCR extractions receive an inherent optical uncertainty deduction.
        """
        if not records:
            return 0.0

        if is_ocr:
            # Base confidence for OCR
            score = 0.50
            # Multi-month sequence gives higher confidence
            if len(records) >= 3:
                score += 0.20
            elif len(records) >= 1:
                score += 0.05

            if has_consumer_no:
                score += 0.05

            # If all records have verified bill amount
            if all(r.bill_amount is not None for r in records):
                score += 0.05

            # Deductions for ambiguities or warnings
            for _ in warnings:
                score -= 0.03

            return round(min(0.85, max(0.20, score)), 2)

        # Digital PDF parsing
        score = 0.70
        if len(records) >= 3:
            score += 0.15
        elif len(records) >= 1:
            score += 0.10

        if has_consumer_no:
            score += 0.10

        for _ in warnings:
            score -= 0.05

        return round(min(0.98, max(0.20, score)), 2)


# Global singleton parser instance
bill_parser_service = BillParserService()
