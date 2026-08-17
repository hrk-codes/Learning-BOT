import io
import re

from pypdf import PdfReader

from rag.models import ParsedPage


class PdfParseError(Exception):
    """Raised when a PDF cannot produce usable text."""


class PdfParser:
    def parse(self, content: bytes) -> list[ParsedPage]:
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise PdfParseError("Encrypted PDFs are not supported in Stage 5.")
        except PdfParseError:
            raise
        except Exception as exc:
            raise PdfParseError(f"The PDF could not be opened: {exc}") from exc

        pages: list[ParsedPage] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = clean_extracted_text(page.extract_text() or "")
            except Exception as exc:
                raise PdfParseError(f"Text extraction failed on page {page_number}: {exc}") from exc
            if text:
                # Page metadata survives every later boundary so retrieval can
                # be debugged and user-facing citations can name their source.
                pages.append(ParsedPage(page_number=page_number, text=text))

        if not pages:
            raise PdfParseError(
                "No extractable text was found. The PDF may be scanned and require OCR, "
                "which is intentionally outside the first Stage 5 implementation."
            )
        return pages


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
