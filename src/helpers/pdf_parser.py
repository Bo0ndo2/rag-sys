import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Arabic normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_arabic(text: str) -> str:
    
    # Remove tashkeel (U+064B–U+065F, U+0670)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Alef with hamza variants → bare alef
    text = re.sub(r"[إأآ]", "ا", text)
    # Teh marbuta → heh
    text = text.replace("ة", "ه")
    # Remove tatweel
    text = text.replace("\u0640", "")
    return text


def _clean_text(text: str) -> str:
    """General cleanup: collapse whitespace, fix soft-hyphens, strip control chars."""
    text = text.replace("\u00AD", "")          # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)        # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)     # max 2 consecutive newlines
    text = re.sub(r"[^\S\n]+\n", "\n", text)   # trailing spaces on lines
    return text.strip()


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class PDFParser:
  

    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

    def _is_arabic_dominant(self, text: str) -> bool:
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        total_alpha  = len(re.findall(r"[A-Za-z\u0600-\u06FF]", text))
        return total_alpha > 0 and (arabic_chars / total_alpha) > 0.4


    def parse(self) -> "ParseResult":
        doc = fitz.open(str(self.path))
        pages: list[str] = []
        is_arabic = False
        

        for page in doc:
            # Extract text preserving reading order; for RTL pages use "rawdict"
            raw = page.get_text("text", sort=True)  # sort=True → reading order

            raw = _clean_text(raw)
            if not raw:
                continue

            if self._is_arabic_dominant(raw):
                raw = _normalise_arabic(raw)
                is_arabic = True

            pages.append(raw)

        metadata = {
            "title":      doc.metadata.get("title", self.path.stem),
            "author":     doc.metadata.get("author", "unknown"),
            "page_count": len(doc),
            "file_name":  self.path.name,
            "is_arabic":  is_arabic,
        }
        doc.close()

        return ParseResult(pages=pages, metadata=metadata)


class ParseResult:
    def __init__(self, pages: list[str], metadata: dict):
        self.pages    = pages
        self.metadata = metadata

    @property
    def full_text(self) -> str:
        return "\n\n".join(self.pages)

    def __repr__(self):
        return (
            f"<ParseResult pages={len(self.pages)} "
            f"chars={len(self.full_text)} "
            f"arabic={self.metadata['is_arabic']}>"
        )
