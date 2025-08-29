import re
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from docx import Document

# Basic PPI regex patterns
PATTERNS = {
    "email": re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'),
    "phone": re.compile(r'\b(?:\+?\d{1,3})?[\s.-]?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b'),
    "name": re.compile(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'),  # e.g., John Doe
}

def redact_text(text: str) -> str:
    for pattern in PATTERNS.values():
        text = pattern.sub("[REDACTED]", text)
    return text

def redact_docx(path: Path) -> Optional[Path]:
    try:
        doc = Document(str(path))
        for para in doc.paragraphs:
            para.text = redact_text(para.text)
        output_path = Path("data/redacted") / path.name
        output_path.parent.mkdir(exist_ok=True)
        doc.save(str(output_path))
        return output_path
    except Exception:
        return None

def redact_pdf(path: Path) -> Optional[Path]:
    try:
        doc = fitz.open(str(path))
        output_path = Path("data/redacted") / path.name
        output_path.parent.mkdir(exist_ok=True)

        for page in doc:
            text = page.get_text("text")  # type: ignore

            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    rects = page.search_for(match.group())  # type: ignore
                    for rect in rects:
                        page.add_redact_annot(rect, fill=(0, 0, 0))  # type: ignore

        doc.apply_redactions()  # type: ignore
        doc.save(str(output_path))
        return output_path
    except Exception:
        return None
