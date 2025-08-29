import pytesseract
from PIL import Image
import re
from pathlib import Path
from typing import Optional

def extract_text_from_image(image_path: Path) -> Optional[str]:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return None

def extract_urls(text: str) -> list[str]:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)
