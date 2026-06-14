import re
from rapidfuzz import fuzz

def sanitize_filename(text: str) -> str:
    sanitized = text.strip()
    sanitized = re.sub(r'[<>:"/\\|?*]', '', sanitized)
    sanitized = re.sub(r'\s+', '_', sanitized)
    if not sanitized:
        return "untitled"
    return sanitized

def is_fuzzy_match(str1: str, str2: str, threshold: int = 95) -> bool:
    if not str1 or not str2:
        return False
    return fuzz.token_set_ratio(str1, str2) >= threshold

def clean_string(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text