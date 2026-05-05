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

def clean_string(s: str) -> str:
    words = re.findall(r'[a-zA-Z0-9]+', s)
    return ' '.join(words)

def file_name2id(file_name: str) -> str:
    words = re.findall(r'[a-zA-Z]+', file_name)
    return '+'.join(words)