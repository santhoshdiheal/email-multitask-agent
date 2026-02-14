import re
from typing import List, Tuple

from bs4 import BeautifulSoup


URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
QUOTE_SEPARATORS = [
    r"^On .*wrote:$",
    r"^From:\s.*$",
    r"^Sent:\s.*$",
    r"^Subject:\s.*$",
    r"^Original Message$",
]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(" ")
    return text


def strip_quoted_history(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    quote_regexes = [re.compile(pat, re.IGNORECASE) for pat in QUOTE_SEPARATORS]
    for line in lines:
        if any(r.match(line.strip()) for r in quote_regexes):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def remove_signature(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return text.strip()

    signature_markers = [
        "--",
        "Thanks",
        "Regards",
        "Best",
        "Sincerely",
        "Thank you",
    ]

    cutoff = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if any(line.startswith(m) for m in signature_markers):
            cutoff = i
            break
    return "\n".join(lines[:cutoff]).strip()


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_urls(text: str) -> List[str]:
    return list({match.group(0).rstrip('.,)') for match in URL_PATTERN.finditer(text)})


def truncate_thread(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def clean_email_text(subject: str, body: str) -> Tuple[str, List[str]]:
    text = body or ""
    text = strip_quoted_history(text)
    text = remove_signature(text)
    urls = extract_urls(text)
    text = normalize_whitespace(text)
    text = truncate_thread(text)
    combined = f"Subject: {subject}\n{text}" if subject else text
    return combined, urls


def extract_preferred_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    return part.get_content().strip()
                except Exception:
                    continue
        for part in message.walk():
            if part.get_content_type() == "text/html":
                try:
                    return html_to_text(part.get_content())
                except Exception:
                    continue
    else:
        content_type = message.get_content_type()
        try:
            if content_type == "text/html":
                return html_to_text(message.get_content())
            return message.get_content().strip()
        except Exception:
            return ""
    return ""
