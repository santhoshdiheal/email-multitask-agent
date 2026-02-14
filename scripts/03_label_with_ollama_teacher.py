#!/usr/bin/env python
"""Label emails using a local Ollama teacher model."""
import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm


IN_PATH = Path("data/processed/enron_clean.jsonl")
OUT_PATH = Path("data/labeled/train.jsonl")

OLLAMA_URL = "http://localhost:11434/api/chat"

# Default teacher: fast, good-quality labels
OLLAMA_MODEL = "qwen2.5:7b-instruct"

MAX_ITEMS = 2000  # overnight run; adjust as needed (e.g., 5000/10000)
SLEEP_SEC = 0.0

SYSTEM = """You summarize emails. Output EXACTLY 5 lines with this schema:
1) Purpose: ...
2) Key details: ...
3) Action needed: ...
4) Deadline/Date: ...
5) Important context/links: ...
Rules:
- EXACTLY 5 lines only (no extra text, no blank lines).
- Be faithful to the email.
- Include names, numbers, and links when present.
"""

LINE_RE = re.compile(
    r"^\s*\d\)\s*(Purpose|Key details|Action needed|Deadline/Date|Important context/links):\s*.+",
    re.IGNORECASE,
)


def is_valid(summary: str) -> bool:
    lines = [l.strip() for l in summary.strip().splitlines() if l.strip()]
    if len(lines) != 5:
        return False
    return all(LINE_RE.match(lines[i]) for i in range(5))


def call_ollama(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 140},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def label_one(subject: str, email_text: str):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Subject: {subject}\n\nEmail:\n{email_text}"},
    ]
    out = call_ollama(msgs)
    if is_valid(out):
        return out

    # Retry once with strict correction
    msgs2 = msgs + [
        {"role": "assistant", "content": out},
        {"role": "user", "content": "Fix ONLY the format. Output EXACTLY 5 lines in the required schema. No extra text."},
    ]
    out2 = call_ollama(msgs2)
    if is_valid(out2):
        return out2

    return None


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run your email cleaning step first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = IN_PATH.read_text(encoding="utf-8").splitlines()[:MAX_ITEMS]

    kept = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for line in tqdm(rows, desc=f"Labeling with {OLLAMA_MODEL}"):
            rec = json.loads(line)
            subject = (rec.get("subject") or "").strip()
            email_text = (rec["clean_text"] or "")[:4000]  # truncate long emails for speed

            try:
                summary = label_one(subject, email_text)
                if not summary:
                    continue

                train_item = {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Subject: {subject}\n\nEmail:\n{email_text}"},
                        {"role": "assistant", "content": summary},
                    ]
                }
                f.write(json.dumps(train_item, ensure_ascii=False) + "\n")
                kept += 1

                if SLEEP_SEC:
                    time.sleep(SLEEP_SEC)

            except Exception:
                continue

    print(f"\nDone. Labeled {kept} examples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
