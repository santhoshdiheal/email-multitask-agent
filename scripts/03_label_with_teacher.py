#!/usr/bin/env python
"""Call teacher model to label processed emails."""
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from progress.bar import Bar

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402


load_dotenv()

INPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "enron_clean.jsonl"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "labeled" / "enron_labeled.jsonl"

API_BASE = os.getenv("TEACHER_API_BASE", "")
API_KEY = os.getenv("TEACHER_API_KEY", "")
MODEL = os.getenv("TEACHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
MAX_ITEMS = int(os.getenv("MAX_LABEL_ITEMS", "0") or 0)
SLEEP_SEC = float(os.getenv("SLEEP_SEC", "0.5"))


def build_messages(subject: str, body: str):
    user_prompt = USER_PROMPT_TEMPLATE.format(subject=subject or "(no subject)", body=body)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def is_valid_summary(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) != 5:
        return False
    expected_prefixes = ["1) Purpose:", "2) Key details:", "3) Action needed:", "4) Deadline/Date:", "5) Important context/links:"]
    for line, prefix in zip(lines, expected_prefixes):
        if not line.startswith(prefix):
            return False
    return True


def call_teacher(messages):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    url = API_BASE.rstrip("/") + "/v1/chat/completions"
    resp = httpx.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def label_item(record):
    messages = build_messages(record.get("subject", ""), record.get("clean_text", ""))
    output = call_teacher(messages)
    if is_valid_summary(output):
        return messages + [{"role": "assistant", "content": output}]

    # retry with correction
    correction = messages + [
        {
            "role": "user",
            "content": "The previous answer was invalid. Output EXACTLY 5 lines with the specified labels.",
        }
    ]
    output = call_teacher(correction)
    if is_valid_summary(output):
        return messages + [{"role": "assistant", "content": output}]
    return None


def main():
    if not API_BASE or not API_KEY:
        print("Please set TEACHER_API_BASE and TEACHER_API_KEY in your .env", file=sys.stderr)
        sys.exit(1)
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with INPUT.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if MAX_ITEMS and MAX_ITEMS > 0:
        records = records[:MAX_ITEMS]

    with OUTPUT.open("w", encoding="utf-8") as out_f:
        bar = Bar('Labeling', max=len(records))
        for rec in records:
            labeled = label_item(rec)
            if labeled:
                out_f.write(json.dumps({"messages": labeled}, ensure_ascii=False) + "\n")
            bar.next()
            time.sleep(SLEEP_SEC)
        bar.finish()
    print(f"Saved labeled data to {OUTPUT}")


if __name__ == "__main__":
    main()
