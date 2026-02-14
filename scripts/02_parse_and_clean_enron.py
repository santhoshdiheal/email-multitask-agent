#!/usr/bin/env python
"""Parse Enron maildir emails, clean, and export JSONL."""
import json
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

from progress.bar import Bar

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.cleaner import clean_email_text, extract_preferred_body  # noqa: E402


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "enron_clean.jsonl"
def main():
    maildir_root = RAW_DIR / "maildir"
    if not maildir_root.exists():
        print(f"Expected extracted maildir at {maildir_root}. Extract tarball first.")
        sys.exit(1)

    files = list(maildir_root.rglob("*") )
    msg_files = [p for p in files if p.is_file() and not p.name.startswith('.')]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as out_f:
        bar = Bar('Processing', max=len(msg_files))
        for path in msg_files:
            try:
                with path.open("rb") as f:
                    msg = BytesParser(policy=policy.default).parse(f)
                subject = msg.get("subject", "") or ""
                body = extract_preferred_body(msg)
                clean_text, urls = clean_email_text(subject, body)
                record = {
                    "source": "enron",
                    "path": str(path),
                    "subject": subject,
                    "clean_text": clean_text,
                    "urls": urls,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:  # pragma: no cover - log and continue
                print(f"Failed to parse {path}: {exc}", file=sys.stderr)
            finally:
                bar.next()
        bar.finish()
    print(f"Saved cleaned data to {OUTPUT}")


if __name__ == "__main__":
    main()
