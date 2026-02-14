#!/usr/bin/env python
"""Run local inference with the fine-tuned adapter."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.summarizer import LocalSummarizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Local email summarizer inference")
    parser.add_argument("--adapter_path", default="train/output", help="Path to LoRA adapter directory")
    parser.add_argument("--model_name", default="Qwen3-4B-Instruct-2507")
    parser.add_argument("--subject", default="(no subject)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", dest="file_path")
    group.add_argument("--text", dest="text")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.file_path:
        text = Path(args.file_path).read_text(encoding="utf-8")
    else:
        text = args.text

    summarizer = LocalSummarizer(model_name=args.model_name, adapter_path=args.adapter_path)
    summary = summarizer.summarize(subject=args.subject, body=text)
    print(summary)


if __name__ == "__main__":
    main()
