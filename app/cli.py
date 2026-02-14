import os
import sys
from pathlib import Path

import typer

from app.cleaner import clean_email_text
from app.summarizer import LocalSummarizer


DEFAULT_MODEL = os.getenv(
    "SUMMARY_MODEL_NAME",
    "models/qwen2.5-1.5b-instruct/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)
DEFAULT_ADAPTER = os.getenv("SUMMARY_ADAPTER_PATH", "train/output")


def summarize(
    subject: str = typer.Option("(no subject)", help="Email subject"),
    file: Path = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="Path to email text file"),
    text: str = typer.Option(None, help="Email body as text"),
    adapter_path: str = typer.Option(DEFAULT_ADAPTER, help="LoRA adapter directory"),
    model_name: str = typer.Option(DEFAULT_MODEL, help="Base model name"),
):
    """Summarize an email into exactly 5 lines."""
    body = text or (file.read_text(encoding="utf-8") if file else None)
    if body is None:
        if sys.stdin.isatty():
            typer.echo("Paste the email body, then press Ctrl-D (Ctrl-Z on Windows):", err=True)
        body = sys.stdin.read()
    body = body.strip() if body else ""
    if not body:
        typer.echo("No email content provided. Use --file, --text, or paste via stdin.", err=True)
        raise typer.Exit(code=1)
    clean_body, _ = clean_email_text("", body)

    summarizer = LocalSummarizer(model_name=model_name, adapter_path=adapter_path)
    summary = summarizer.summarize(subject=subject, body=clean_body)
    typer.echo("\n--- Summary ---")
    typer.echo(summary)


def main():
    typer.run(summarize)


if __name__ == "__main__":
    main()
