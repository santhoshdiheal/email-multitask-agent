import os
import sys
from pathlib import Path

import typer

from app.cleaner import clean_email_text
from app.summarizer import LocalSummarizer
from app.task_router import SUPPORTED_TASKS, run_task
from configs.task_prompts import DEFAULT_CATEGORIES


DEFAULT_MODEL = os.getenv(
    "SUMMARY_MODEL_NAME",
    "models/qwen2.5-1.5b-instruct/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)
DEFAULT_ADAPTER = os.getenv("SUMMARY_ADAPTER_PATH", "train/output")


def summarize(
    subject: str = typer.Option("(no subject)", help="Email subject"),
    file: Path = typer.Option(None, exists=True, file_okay=True, dir_okay=False, help="Path to email text file"),
    text: str = typer.Option(None, help="Email body as text"),
    task: str = typer.Option(
        "summarize",
        help=f"Task to perform. Options: {', '.join(sorted(SUPPORTED_TASKS))}",
        case_sensitive=False,
    ),
    tone: str = typer.Option("neutral", help="Tone for replies (e.g., neutral, friendly, direct)"),
    style: str = typer.Option("professional", help="Rewrite style (professional, friendly, short)"),
    length: str = typer.Option("short", help="Target length for replies (short, medium, long)"),
    categories: list[str] = typer.Option(
        DEFAULT_CATEGORIES,
        help="Category labels for auto-categorize. Provide multiple to override defaults.",
    ),
    attachments: list[str] = typer.Option(
        [],
        "--attachment",
        help="Attachment names/types for attachment analysis (repeat per file).",
    ),
    adapter_path: str = typer.Option(DEFAULT_ADAPTER, help="LoRA adapter directory"),
    model_name: str = typer.Option(DEFAULT_MODEL, help="Base model name"),
):
    """Run summarization or other email tasks."""
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
    try:
        output = run_task(
            summarizer,
            task=task,
            subject=subject,
            body=clean_body,
            tone=tone,
            style=style,
            length=length,
            categories=categories,
            attachments=attachments,
        )
    except NotImplementedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n--- {task.capitalize()} ---")
    typer.echo(output)


def main():
    typer.run(summarize)


if __name__ == "__main__":
    main()
