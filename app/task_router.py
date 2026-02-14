from typing import List

from app.summarizer import LocalSummarizer
from configs.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from configs.task_prompts import DEFAULT_CATEGORIES, MULTITASK_SYSTEM_PROMPTS, MULTITASK_USER_TEMPLATES


SUPPORTED_TASKS = {
    "summarize",
    "reply",
    "rewrite",
    "phishing",
    "spam",
    "categorize",
    "attachment",
    "search",  # placeholder; requires embedding index
}


def build_messages(
    task: str,
    subject: str,
    body: str,
    tone: str = "neutral",
    style: str = "professional",
    length: str = "short",
    categories: List[str] | None = None,
    attachments: List[str] | None = None,
):
    task = task.lower()
    if task == "summarize":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(subject=subject or "(no subject)", body=body),
            },
        ]

    if task == "search":
        raise NotImplementedError(
            "Smart search requires an embedding index; build one with a sentence-transformer and FAISS first."
        )

    sys_prompt = MULTITASK_SYSTEM_PROMPTS[task]
    user_template = MULTITASK_USER_TEMPLATES[task]

    if task == "reply":
        user = user_template.format(subject=subject or "(no subject)", body=body, tone=tone, length=length)
    elif task == "rewrite":
        user = user_template.format(subject=subject or "(no subject)", body=body, style=style)
    elif task == "categorize":
        cats = ", ".join(categories or DEFAULT_CATEGORIES)
        user = user_template.format(subject=subject or "(no subject)", body=body, categories=cats)
    elif task == "attachment":
        attach_str = ", ".join(attachments or ["(none provided)"])
        user = user_template.format(subject=subject or "(no subject)", body=body, attachments=attach_str)
    else:  # phishing, spam
        user = user_template.format(subject=subject or "(no subject)", body=body)

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def run_task(
    summarizer: LocalSummarizer,
    task: str,
    subject: str,
    body: str,
    tone: str = "neutral",
    style: str = "professional",
    length: str = "short",
    categories: List[str] | None = None,
    attachments: List[str] | None = None,
) -> str:
    task = task.lower()
    if task == "summarize":
        return summarizer.summarize(subject=subject, body=body)

    messages = build_messages(
        task=task,
        subject=subject,
        body=body,
        tone=tone,
        style=style,
        length=length,
        categories=categories,
        attachments=attachments,
    )

    max_tokens = 256 if task in {"reply", "rewrite"} else 160
    return summarizer.generate(messages, max_new_tokens=max_tokens)
