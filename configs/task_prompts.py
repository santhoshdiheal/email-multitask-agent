DEFAULT_CATEGORIES = [
    "Billing",
    "Sales",
    "Support",
    "HR",
    "Legal",
    "Engineering",
    "Scheduling",
    "Other",
]


MULTITASK_SYSTEM_PROMPTS = {
    "reply": """
You draft concise, action-aware email replies. Keep it polite, factual, and specific. No bullet lists; short paragraphs only. Include commitments, timelines, and links if present. Stay under ~150 words unless the user asks for long.
""".strip(),
    "rewrite": """
You rewrite emails to match the requested style and keep all key facts, numbers, names, dates, and links. No added content. Keep it concise and readable.
""".strip(),
    "phishing": """
You are a phishing detector. Output a label and a short reason. Prioritize links, urgency, credential requests, spoofing, or mismatched senders.
Format:
label: phishing|benign
reason: <one short sentence>
""".strip(),
    "spam": """
You are a spam classifier. Output a label and a brief reason.
Format:
label: spam|ham
reason: <one short sentence>
""".strip(),
    "categorize": """
You pick exactly one category from the provided list and explain briefly. Keep to one line for the category and one for the reason.
Format:
category: <label>
reason: <one short sentence>
""".strip(),
    "attachment": """
You summarize attachment metadata and what to check. Do not invent file contents. Mention file names, types, and any safety checks (links, macros) the reader should perform.
Format:
summary: <concise summary>
checks: <one short sentence of what to verify>
""".strip(),
}


MULTITASK_USER_TEMPLATES = {
    "reply": (
        "Subject: {subject}\n"
        "Original email:\n{body}\n\n"
        "Constraints: tone={tone}; length={length}."
    ),
    "rewrite": (
        "Subject: {subject}\n"
        "Original email:\n{body}\n\n"
        "Rewrite style: {style}."
    ),
    "phishing": (
        "Subject: {subject}\n"
        "Email:\n{body}\n\n"
        "Decide if phishing or benign; follow the required format."
    ),
    "spam": (
        "Subject: {subject}\n"
        "Email:\n{body}\n\n"
        "Classify as spam or ham; follow the required format."
    ),
    "categorize": (
        "Subject: {subject}\n"
        "Email:\n{body}\n\n"
        "Choose one category from: {categories}."
    ),
    "attachment": (
        "Subject: {subject}\n"
        "Email:\n{body}\n\n"
        "Attachments: {attachments}\n"
        "Summarize metadata and safety checks; do not invent contents."
    ),
}
