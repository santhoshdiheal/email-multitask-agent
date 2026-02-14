"""Placeholder email provider integrations."""
from datetime import datetime
from typing import List, Dict


def fetch_emails_imap(limit: int = 3) -> List[Dict[str, str]]:
    """
    Placeholder IMAP fetcher.
    Replace with real IMAP or Gmail API integration later.
    """
    sample_body = (
        "Hi team,\n\nPlease review the attached report and share feedback by Friday."
        " Also note the updated dashboard link: https://example.com/dashboard\n\nThanks!"
    )
    return [
        {
            "id": f"stub-{i}",
            "subject": f"Sample message {i+1}",
            "body": sample_body,
            "received_at": datetime.utcnow().isoformat(),
        }
        for i in range(limit)
    ]


def explain_future_integration() -> str:
    return (
        "To integrate Gmail later, use the Gmail API (OAuth2) or IMAP with app passwords. "
        "Implement token storage, incremental sync, and MIME parsing before calling the cleaner."
    )
