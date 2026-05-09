import re


def mask_pii(text):
    # Email
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text
    )

    # Phone number
    text = re.sub(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[PHONE_REDACTED]",
        text
    )

    # SSN
    text = re.sub(
        r"\b\d{3}-\d{2}-\d{4}\b",
        "[SSN_REDACTED]",
        text
    )

    # Credit card-like number
    text = re.sub(
        r"\b(?:\d[ -]*?){13,16}\b",
        "[CARD_REDACTED]",
        text
    )

    return text