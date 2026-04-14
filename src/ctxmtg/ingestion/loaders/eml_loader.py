# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
EML Email File Loader
=====================

This module loads .eml email files and converts them into Interaction
objects. It uses Python's stdlib `email` module (no external deps) to
parse MIME messages, extracting:
    - From → participants (sender)
    - To/CC → participants (recipients)
    - Subject → title
    - Body (text/plain or text/html) → content
    - Date → created_at
    - Attachments → metadata stubs: [ATTACHMENT: name, type, size]

Attachments are NOT stored as binary content. Instead, each attachment
becomes a metadata stub appended to the interaction content. This keeps
storage lean while preserving the knowledge that an attachment existed
(its name, type, and size may still be relevant context).

Depends on:
    - email (Python stdlib -- MIME message parsing)
    - datetime (timestamp handling)
    - ctxmtg.models.interaction (Interaction, SourceType)
    - ctxmtg.storage.id_gen (generate_interaction_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .eml extension)
    - ctxmtg.ingestion.worker (loads files before extraction)
"""

from __future__ import annotations

import email
import email.policy
import email.utils
from datetime import datetime, timezone
from pathlib import Path

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import Interaction, SourceType
from ctxmtg.storage.id_gen import generate_interaction_id

# ---------------------------------------------------------------
# Module-level logger -- logs file metadata, never email content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders.eml_loader")


def load_eml_file(file_path: Path) -> Interaction:
    """
    Load an .eml email file and produce an Interaction object.

    Parses the MIME message structure to extract sender, recipients,
    subject, body text, date, and attachment metadata. Binary attachment
    content is replaced with metadata stubs.

    Args:
        file_path: Path to the .eml file to load.

    Returns:
        An Interaction with source_type=EMAIL, containing the email
        body and attachment metadata stubs.

    Raises:
        IntakeError: If the file cannot be read or parsed.
    """
    # Validate file existence
    if not file_path.exists():
        logger.error(
            "eml_file_not_found",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"EML file not found: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    if not file_path.is_file():
        logger.error(
            "eml_path_not_a_file",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Path is not a file: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        logger.error(
            "eml_read_failed",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot read EML file {file_path}: {exc}",
            error_code="CTXMTG-ING-001",
        ) from exc

    try:
        msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)
    except Exception as exc:
        logger.error(
            "eml_parse_failed",
            error_code="CTXMTG-ING-002",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot parse EML file {file_path}: {exc}",
            error_code="CTXMTG-ING-002",
        ) from exc

    # ---------------------------------------------------------------
    # Extract header fields
    # ---------------------------------------------------------------

    # From: sender address (may be "Name <email>" format)
    sender = str(msg.get("From", ""))

    # To: and CC: recipients
    to_header = str(msg.get("To", ""))
    cc_header = str(msg.get("CC", ""))

    # Build participants list from all address fields
    participants = _extract_participants(sender, to_header, cc_header)

    # Subject line → title
    subject = str(msg.get("Subject", ""))
    title = subject if subject else file_path.stem

    # Date header → created_at
    created_at = _parse_date(msg.get("Date", ""))

    # ---------------------------------------------------------------
    # Extract body text (prefer text/plain, fallback to text/html)
    # ---------------------------------------------------------------
    body = _extract_body(msg)

    # ---------------------------------------------------------------
    # Extract attachment metadata stubs (no binary content stored)
    # ---------------------------------------------------------------
    attachment_stubs = _extract_attachment_stubs(msg)

    # Combine body and attachment stubs into content
    content_parts = [body] if body else []
    if attachment_stubs:
        content_parts.append("\n\n--- Attachments ---")
        content_parts.extend(attachment_stubs)

    content = "\n".join(content_parts).strip()

    if not content:
        logger.error(
            "eml_no_content",
            error_code="CTXMTG-ING-003",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"EML file has no extractable content: {file_path}",
            error_code="CTXMTG-ING-003",
        )

    # Generate deterministic interaction ID
    interaction_id = generate_interaction_id("email", content)

    # Build metadata dict with email-specific info
    metadata: dict[str, object] = {
        "source_file": str(file_path.name),
        "from": sender,
        "subject": subject,
        "attachment_count": len(attachment_stubs),
    }

    # Add CC info to metadata if present
    if cc_header:
        metadata["cc"] = cc_header

    logger.info(
        "eml_file_loaded",
        file_path=str(file_path),
        participant_count=len(participants),
        attachment_count=len(attachment_stubs),
    )

    return Interaction(
        id=interaction_id,
        source_type=SourceType.EMAIL,
        title=title,
        content=content,
        participants=participants,
        metadata=metadata,
        created_at=created_at,
    )


def _extract_participants(sender: str, to_header: str, cc_header: str) -> list[str]:
    """
    Extract participant names/emails from email headers.

    Parses "Name <email>" format and plain email addresses. Deduplicates
    participants and returns a clean list of unique names/addresses.

    Args:
        sender: The From: header value.
        to_header: The To: header value.
        cc_header: The CC: header value.

    Returns:
        A deduplicated list of participant names or email addresses.
    """
    participants: list[str] = []
    seen: set[str] = set()

    # Parse all address headers together
    all_addresses = f"{sender}, {to_header}, {cc_header}"
    for _name, addr in email.utils.getaddresses([all_addresses]):
        # Prefer the display name, fall back to the email address
        participant = _name.strip() if _name.strip() else addr.strip()
        if participant and participant.lower() not in seen:
            seen.add(participant.lower())
            participants.append(participant)

    return participants


def _parse_date(date_str: str | object) -> datetime:
    """
    Parse an email Date header into a datetime object.

    Handles common email date formats (RFC 2822). Falls back to
    current UTC time if parsing fails.

    Args:
        date_str: The Date header value (may be a HeaderMissing object).

    Returns:
        A timezone-aware datetime object.
    """
    if not date_str or not isinstance(date_str, str):
        return datetime.now(timezone.utc)

    try:
        # email.utils.parsedate_to_datetime handles RFC 2822 dates
        parsed = email.utils.parsedate_to_datetime(str(date_str))
        # Ensure timezone-aware
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        # Parsing failed -- use current time as fallback
        return datetime.now(timezone.utc)


def _extract_body(msg: email.message.Message) -> str:
    """
    Extract the text body from a MIME email message.

    Prefers text/plain parts. Falls back to text/html (with tag stripping)
    if no plain text is available.

    Args:
        msg: The parsed email message.

    Returns:
        The email body as plain text, or empty string if no body found.
    """
    # Try to get the plain text body first
    body = msg.get_body(preferencelist=("plain",))
    if body:
        content = body.get_content()
        if isinstance(content, str):
            return content.strip()

    # Fallback: try HTML body with basic tag stripping
    html_body = msg.get_body(preferencelist=("html",))
    if html_body:
        content = html_body.get_content()
        if isinstance(content, str):
            # Basic HTML tag stripping (no external dependency)
            import re

            text = re.sub(r"<[^>]+>", " ", content)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

    # Walk all parts as last resort
    for part in msg.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or "utf-8"
                try:
                    return payload.decode(charset).strip()
                except (UnicodeDecodeError, LookupError):
                    return payload.decode("latin-1").strip()

    return ""


def _extract_attachment_stubs(msg: email.message.Message) -> list[str]:
    """
    Extract attachment metadata stubs from a MIME message.

    Each attachment becomes a stub: [ATTACHMENT: filename, mime_type, size].
    Binary content is NOT extracted -- only metadata about the attachment.

    Args:
        msg: The parsed email message.

    Returns:
        A list of attachment stub strings.
    """
    stubs: list[str] = []

    for part in msg.walk():
        # Skip non-attachment parts
        content_disposition = str(part.get("Content-Disposition", ""))
        if "attachment" not in content_disposition.lower():
            continue

        # Get attachment filename
        filename = part.get_filename() or "unnamed"

        # Get MIME type
        mime_type = part.get_content_type() or "application/octet-stream"

        # Get size from payload (decoded)
        payload = part.get_payload(decode=True)
        if payload:
            size_bytes = len(payload)
            size_str = _format_size(size_bytes)
        else:
            size_str = "unknown"

        # Build the metadata stub
        stub = f"[ATTACHMENT: {filename}, {mime_type}, {size_str}]"
        stubs.append(stub)

    return stubs


def _format_size(size_bytes: int) -> str:
    """
    Format a byte count into a human-readable size string.

    Args:
        size_bytes: The number of bytes.

    Returns:
        A human-readable string (e.g., "5.2MB", "340KB", "128B").
    """
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f}MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f}KB"
    else:
        return f"{size_bytes}B"
