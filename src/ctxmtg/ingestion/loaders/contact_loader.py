# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Contact (VCF) File Loader
==========================

This module loads .vcf vCard files and converts each contact into
an Interaction with source_type=CONTACT. Like the calendar loader,
contact data is already structured -- names, organizations, emails,
and phone numbers are in machine-readable fields. Therefore, this
loader BYPASSES NLP extraction and directly creates Entity objects.

Why bypass NLP? A vCard tells us explicitly: FN=Alice Smith,
ORG=Acme Corp, EMAIL=alice@acme.com. There's nothing to "extract"
-- the data is already structured.

The loader uses the `vobject` library if available. If the library
is not installed, it logs a warning and returns an empty list
(graceful skip). The vobject library is NOT a core dependency.

Depends on:
    - vobject (optional -- graceful skip if not installed)
    - pathlib (file path handling)
    - ctxmtg.models.interaction (Interaction, Entity, SourceType, EntityType)
    - ctxmtg.storage.id_gen (generate_interaction_id, generate_entity_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .vcf extension)
    - ctxmtg.ingestion.worker (loads files, uses bypass_nlp flag)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ctxmtg.exceptions import IntakeError
from ctxmtg.models.interaction import (
    Entity,
    EntityType,
    Interaction,
    SourceType,
)
from ctxmtg.storage.id_gen import (
    generate_entity_id,
    generate_interaction_id,
)

# ---------------------------------------------------------------
# Module-level logger -- logs file metadata, never contact details.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders.contact_loader")

# ---------------------------------------------------------------
# Flag indicating whether the vobject library is available.
# If not, the loader logs a warning and returns empty results.
# ---------------------------------------------------------------
_VOBJECT_AVAILABLE = False
try:
    import vobject

    _VOBJECT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ContactLoadResult:
    """
    Result of loading a contact file.

    Contains the Interaction plus directly-created entities that bypass
    the NLP extraction pipeline. The ingestion worker should store
    these entities directly without running extraction.
    """

    interaction: Interaction
    entities: list[Entity] = field(default_factory=list)


def load_vcf_file(file_path: Path) -> list[ContactLoadResult]:
    """
    Load a .vcf vCard file and produce ContactLoadResult objects.

    Each vCard in the file becomes one ContactLoadResult containing:
    - An Interaction with source_type=CONTACT
    - Entity objects created directly from contact fields

    If the vobject library is not installed, logs a warning and returns
    an empty list.

    Args:
        file_path: Path to the .vcf file to load.

    Returns:
        A list of ContactLoadResult objects (one per vCard).

    Raises:
        IntakeError: If the file cannot be read.
    """
    # Check if the vobject library is available
    if not _VOBJECT_AVAILABLE:
        logger.warning(
            "vobject_not_installed",
            error_code="CTXMTG-ING-005",
            message="Skipping .vcf file -- install 'vobject' package for contact support",
            file_path=str(file_path),
        )
        return []

    # Validate file existence
    if not file_path.exists():
        logger.error("vcf_file_not_found", error_code="CTXMTG-ING-001", file_path=str(file_path))
    raise IntakeError(f"VCF file not found: {file_path}", error_code="CTXMTG-ING-001")

    if not file_path.is_file():
        logger.error("vcf_path_not_a_file", error_code="CTXMTG-ING-001", file_path=str(file_path))
    raise IntakeError(f"Path is not a file: {file_path}", error_code="CTXMTG-ING-001")

    # Read the VCF file content
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_text = file_path.read_text(encoding="latin-1")
        except OSError as exc:
            logger.error("vcf_read_failed", error_code="CTXMTG-ING-001", file_path=str(file_path), error=str(exc))
        raise IntakeError(f"Cannot read VCF file {file_path}: {exc}", error_code="CTXMTG-ING-001") from exc
    except OSError as exc:
        logger.error("vcf_read_failed", error_code="CTXMTG-ING-001", file_path=str(file_path), error=str(exc))
        raise IntakeError(f"Cannot read VCF file {file_path}: {exc}", error_code="CTXMTG-ING-001") from exc

    if not raw_text.strip():
        logger.error("vcf_file_empty", error_code="CTXMTG-ING-001", file_path=str(file_path))
    raise IntakeError(f"VCF file is empty: {file_path}", error_code="CTXMTG-ING-001")

    # Parse all vCards in the file
    results: list[ContactLoadResult] = []
    try:
        for vcard in vobject.readComponents(raw_text):
            result = _process_vcard(vcard, file_path)
            if result is not None:
                results.append(result)
    except Exception as exc:
        logger.error("vcf_parse_failed", error_code="CTXMTG-ING-002", file_path=str(file_path), error=str(exc))
    raise IntakeError(f"Cannot parse VCF file {file_path}: {exc}", error_code="CTXMTG-ING-002") from exc

    logger.info(
        "vcf_file_loaded",
        file_path=str(file_path),
        contact_count=len(results),
    )

    return results


def _process_vcard(vcard: Any, file_path: Path) -> ContactLoadResult | None:
    """
    Process a single vCard into a ContactLoadResult.

    Extracts contact name, organization, emails, phone numbers, and
    builds Entity objects directly from the structured data.

    Args:
        vcard: A vobject vCard component.
        file_path: Source file path for metadata.

    Returns:
        A ContactLoadResult, or None if the vCard has no useful data.
    """
    # Extract the full name (FN is required by vCard spec)
    full_name = ""
    if hasattr(vcard, "fn"):
        full_name = str(vcard.fn.value).strip()

    if not full_name and hasattr(vcard, "n"):
        # Try to build from N (structured name) field
        name_parts = vcard.n.value
        full_name = f"{name_parts.given} {name_parts.family}".strip()

    if not full_name:
        return None  # Skip vCards without a name

    # Extract organization
    org = ""
    if hasattr(vcard, "org"):
        org_val = vcard.org.value
        org = (org_val[0] if org_val else "") if isinstance(org_val, list) else str(org_val)
        org = org.strip()

    # Extract email addresses
    emails: list[str] = []
    if hasattr(vcard, "email"):
        email_val = vcard.email
        if isinstance(email_val, list):
            emails = [str(e.value).strip() for e in email_val if e.value]
        else:
            if email_val.value:
                emails = [str(email_val.value).strip()]

    # Extract phone numbers
    phones: list[str] = []
    if hasattr(vcard, "tel"):
        tel_val = vcard.tel
        if isinstance(tel_val, list):
            phones = [str(t.value).strip() for t in tel_val if t.value]
        else:
            if tel_val.value:
                phones = [str(tel_val.value).strip()]

    # Extract title/role
    title = ""
    if hasattr(vcard, "title"):
        title = str(vcard.title.value).strip()

    # Build interaction content from structured fields
    content_parts = [f"Contact: {full_name}"]
    if org:
        content_parts.append(f"Organization: {org}")
    if title:
        content_parts.append(f"Title: {title}")
    if emails:
        content_parts.append(f"Email: {', '.join(emails)}")
    if phones:
        content_parts.append(f"Phone: {', '.join(phones)}")

    content = "\n".join(content_parts)

    # Create the Interaction object
    interaction_id = generate_interaction_id("contact", content)
    now = datetime.now(timezone.utc)

    interaction = Interaction(
        id=interaction_id,
        source_type=SourceType.CONTACT,
        title=f"Contact: {full_name}",
        content=content,
        participants=[full_name],
        metadata={
            "source_file": str(file_path.name),
            "organization": org,
        },
        created_at=now,
    )

    # ---------------------------------------------------------------
    # Create entities directly from structured data (BYPASS NLP).
    # Name → PERSON entity, Org → ORG entity.
    # ---------------------------------------------------------------
    entities: list[Entity] = []

    # PERSON entity with contact details as tags
    person_tags: dict[str, str] = {"source_type": "contact"}
    if emails:
        person_tags["email"] = emails[0]  # Primary email
    if phones:
        person_tags["phone"] = phones[0]  # Primary phone
    if org:
        person_tags["organization"] = org
    if title:
        person_tags["title"] = title

    person_entity_id = generate_entity_id(interaction_id, full_name, "person")
    entities.append(
        Entity(
            id=person_entity_id,
            interaction_id=interaction_id,
            name=full_name,
            entity_type=EntityType.PERSON,
            confidence=1.0,
            provenance="contact:vcf_loader",
            context={"summary": f"Contact: {full_name}" + (f" at {org}" if org else "")},
            tags=person_tags,
            created_at=now,
        )
    )

    # ORG entity if organization is present
    if org:
        org_entity_id = generate_entity_id(interaction_id, org, "org")
        entities.append(
            Entity(
                id=org_entity_id,
                interaction_id=interaction_id,
                name=org,
                entity_type=EntityType.ORG,
                confidence=1.0,
                provenance="contact:vcf_loader",
                context={"summary": f"Organization of {full_name}"},
                tags={"source_type": "contact"},
                created_at=now,
            )
        )

    return ContactLoadResult(
        interaction=interaction,
        entities=entities,
    )
