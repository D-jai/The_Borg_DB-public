# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Calendar (ICS) File Loader
==========================

This module loads .ics calendar files and converts each VEVENT into
an Interaction with source_type=CALENDAR. Unlike other loaders,
calendar data is STRUCTURED -- it already contains entities (attendees,
locations) and facts (scheduled times, descriptions) in machine-
readable form. Therefore, this loader BYPASSES NLP extraction entirely
and directly creates Entity and Fact objects from the structured data.

Why bypass NLP? Running spaCy NER on "Meeting with Alice at Room 101
on 2026-03-15 at 10:00" would try to figure out what's already known.
The .ics file tells us exactly: attendee=Alice, location=Room 101,
dtstart=2026-03-15T10:00. No guessing needed.

The loader uses the `icalendar` library if available. If the library
is not installed, it logs a warning and returns an empty list (graceful
skip). The icalendar library is NOT a core dependency -- it's optional
for calendar support.

Depends on:
    - icalendar (optional -- graceful skip if not installed)
    - pathlib (file path handling)
    - ctxmtg.models.interaction (Interaction, Entity, Fact, SourceType, EntityType)
    - ctxmtg.storage.id_gen (generate_interaction_id, generate_entity_id, generate_fact_id)
    - ctxmtg.exceptions (IntakeError)

Used by:
    - ctxmtg.ingestion.loaders (registered for .ics extension)
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
    Fact,
    Interaction,
    SourceType,
)
from ctxmtg.storage.id_gen import (
    generate_entity_id,
    generate_fact_id,
    generate_interaction_id,
)

# ---------------------------------------------------------------
# Module-level logger -- logs file metadata, never event content.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.ingestion.loaders.calendar_loader")

# ---------------------------------------------------------------
# Flag indicating whether the icalendar library is available.
# If not, the loader logs a warning and returns empty results.
# ---------------------------------------------------------------
_ICALENDAR_AVAILABLE = False
try:
    import icalendar

    _ICALENDAR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CalendarLoadResult:
    """
    Result of loading a calendar file.

    Contains the Interaction plus directly-created entities and facts
    that bypass the NLP extraction pipeline. The ingestion worker
    should store these entities/facts directly without running extraction.
    """

    interaction: Interaction
    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)


def load_ics_file(file_path: Path) -> list[CalendarLoadResult]:
    """
    Load an .ics calendar file and produce CalendarLoadResult objects.

    Each VEVENT in the calendar becomes one CalendarLoadResult containing:
    - An Interaction with source_type=CALENDAR
    - Entity objects created directly from attendees and locations
    - Fact objects created directly from event relationships

    If the icalendar library is not installed, logs a warning and returns
    an empty list.

    Args:
        file_path: Path to the .ics file to load.

    Returns:
        A list of CalendarLoadResult objects (one per VEVENT).

    Raises:
        IntakeError: If the file cannot be read.
    """
    # Check if the icalendar library is available
    if not _ICALENDAR_AVAILABLE:
        logger.warning(
            "icalendar_not_installed",
            error_code="CTXMTG-ING-005",
            message="Skipping .ics file -- install 'icalendar' package for calendar support",
            file_path=str(file_path),
        )
        return []

    # Validate file existence
    if not file_path.exists():
        logger.error(
            "ics_file_not_found",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"ICS file not found: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    if not file_path.is_file():
        logger.error(
            "ics_path_not_a_file",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
        )
        raise IntakeError(
            f"Path is not a file: {file_path}",
            error_code="CTXMTG-ING-001",
        )

    # Read the raw calendar data
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        logger.error(
            "ics_read_failed",
            error_code="CTXMTG-ING-001",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot read ICS file {file_path}: {exc}",
            error_code="CTXMTG-ING-001",
        ) from exc

    # Parse the iCalendar data
    try:
        cal = icalendar.Calendar.from_ical(raw_bytes)
    except Exception as exc:
        logger.error(
            "ics_parse_failed",
            error_code="CTXMTG-ING-002",
            file_path=str(file_path),
            error=str(exc),
        )
        raise IntakeError(
            f"Cannot parse ICS file {file_path}: {exc}",
            error_code="CTXMTG-ING-002",
        ) from exc

    # Process each VEVENT component
    results: list[CalendarLoadResult] = []
    for component in cal.walk():
        if component.name == "VEVENT":
            result = _process_vevent(component, file_path)
            if result is not None:
                results.append(result)

    logger.info(
        "ics_file_loaded",
        file_path=str(file_path),
        event_count=len(results),
    )

    return results


def _process_vevent(component: Any, file_path: Path) -> CalendarLoadResult | None:
    """
    Process a single VEVENT component into a CalendarLoadResult.

    Extracts event summary, description, start/end times, attendees,
    organizer, and location. Creates Entity and Fact objects directly
    from the structured calendar data.

    Args:
        component: An icalendar VEVENT component.
        file_path: Source file path for metadata.

    Returns:
        A CalendarLoadResult, or None if the event has no useful content.
    """
    # Extract event fields from the VEVENT component
    summary = str(component.get("SUMMARY", "Untitled Event"))
    description = str(component.get("DESCRIPTION", ""))
    location = str(component.get("LOCATION", ""))

    # Parse start and end times
    dtstart = component.get("DTSTART")
    dtend = component.get("DTEND")
    start_time = _parse_ical_datetime(dtstart)
    end_time = _parse_ical_datetime(dtend)

    # Build the interaction content from structured data
    content_parts = [f"Event: {summary}"]
    if description:
        content_parts.append(f"Description: {description}")
    if location:
        content_parts.append(f"Location: {location}")
    if start_time:
        content_parts.append(f"Start: {start_time.isoformat()}")
    if end_time:
        content_parts.append(f"End: {end_time.isoformat()}")

    # Extract attendees
    attendees = _extract_attendees(component)
    if attendees:
        content_parts.append(f"Attendees: {', '.join(attendees)}")

    # Extract organizer
    organizer = _extract_organizer(component)
    if organizer:
        content_parts.append(f"Organizer: {organizer}")

    content = "\n".join(content_parts)
    if not content.strip():
        return None

    # Create the Interaction object
    created_at = start_time or datetime.now(timezone.utc)
    interaction_id = generate_interaction_id("calendar", content)

    participants = list(attendees)
    if organizer and organizer not in participants:
        participants.insert(0, organizer)

    interaction = Interaction(
        id=interaction_id,
        source_type=SourceType.CALENDAR,
        title=summary,
        content=content,
        participants=participants,
        metadata={
            "source_file": str(file_path.name),
            "location": location,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
        },
        created_at=created_at,
    )

    # ---------------------------------------------------------------
    # Create entities directly from structured data (BYPASS NLP).
    # Attendees → PERSON entities, location → LOCATION entity.
    # ---------------------------------------------------------------
    entities: list[Entity] = []
    now = datetime.now(timezone.utc)

    # Create PERSON entities from attendees and organizer
    all_people = list(attendees)
    if organizer and organizer not in all_people:
        all_people.append(organizer)

    for person_name in all_people:
        entity_id = generate_entity_id(interaction_id, person_name, "person")
        entities.append(
            Entity(
                id=entity_id,
                interaction_id=interaction_id,
                name=person_name,
                entity_type=EntityType.PERSON,
                confidence=1.0,
                provenance="calendar:ics_loader",
                context={"summary": f"Attendee at {summary}"},
                tags={"source_type": "calendar"},
                created_at=now,
            )
        )

    # Create LOCATION entity if location is present
    if location:
        loc_entity_id = generate_entity_id(interaction_id, location, "location")
        entities.append(
            Entity(
                id=loc_entity_id,
                interaction_id=interaction_id,
                name=location,
                entity_type=EntityType.LOCATION,
                confidence=1.0,
                provenance="calendar:ics_loader",
                context={"summary": f"Location of {summary}"},
                tags={"source_type": "calendar"},
                created_at=now,
            )
        )

    # Create EVENT entity for the event itself
    event_entity_id = generate_entity_id(interaction_id, summary, "event")
    entities.append(
        Entity(
            id=event_entity_id,
            interaction_id=interaction_id,
            name=summary,
            entity_type=EntityType.EVENT,
            confidence=1.0,
            provenance="calendar:ics_loader",
            context={"summary": f"Calendar event: {summary}"},
            tags={"source_type": "calendar"},
            created_at=now,
        )
    )

    # ---------------------------------------------------------------
    # Create facts directly from structured data (BYPASS NLP).
    # Attendees → invited_to event, event → scheduled_at time.
    # ---------------------------------------------------------------
    facts: list[Fact] = []

    # Fact: event scheduled_at start_time
    if start_time:
        fact_id = generate_fact_id(event_entity_id, "scheduled_at", start_time.isoformat())
        facts.append(
            Fact(
                id=fact_id,
                interaction_id=interaction_id,
                subject_entity_id=event_entity_id,
                predicate="scheduled_at",
                object_literal=start_time.isoformat(),
                confidence=1.0,
                source_span=f"DTSTART: {start_time.isoformat()}",
                created_at=now,
            )
        )

    # Fact: each attendee invited_to the event
    for person_name in all_people:
        person_entity_id = generate_entity_id(interaction_id, person_name, "person")
        fact_id = generate_fact_id(person_entity_id, "invited_to", event_entity_id)
        facts.append(
            Fact(
                id=fact_id,
                interaction_id=interaction_id,
                subject_entity_id=person_entity_id,
                predicate="invited_to",
                object_entity_id=event_entity_id,
                confidence=1.0,
                source_span=f"ATTENDEE: {person_name}",
                created_at=now,
            )
        )

    # Fact: event located_at location
    if location:
        loc_eid = generate_entity_id(interaction_id, location, "location")
        fact_id = generate_fact_id(event_entity_id, "located_at", loc_eid)
        facts.append(
            Fact(
                id=fact_id,
                interaction_id=interaction_id,
                subject_entity_id=event_entity_id,
                predicate="located_at",
                object_entity_id=loc_eid,
                confidence=1.0,
                source_span=f"LOCATION: {location}",
                created_at=now,
            )
        )

    return CalendarLoadResult(
        interaction=interaction,
        entities=entities,
        facts=facts,
    )


def _extract_attendees(component: Any) -> list[str]:
    """
    Extract attendee names/emails from a VEVENT component.

    Parses ATTENDEE properties which may be "mailto:user@example.com"
    or contain a CN (Common Name) parameter.

    Args:
        component: An icalendar VEVENT component.

    Returns:
        A list of attendee names or email addresses.
    """
    attendees: list[str] = []
    raw_attendees = component.get("ATTENDEE")

    if raw_attendees is None:
        return attendees

    # ATTENDEE can be a single value or a list
    if not isinstance(raw_attendees, list):
        raw_attendees = [raw_attendees]

    for attendee in raw_attendees:
        # Try to get the CN (Common Name) parameter first
        name = str(attendee.params.get("CN", "")) if hasattr(attendee, "params") else ""

        if not name:
            # Fallback: extract from mailto: URI
            addr = str(attendee)
            name = addr.replace("mailto:", "").replace("MAILTO:", "").strip()

        if name:
            attendees.append(name)

    return attendees


def _extract_organizer(component: Any) -> str | None:
    """
    Extract the organizer name/email from a VEVENT component.

    Args:
        component: An icalendar VEVENT component.

    Returns:
        The organizer name or email, or None if not present.
    """
    organizer = component.get("ORGANIZER")
    if organizer is None:
        return None

    # Try CN parameter first
    name = str(organizer.params.get("CN", "")) if hasattr(organizer, "params") else ""
    if name:
        return name

    # Fallback: extract from mailto: URI
    addr = str(organizer)
    return addr.replace("mailto:", "").replace("MAILTO:", "").strip() or None


def _parse_ical_datetime(dt_prop: Any) -> datetime | None:
    """
    Parse an iCalendar date/datetime property into a Python datetime.

    Handles both date and datetime properties, with or without timezone.

    Args:
        dt_prop: An icalendar DTSTART/DTEND property.

    Returns:
        A timezone-aware datetime, or None if parsing fails.
    """
    if dt_prop is None:
        return None

    try:
        dt = dt_prop.dt
        if isinstance(dt, datetime):
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        # It's a date, not datetime -- convert to datetime at midnight
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    except (AttributeError, ValueError):
        return None
