# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Domain Profile Loader
=====================

This module provides the ProfileLoader class, which reads YAML
configuration files from the profiles/ directory, validates them
against the DomainProfile Pydantic model, and returns fully
initialized DomainProfile instances.

The loader is the entry point for all profile-related operations.
Every pipeline component that needs domain-specific configuration
(NER entity types, farming schedule, embedding model, intake rules)
gets it through a DomainProfile loaded by this class.

How it works:
    1. Caller requests a profile by name (e.g., "general", "legal").
    2. Loader resolves the name to a YAML file in the profiles/ directory.
    3. YAML is parsed and validated against the DomainProfile Pydantic model.
    4. Missing fields get sensible defaults (from the Pydantic model).
    5. Invalid data raises ProfileError with a clear, actionable message.

The loader supports two resolution strategies:
    - Name-based: ProfileLoader.load("general") → profiles/general.yaml
    - Path-based: ProfileLoader.load_from_path("/custom/path/my_profile.yaml")

Depends on:
    - pyyaml (YAML parsing)
    - pydantic (validation via DomainProfile model)
    - structlog (structured logging)
    - ctxmtg.models.profile (DomainProfile and sub-models)
    - ctxmtg.exceptions (ProfileError for error reporting)

Used by:
    - ctxmtg.ingestion.worker (loads profile before extraction)
    - ctxmtg.query.executor (loads profile for query planning)
    - ctxmtg.cli (loads profile specified by user or config)
    - ctxmtg.config.settings (resolves profile_name to DomainProfile)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from ctxmtg.exceptions import ProfileError
from ctxmtg.models.profile import DomainProfile

# ---------------------------------------------------------------
# Module-level logger: all log entries from this module are tagged
# with "ctxmtg.profile.loader" for easy filtering.
# ---------------------------------------------------------------
logger = structlog.get_logger("ctxmtg.profile.loader")

# ---------------------------------------------------------------
# Default profiles directory: located at the project root under
# profiles/. This is where bundled profiles (general, legal,
# personal) live. Users can also specify custom paths.
# ---------------------------------------------------------------
_DEFAULT_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "profiles"


class ProfileLoader:
    """
    Loads and validates domain profiles from YAML files.

    The ProfileLoader is the single entry point for getting a
    DomainProfile instance. It handles file resolution, YAML parsing,
    Pydantic validation, and default value population.

    Why a class with static/class methods instead of plain functions?
    Because we want a clear namespace (ProfileLoader.load) and the
    ability to override the profiles directory for testing or custom
    deployments (ProfileLoader.load with profiles_dir parameter).

    Usage:
        # Load a bundled profile by name:
        profile = ProfileLoader.load("general")

        # Load from a custom path:
        profile = ProfileLoader.load_from_path("/my/profiles/custom.yaml")

        # List available profiles in the default directory:
        names = ProfileLoader.list_profiles()

        # Load with a custom profiles directory:
        profile = ProfileLoader.load("legal", profiles_dir=Path("/custom/dir"))
    """

    @staticmethod
    def load(
        name: str,
        profiles_dir: Path | None = None,
    ) -> DomainProfile:
        """
        Load a domain profile by name from the profiles directory.

        Resolves the profile name to a YAML file path by appending
        ".yaml" and looking in the profiles directory. The YAML is
        parsed and validated against the DomainProfile model. Any
        missing fields get Pydantic default values automatically.

        Args:
            name: Profile name without extension (e.g., "general",
                  "legal", "personal"). Case-sensitive.
            profiles_dir: Optional override for the profiles directory.
                          Defaults to the bundled profiles/ directory
                          at the project root.

        Returns:
            A fully validated DomainProfile instance.

        Raises:
            ProfileError: If the file is not found, the YAML is
                          invalid, or validation fails.
        """
        # Determine which directory to look in.
        # Default: the bundled profiles/ directory alongside the source.
        directory = profiles_dir if profiles_dir is not None else _DEFAULT_PROFILES_DIR

        # Build the full path to the YAML file.
        # Profile names map directly to filenames: "general" → "general.yaml".
        yaml_path = directory / f"{name}.yaml"

        # Log the load attempt for debugging and auditing.
        logger.info(
            "profile_load_requested",
            profile_name=name,
            profiles_dir=str(directory),
            yaml_path=str(yaml_path),
        )

        # Delegate to the path-based loader for actual file I/O and validation.
        return ProfileLoader.load_from_path(yaml_path)

    @staticmethod
    def load_from_path(yaml_path: Path | str) -> DomainProfile:
        """
        Load a domain profile from a specific YAML file path.

        This is the core loading method. It handles:
        1. File existence check (raises ProfileError if missing).
        2. YAML parsing (raises ProfileError on syntax errors).
        3. Pydantic validation (raises ProfileError on schema violations).
        4. Default value population (via Pydantic model defaults).

        Args:
            yaml_path: Absolute or relative path to the YAML file.
                       Can be a string or Path object.

        Returns:
            A fully validated DomainProfile instance.

        Raises:
            ProfileError: If the file is missing, YAML is malformed,
                          or the data fails DomainProfile validation.
        """
        # Normalize to a Path object for consistent path handling.
        path = Path(yaml_path)

        # -------------------------------------------------------
        # Step 1: Check that the file exists.
        # Give a clear error message with the full path so the user
        # knows exactly which file is missing and where to create it.
        # -------------------------------------------------------
        if not path.exists():
            logger.error(
                "profile_not_found",
                error_code="CTXMTG-PRF-001",
                yaml_path=str(path),
            )
            raise ProfileError(
                f"Profile file not found: {path}. "
                f"Available profiles are in the profiles/ directory. "
                f"Use ProfileLoader.list_profiles() to see available names.",
                error_code="CTXMTG-PRF-001",
            )

        # -------------------------------------------------------
        # Step 2: Check that the path points to a file (not a directory).
        # -------------------------------------------------------
        if not path.is_file():
            logger.error(
                "profile_path_not_a_file",
                error_code="CTXMTG-PRF-001",
                yaml_path=str(path),
            )
            raise ProfileError(
                f"Profile path is not a file: {path}. "
                f"Expected a YAML file with profile configuration.",
                error_code="CTXMTG-PRF-001",
            )

        # -------------------------------------------------------
        # Step 3: Read and parse the YAML content.
        # yaml.safe_load prevents arbitrary code execution from
        # malicious YAML (no Python object deserialization).
        # -------------------------------------------------------
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(
                "profile_file_read_failed",
                error_code="CTXMTG-PRF-001",
                yaml_path=str(path),
                error=str(e),
            )
            raise ProfileError(
                f"Cannot read profile file {path}: {e}",
                error_code="CTXMTG-PRF-001",
            ) from e

        # Parse YAML into a Python dict. safe_load returns None for
        # empty files, which we handle below.
        try:
            raw_data = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            logger.error(
                "profile_yaml_parse_failed",
                error_code="CTXMTG-PRF-002",
                yaml_path=str(path),
                error=str(e),
            )
            raise ProfileError(
                f"Invalid YAML in profile file {path}: {e}",
                error_code="CTXMTG-PRF-002",
            ) from e

        # -------------------------------------------------------
        # Step 4: Validate the parsed data.
        # An empty file or one that parses to None is not a valid profile.
        # -------------------------------------------------------
        if raw_data is None:
            logger.error(
                "profile_file_empty",
                error_code="CTXMTG-PRF-002",
                yaml_path=str(path),
            )
            raise ProfileError(
                f"Profile file is empty: {path}. "
                f"A profile must have at least 'name' and 'version' fields.",
                error_code="CTXMTG-PRF-002",
            )

        # The YAML must parse to a dictionary (not a list, string, etc.).
        if not isinstance(raw_data, dict):
            logger.error(
                "profile_not_a_mapping",
                error_code="CTXMTG-PRF-002",
                yaml_path=str(path),
                actual_type=type(raw_data).__name__,
            )
            raise ProfileError(
                f"Profile file must contain a YAML mapping (dict), "
                f"got {type(raw_data).__name__} in {path}.",
                error_code="CTXMTG-PRF-002",
            )

        # -------------------------------------------------------
        # Step 5: Construct the DomainProfile from the parsed data.
        # Pydantic handles validation and fills in defaults for any
        # fields not present in the YAML. If required fields are
        # missing (name, version), Pydantic raises ValidationError.
        # -------------------------------------------------------
        profile = ProfileLoader._validate_profile_data(raw_data, path)

        # Log successful load with key profile metadata.
        logger.info(
            "profile_loaded",
            profile_name=profile.name,
            profile_version=profile.version,
            entity_types_count=len(profile.ner.entity_types),
            stages_count=len(profile.stages),
        )

        return profile

    @staticmethod
    def _validate_profile_data(
        raw_data: dict[str, Any],
        source_path: Path,
    ) -> DomainProfile:
        """
        Validate parsed YAML data against the DomainProfile Pydantic model.

        This internal method separates validation logic from file I/O
        for easier testing. It takes a dict (already parsed from YAML)
        and returns a DomainProfile or raises ProfileError.

        Args:
            raw_data: Dictionary parsed from YAML. Expected to have
                      keys matching DomainProfile fields (name, version,
                      stages, ner, farming, embedding, intake).
            source_path: Path of the source YAML file (for error messages).

        Returns:
            A validated DomainProfile instance.

        Raises:
            ProfileError: If validation fails (missing required fields,
                          wrong types, constraint violations).
        """
        try:
            # Pydantic's model_validate handles:
            # - Type coercion (e.g., string "0.1" → float 0.1)
            # - Default value filling (missing fields get model defaults)
            # - Constraint checking (e.g., confidence between 0.0 and 1.0)
            # - Nested model construction (StageConfig, NERConfig, etc.)
            profile = DomainProfile.model_validate(raw_data)
        except ValidationError as e:
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            logger.error(
                "profile_validation_failed",
                error_code="CTXMTG-PRF-003",
                yaml_path=str(source_path),
                error=error_details,
            )
            raise ProfileError(
                f"Profile validation failed for {source_path}: {error_details}",
                error_code="CTXMTG-PRF-003",
            ) from e

        return profile

    @staticmethod
    def list_profiles(
        profiles_dir: Path | None = None,
    ) -> list[str]:
        """
        List all available profile names in the profiles directory.

        Scans the directory for .yaml files and returns their names
        (without the .yaml extension). Useful for CLI tab-completion
        and the `ctxmtg profile --list` command.

        Args:
            profiles_dir: Optional override for the profiles directory.
                          Defaults to the bundled profiles/ directory.

        Returns:
            Sorted list of profile names (e.g., ["general", "legal", "personal"]).
        """
        # Determine which directory to scan.
        directory = profiles_dir if profiles_dir is not None else _DEFAULT_PROFILES_DIR

        # If the directory doesn't exist, return an empty list.
        # This can happen if the project isn't installed from source.
        if not directory.is_dir():
            logger.warning(
                "profiles_dir_not_found",
                error_code="CTXMTG-PRF-001",
                profiles_dir=str(directory),
            )
            return []

        # Collect all .yaml files and strip the extension to get profile names.
        # Sort alphabetically for consistent ordering.
        profile_names = sorted(
            f.stem for f in directory.iterdir() if f.is_file() and f.suffix == ".yaml"
        )

        logger.debug(
            "profiles_listed",
            profiles_dir=str(directory),
            count=len(profile_names),
            names=profile_names,
        )

        return profile_names
