# Copyright (c) 2026 Aliud Inquisito Inc. / Dhananjay Raol. All rights reserved.
# Proprietary and confidential. See LICENSE file for details.
"""
Web Route Modules
=================

Each submodule provides a FastAPI APIRouter for a section of the
command center:
    - auth_routes.py   : login / logout / password setup
    - dashboard.py     : main dashboard (all locals, hive overview)
    - local.py         : per-local management (farming, LLM, profiles)
    - completions.py   : OpenAI-compatible /v1/chat/completions
"""
