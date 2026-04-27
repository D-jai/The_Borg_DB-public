# runtimechange.md

> Canonical record of the runtime-relocation change that landed in
> v0.7.1. Read this to understand why runtime data lives at
> `<project_root>/.runtime/` and how the resolver works.

## Why this change

Before v0.7.1, every runtime artifact -- the SQLite knowledge
database, the LanceDB vector store, the inbox/processed/outbox
directories, `web_auth.json`, `hive_web_auth.json`, the per-role
LLM `.env` file, evaluation snapshots, `archive.db` -- lived under
the user's home directory at `~/.ctxmtg/`.

That single shared location had three persistent failure modes:

1. **Two clones of the source tree shared one runtime store.**
   Cloning `The_Borg_DB-public` once and again as
   `The_Borg_DB-public2` caused both checkouts to read and write
   the same `~/.ctxmtg/knowledge.db`, the same vector store, the
   same outbox. Two projects, one filesystem state -- silent
   corruption was a question of when, not if.
2. **The multi-instance escape hatch was a per-environment
   `CTXMTG_HOME` variable** that had to be set in every shell, in
   every cron job, in every IDE run config. Forgetting it once
   reverted to the shared default and contaminated the canonical
   store.
3. **No unit-test isolation.** Every test run that didn't go
   through the temp-dir helpers polluted the developer's real
   data.

The fix is a single-rule resolver and a single environment
variable.

## The resolver

`src/ctxmtg/paths.py` is the leaf module that resolves runtime
locations. It has one public entry point and a fixed set of
per-artifact helpers. The resolver:

1. If `CTXMTG_DATA_ROOT` is set, expand `~`, `os.path.realpath()`,
   and use that.
2. Otherwise compute `Path(__file__).resolve().parents[2] / ".runtime"`.
   `parents[2]` walks `paths.py -> ctxmtg/ -> src/ -> project_root`.

Resolution is `lru_cache(maxsize=1)`-memoised so the per-call cost
is one dict lookup. Tests that need to flip the env mid-process
call `paths.get_data_root.cache_clear()`.

```python
@lru_cache(maxsize=1)
def get_data_root() -> Path:
    env = os.environ.get("CTXMTG_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / ".runtime").resolve()
```

Each artifact gets its own getter:

| Helper | Returns |
|---|---|
| `get_db_path()` | `<root>/knowledge.db` |
| `get_vector_path()` | `<root>/vectors/` |
| `get_inbox_path()` | `<root>/inbox/` |
| `get_processed_path()` | `<root>/processed/` |
| `get_hive_db_path()` | `<root>/hive.db` |
| `get_hive_vector_path()` | `<root>/hive_vectors/` |
| `get_outbox_path()` | `<root>/outbox/` |
| `get_env_file_path()` | `<root>/.env` |
| `get_archive_db_path()` | `<root>/archive.db` |
| `get_eval_dir()` | `<root>/evaluations/` |
| `get_web_auth_path()` | `<root>/web_auth.json` |
| `get_hive_web_auth_path()` | `<root>/hive_web_auth.json` |
| `get_profile_dir()` | `<root>/profiles/` |
| `get_config_yaml_path()` | `<root>/config.yaml` |

## Wiring it through the codebase

The resolver itself is useless without the rest of the source
tree calling it. Sixteen files were edited so that every place
that used to compute its own `~/.ctxmtg/...` path now goes
through `paths.py`.

The edits split into three patterns.

### Pattern 1 -- Pydantic field default factories

`config/settings.py` previously had string defaults like
`default="~/.ctxmtg/knowledge.db"`. Pydantic computes field
defaults at class-definition time (import time), so a string
default is frozen the moment Python first imports the settings
module. That made `CTXMTG_DATA_ROOT` overrides take effect only
if you set them before the first `import ctxmtg.config.settings`.

The fix is `default_factory`:

```python
db_path: str = Field(
    default_factory=lambda: str(paths.get_db_path()),
    description="Path to the SQLite database file",
)
```

Pydantic now evaluates the factory at instantiation time, so
the env var is honoured no matter when it was set. Eight fields
moved to default factories: `db_path`, `vector_path`, `inbox_path`,
`processed_path`, `hive.local_db_path`, `hive.local_vector_path`,
`hive.outbox_path`, and the model's `env_file` config key.

### Pattern 2 -- Late-bound module helpers

Several modules (`web/auth.py`, `web/hive_auth.py`,
`config/env_file.py`) had module-level constants like:

```python
ENV_PATH = Path("~/.ctxmtg/.env").expanduser()
```

These captured the old default at import time. Replacing the
constant with a function fixes the timing:

```python
def _env_path():
    return paths.get_env_file_path()
```

Every read or write call now resolves the path fresh.

### Pattern 3 -- Direct hardcoded fallback removal

`farming/__init__.py` derived `archive.db` from a
`CTXMTG_HOME` environment variable as a last-ditch fallback.
Replaced with the resolver, while keeping `CTXMTG_DB_PATH` as the
multi-instance override (it puts `archive.db` next to the
custom `knowledge.db`).

`query/evaluation.py` had `DEFAULT_EVAL_DIR = "~/.ctxmtg/evaluations"`
as a module constant. Removed; `get_eval_dir()` now falls through
to `paths.get_eval_dir()`.

`constants.py` had four `DEFAULT_*_PATH` constants
(`DEFAULT_DATA_DIR`, `DEFAULT_DB_PATH`, `DEFAULT_VECTOR_PATH`,
`DEFAULT_PROFILE_DIR`) that were never imported by any other
module. Removed.

## Configuration & operations changes

- **`configs/default.yaml`** -- `storage.db_path`,
  `storage.vector_path`, `hive.local_db_path`,
  `hive.local_vector_path` now default to `""`. The YAML loader
  treats empty strings as "use the resolver's default", so
  out-of-the-box installs need no path config at all.

- **`.env.example`** -- carries a commented-out
  `CTXMTG_DATA_ROOT=...` line documenting the override and a
  short paragraph explaining the relationship between the
  per-clone default and the override.

- **`.gitignore`** -- adds `.runtime/` and `.venv/`. Both are
  per-clone artifacts that should never be committed.

- **`README.md`** -- adds a path-note callout in Quick Start and
  updates the inbox feature row from `~/.ctxmtg/inbox/` to
  `<project>/.runtime/inbox/`.

## Migration path for existing installations

Existing `~/.ctxmtg/` users have three options:

1. **Move the data.** Copy `~/.ctxmtg/*` to
   `<project_root>/.runtime/` and the next run picks it up.
2. **Keep the data where it is.** Set
   `CTXMTG_DATA_ROOT=~/.ctxmtg` in the shell or in
   `.env`. The resolver expands `~` and uses the old location.
3. **Use both, partitioned.** Different clones get different
   roots; one of them can keep pointing at `~/.ctxmtg`. This is
   the v0.7.1-and-later equivalent of the old `CTXMTG_HOME`
   discipline.

Multi-instance setups (e.g. `Local_Tickets`, `Local_Emails`,
`Hive` running on the same host) should use option 1: each
clone gets its own `.runtime/` and there is no shared state to
forget about.

## What v0.7.1 deliberately did NOT do

A few attractive ideas were considered and dropped for scope.

- **`--data-root` CLI flag.** The env var covers the same need
  with less surface area. Adding a flag means threading it
  through every Click command's lifespan and reasoning about
  precedence vs the env var. Deferred.
- **Auto-migration of `~/.ctxmtg`.** Detect the old location and
  move data on first run. The convenience is real but the data-
  loss risk on a misconfigured `CTXMTG_DATA_ROOT` is also real.
  Dropped in favour of explicit user-driven migration.
- **Removing `CTXMTG_DB_PATH` and `CTXMTG_HOME` entirely.**
  `CTXMTG_DB_PATH` is still used by multi-instance setups where
  the archive must live alongside a custom-located knowledge.db
  (because the legacy DGX deployment relies on it). It stays.
  `CTXMTG_HOME` was retired from the source-of-truth path
  (`farming/__init__.py` no longer reads it), but the env var is
  not actively rejected anywhere -- if someone happens to have
  it set, nothing reads it and nothing breaks.

## Verification

End-to-end smoke test on Windows / Python 3.14.3 against a fresh
clone:

| Step | Result |
|---|---|
| `git clone https://github.com/D-jai/The_Borg_DB-public.git` | OK |
| `python -m venv .venv && .venv\Scripts\pip install -e ".[dev,web]"` | OK, 87 packages |
| `pip install <bridge>/en_core_web_sm-3.8.0-py3-none-any.whl` | OK |
| `ctxmtg health` | PASS, data root resolved to `<project>\.runtime` |
| `ctxmtg ingest "Alice proposed migrating auth..."` | PASS, 3 entities + 1 fact + 1 embedding |
| `ctxmtg query "What did Alice propose?"` | PASS, 2 results, vector path active, no quality_log warning |
| `ctxmtg farm run` | PASS, 18 / 18 stages, 5 insights, 263 ms |
| `ctxmtg serve --port 8081`, `GET /login` | HTTP 200 |
| `~/.ctxmtg/` after the run | does not exist |

Override path was also verified:

```
> $env:CTXMTG_DATA_ROOT = "C:\tmp\borg-test"
> ctxmtg health
... db_path=C:\tmp\borg-test\knowledge.db ...
```

(Override happens at process start; the lru_cache is per-process,
which is the correct scope for a CLI invocation.)
