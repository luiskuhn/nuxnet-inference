# Configuration Audit (Potentially Obsolete / Unused)

This audit highlights **potentially** obsolete or unused configuration files based on static inspection of the current repository.

## 1) `mlf_core.cfg`

**Why it may be obsolete**
- Appears to be metadata for an external templating/sync workflow (`mlf_core`/`bumpversion`) that is not referenced elsewhere in this repository.
- References `.mlf_core.yml`, which is not present in the repository.
- Tracks `setup.py` for versioning, while canonical project metadata is defined in `pyproject.toml`.

**Suggested action**
- If `mlf-core` automation is no longer used in CI/release workflows, remove `mlf_core.cfg`.
- If still used externally, keep it but document the external dependency in the README or CONTRIBUTING docs.

## 2) `setup.cfg`

**Why it may be partially obsolete**
- Contains only Flake8 settings.
- Flake8 is referenced in Makefiles but not listed in `requirements_dev.txt`.
- Lint targets currently reference a non-existent `tests` directory.

**Suggested action**
- Either (a) add Flake8 to dev dependencies and align lint targets, or (b) migrate lint config into a single modern tool configuration (e.g., `pyproject.toml`) and remove `setup.cfg`.

## 3) Makefile lint targets (`makefiles/Linux.mk`, `makefiles/Windows.mk`)

**Why they may be stale**
- `lint` target checks `nuxnet_inference_package tests`, but there is no `tests/` directory.
- Build targets still use `setup.py sdist/bdist_wheel` even though packaging is primarily driven by `pyproject.toml`.

**Suggested action**
- Update lint target paths.
- Optionally switch build target to `python -m build` for PEP 517 flow.

## Likely active / still relevant config files

- `pyproject.toml`: primary packaging metadata and dependencies.
- `requirements.txt` and `requirements_dev.txt`: environment and dev dependency inputs.
- `MANIFEST.in`: source distribution includes.
- `docs/conf.py`: Sphinx documentation configuration.

