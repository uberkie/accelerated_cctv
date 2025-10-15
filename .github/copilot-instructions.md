<!--
Repository: accelerated_cctv
Purpose: Guidance for AI coding agents to be productive in this codebase.
Do not add aspirational items; document only patterns and discoverable workflows.
-->

# Copilot / AI agent instructions — accelerated_cctv

Short, actionable guidance to help an AI coding agent make productive, repo-consistent changes.

- Project big picture
  - This repository is an early-stage framework for an accelerated CCTV/security platform (see `TODO.md`).
  - Core capabilities to assume when proposing changes: multi-stream ingestion, notification/workflow orchestration, plugin-friendly architecture, GPU-offload candidates (compute-heavy code may be moved to Go/C++), horizontal scaling, security-first.

- Important files to reference
  - `TODO.md` — living product goals and constraints (use for high-level intent).
  - `README.md` — placeholder (no useful content yet). If you modify behavior, also update the README with quick run instructions.

- Coding conventions & assumptions (discoverable)
  - The repo is minimal and early-stage; prefer small, incremental edits and add tests where possible.
  - When adding compute-heavy pieces, mark them with a comment like: // PERF: consider moving to Go/FFI/GPU kernel — this repository documents such decisions in TODO.md.
  - Keep public APIs stable: add new functions behind feature flags or new modules rather than refactoring existing exports.

- Workflows, build, and tests
  - No build/test tooling is present in the repo root. Before adding heavy changes, create reproducible dev instructions in `README.md` and include dependency manifests (e.g., `requirements.txt`, `go.mod`, `package.json`) alongside new code.
  - Prefer small unit tests that run quickly; add a simple test harness that can be executed with a single command (document it in README).

- Integration points & external dependencies
  - Expect integration with video ingestion, storage backends, notification systems, and optional GPU/Go components. Do not hardcode external credentials or endpoints; use environment variables and document them in README.

- Patterns to follow when editing
  - Minimal, reversible changes: prefer adding new files/modules rather than large rewrites.
  - When creating new components, include a brief design note in the file top comment describing data flow, expected inputs/outputs, and error modes.
  - Add explicit TODO comments for orchestration boundaries (e.g., "ORCH: ingestion -> storage -> workflow") so future contributors can trace cross-component flows.

- Examples from codebase
  - Use `TODO.md` as the single source of product intent. Refer to it for decisions about GPU offload, horizontal scaling, and plugin friendliness.

- Agent behavior & preferences
  - Keep edits small and focused (one logical change per PR). If a change spans multiple modules, open a design PR first with high-level notes.
  - Include test or example usage for any added public API.
  - When uncertain about runtime details (build commands, language versions), add a clear TODO and update `README.md` asking the maintainer to confirm.

- Plugin-first architecture (required)
  - All core components (ingestion, storage, notify, analytics, orchestrator, API surface) must be implemented as plugins.
  - Plugin shape: each plugin must include a short manifest (YAML/JSON) describing id, version, capabilities, config schema, and entrypoint. Example path: `plugins/<type>/<name>/plugin.yaml`.
  - Lifecycle: plugins must support discovery, registration at startup, health-check endpoint (or hook), graceful shutdown, and safe reload (if supported by runtime).
  - Sandboxing & safety: prefer process isolation for untrusted plugins. For heavy compute plugins, include optional GPU hints in manifest (e.g., required devices).
  - Testing: provide an in-repo sample plugin for each type in `plugins/samples/` which demonstrates the expected manifest and minimal tests.



If anything in this guidance is unclear or you need access to missing files (build scripts, CI config, language-specific code), ask the maintainer before making large architectural changes.
