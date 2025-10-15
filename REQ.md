# accelerated_cctv — Requirements

This document captures operational and functional requirements discovered from the project TODO and repository context. Use this as a living reference when implementing features, tests, or infra.

## Overview
Split into Functional Requirements (FR) and Operational / Non‑Functional Requirements (OR). Each item includes acceptance criteria and implementation notes.

---

## Functional Requirements (FR)

1. FR1 — Camera discovery & control (ONVIF)
   - Requirement: Discover ONVIF-capable devices, authenticate, fetch capabilities (profiles/streams), support PTZ and camera events.
   - Acceptance: Agent can discover a camera, list its RTSP/H264/H265 profile URIs, and send PTZ move/stop commands.
   - Implementation notes: `services/ingestion/onvif_connector.py` (or `pkg/ingestion/onvif/*`) wrapping an ONVIF client library; unit tests should mock ONVIF endpoints.

2. FR2 — Stream ingestion (RTSP/RTMP/WebRTC)
   - Requirement: Ingest live streams reliably, reconnect on transient errors, expose ingest pipeline hooks (pre-processing, analytics).
   - Acceptance: Stream remains ingestible under intermittent network drops; reconnection attempts with backoff.
   - Implementation notes: FFmpeg/GStreamer pipeline wrapper `services/ingestion/stream_ingest.py`; provide options for hardware acceleration (VAAPI/NVENC).

3. FR3 — Encoder / Decoder & Transcoding
   - Requirement: Decode incoming streams, optionally transcode to target codecs/resolutions, produce DASH/HLS segments and thumbnails.
   - Acceptance: Produce HLS segments and a 720p derivative from a 4K RTSP stream.
   - Implementation notes: Use FFmpeg or GStreamer; wrappers in `libs/transcode/`. Mark heavy code with `// PERF: consider moving to Go/FFI/GPU kernel`.

4. FR4 — Chunked storage & metadata indexing
   - Requirement: Store original and derived segments (S3 or local FS), index metadata (camera id, timestamps, analytics tags) in a relational DB.
   - Acceptance: Retrieve segments given camera id + time window; metadata query returns segment URIs.
   - Implementation notes: Storage adapter `libs/storage/adapter.py` with S3/local implementations; metadata service `services/indexer/` using Postgres (`db/schema.sql`).

5. FR5 — Notification / Delivery wrapper (pluggable)
   - Requirement: Centralized notification wrapper for webhook, email, SMS, push, Slack; templating, deduplication, retries/backoff.
   - Acceptance: Send notification to a webhook and retry on 5xx with exponential backoff.
   - Implementation notes: `libs/notify/` + transports `libs/notify/transports/*`; durable delivery via message queue (RabbitMQ/Kafka); support HMAC signing.

6. FR6 — Workflow / orchestration engine
   - Requirement: Define and execute workflows triggered by events (motion, schedule, analytics), with branching, retries, and enrichment.
   - Acceptance: Workflow that on "person detected" calls notify wrapper and stores a clip; logs success/failure.
   - Implementation notes: Integrate Temporal/n8n/Argo or implement `services/orchestrator/` with a YAML/JSON DSL.

7. FR7 — Analytics plugin API (model inference)
   - Requirement: Plugin interface for analytics modules (object detection, LPR) with standardized I/O (frames → JSON results).
   - Acceptance: Plugin registered and invoked on a frame; results stored in metadata and available to workflows.
   - Implementation notes: `plugins/analytics/` with a spec and sample plugin; support local/remote model runners (gRPC).

8. FR8 — API & SDK
   - Requirement: Public API for management and clients, OpenAPI spec, and a minimal SDK.
   - Acceptance: `api/openapi.json` present; REST endpoint to authenticate and list cameras.
   - Implementation notes: `services/api/` (FastAPI/Express) with token-based auth.

9. FR9 — Plugin system & lifecycle (plugin-first required)
   - Requirement: All core components must be implemented as plugins (ingestion, storage, notify, analytics, orchestrator, API surface).
   - Acceptance: Installing a plugin (drop-in or via config) results in registration and discovery at startup; uninstall unregisters it.
   - Implementation notes: `plugins/` folder with manifests. Each plugin must include a manifest (`plugins/<type>/<name>/plugin.yaml` or `.json`) describing id, version, capabilities, config schema, entrypoint, and optional GPU hints. Provide `plugins/samples/` demonstrating the expected shape and minimal tests.

   - Lifecycle: plugins should support discovery, registration, health checks, graceful shutdown, and safe reload (if runtime supports it). Prefer process isolation for untrusted plugins; heavy compute plugins may run in separate containers with declared GPU requirements.


10. FR10 — Recording & clip extraction
    - Requirement: Create clips around events (pre/post buffers), generate metadata and thumbnails, apply retention.
    - Acceptance: On event trigger, generate clip with X seconds before & Y after; store segments and thumbnail URIs.
    - Implementation notes: Segmenter in ingestion/transcode pipeline; clip composer `services/clips/`.

---

## Operational / Non‑Functional Requirements (OR)

1. OR1 — Authentication & Authorization
   - Req: TLS everywhere, API auth via OAuth2/JWT, optional mTLS, RBAC for users and clients.

2. OR2 — Encryption & Data protection
   - Req: Encrypt sensitive metadata at rest; server-side encryption for object storage; use KMS for keys.

3. OR3 — Scalability & statelessness
   - Req: Services stateless where possible; state persisted in DB/queue/storage; k8s-friendly.

4. OR4 — High availability & reliability
   - Req: Retry policies, backpressure handling, circuit breakers, durable queues.

5. OR5 — Observability & telemetry
   - Req: Structured logs (JSON), metrics (Prometheus), tracing (OpenTelemetry), dashboards/alerts.

6. OR6 — Monitoring, SLOs & runbooks
   - Req: Define SLOs and provide runbooks in `docs/runbooks/`.

7. OR7 — Deployment & infra as code
   - Req: IaC (Terraform) and charts (Helm). Provide `docker-compose` for local dev.

8. OR8 — Backup & disaster recovery
   - Req: DB snapshots and cross-region object storage replication; documented RTO/RPO.

9. OR9 — Compliance & auditing
   - Req: Audit logs for sensitive actions; configurable retention policies.

10. OR10 — Performance & hardware acceleration
    - Req: Detect and use hardware encoders (NVIDIA/Intel) when available; software fallback.

11. OR11 — Testing & CI
    - Req: Unit tests, fast integration tests (mocked ONVIF/RTSP), e2e smoke tests, linters, pre-commit hooks.

12. OR12 — Upgradeability & migration
    - Req: DB migrations (Alembic/Flyway), versioned plugin manifests, rolling upgrades.

13. OR13 — Cost & resource management
    - Req: Retention policies, archive tiers, throttling of expensive analytics jobs.

---

## MVP (prioritized subset)
- FR2 (RTSP ingest), FR4 (local chunked storage + DB metadata), FR5 (webhook notifications), FR8 (basic API to list cameras/clips), FR10 (clip extraction), OR1 (basic token auth + TLS), OR3 (stateless services + docker-compose), OR11 (tests + CI scaffold).

## Suggested repo layout
- `services/ingestion/`
- `libs/transcode/`
- `libs/storage/`
- `libs/notify/`
- `services/api/`
- `services/orchestrator/`
- `plugins/analytics/`
- `db/`
- `docs/`
- `tests/`
- `docker-compose.yml` & `dev/`

## Implementation hints & conventions (project‑specific)
- Mark heavy compute code: `// PERF: consider moving to Go/FFI/GPU kernel`.
- Keep public APIs stable — add new modules rather than refactoring existing ones.
- Use env vars for external credentials and document them in `README.md`.
- Small, reversible changes; open design PRs for cross-cutting changes.

---

If you want, I can split these items into GitHub issues and scaffold an MVP `docker-compose.yml` and a tiny RTSP ingest + webhook notify example for local testing. Tell me preferred languages/tech choices (Python/Go, Kafka/RabbitMQ, S3/local) and I'll scaffold accordingly.
