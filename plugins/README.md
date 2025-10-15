# Plugins

This folder demonstrates the plugin-first architecture. Each plugin lives under `plugins/<type>/<name>/` and must include:

- `plugin.yaml` (manifest): id, name, version, type, entrypoint, capabilities, config_schema
- `main.py` (or other entrypoint): exposes the plugin class specified in the manifest
- optional `go_worker/` for compute offload
- `README.md` and minimal tests in `tests/`

Sample plugins provided:
- `plugins/samples/onvif/`
- `plugins/samples/notify/`
