# Sample ONVIF plugin

This sample demonstrates the expected plugin shape for an ingestion plugin.

- Manifest: `plugin.yaml`
- Python entrypoint: `main.py` (class `Plugin`)
- Optional Go compute worker: `go_worker/` (stdin/stdout JSON protocol)

How to run the sample (requires Python 3):

```bash
python3 plugins/samples/onvif/main.py
```
