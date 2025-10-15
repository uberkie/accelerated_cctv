import importlib


def test_notify_import():
    mod = importlib.import_module('plugins.samples.notify.main')
    NotifyPlugin = getattr(mod, 'NotifyPlugin')
    n = NotifyPlugin({"max_retries": 1})
    assert hasattr(n, 'send_webhook')
