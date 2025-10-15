import importlib


def test_onvif_import():
    mod = importlib.import_module('plugins.samples.onvif.main')
    Plugin = getattr(mod, 'Plugin')
    p = Plugin({"poll_interval": 1})
    assert hasattr(p, 'discover')
