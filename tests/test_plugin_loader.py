from libs.plugins.loader import PluginRegistry


def test_discover_and_load():
    reg = PluginRegistry()
    reg.discover_and_load("plugins/samples")
    # Expect sample plugin ids to be present
    all_plugins = reg.all()
    # keys include plugin ids or paths; check that sample ids exist
    assert any("samples.onvif" in k or (isinstance(v, dict) and v.get('manifest', {}).get('id') == 'samples.onvif') for k, v in all_plugins.items())
    assert any("samples.notify" in k or (isinstance(v, dict) and v.get('manifest', {}).get('id') == 'samples.notify') for k, v in all_plugins.items())
