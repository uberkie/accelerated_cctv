"""Minimal plugin loader for tests.

Finds plugins/*/plugin.yaml and loads module or module:Class entrypoints.
This file is intentionally small and dependency-free for unit tests.
"""

import glob
import os
import importlib
from typing import Dict, Any, List, Optional


def _manifests(base_path: str) -> List[str]:
    pattern = os.path.join(base_path, "*", "plugin.yaml")
    return sorted(glob.glob(pattern))


def _read_manifest(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    return data


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, Dict[str, Any]] = {}

    def discover_and_load(self, base_path: str = "plugins") -> None:
        for manifest_path in _manifests(base_path):
            try:
                manifest = _read_manifest(manifest_path) or {}
                plugin_dir = os.path.dirname(manifest_path)
                module_root = ".".join(
                    [p for p in os.path.relpath(plugin_dir, os.getcwd()).split(os.sep) if p]
                )
                plugin_id = manifest.get("id") or module_root
                info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

                entrypoint = manifest.get("entrypoint")
                if entrypoint and isinstance(entrypoint, str):
                    if ":" in entrypoint:
                        mod_name, cls_name = entrypoint.split(":", 1)
                    else:
                        mod_name, cls_name = entrypoint, None

                    if mod_name and mod_name.lower() != "readme":
                        full_mod = module_root + "." + mod_name
                        try:
                            module = importlib.import_module(full_mod)
                            info["module"] = module
                            if cls_name:
                                cls = getattr(module, cls_name, None)
                                if cls:
                                    try:
                                        instance = cls(manifest.get("config", {}))
                                    except Exception:
                                        try:
                                            instance = cls()
                                        except Exception:
                                            instance = None
                                    info["instance"] = instance
                                else:
                                    info["load_error"] = f"attribute {cls_name} not found in {full_mod}"
                        except Exception as e:
                            info["load_error"] = str(e)
                else:
                    info.setdefault("load_error", "no entrypoint")

                self._plugins[plugin_id] = info
            except Exception as e:
                # record failure keyed by manifest path
                self._plugins[manifest_path] = {"error": str(e)}

    def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        return self._plugins.get(plugin_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._plugins)


__all__ = ["PluginRegistry"]
"""Very small plugin loader used by tests.

This loader is deliberately tiny. It searches plugins/*/plugin.yaml, reads a
very small manifest (key: value lines) and imports the declared Python
entrypoint (module or module:Class).
"""

import glob
import os
import importlib
from typing import Dict, Any, List, Optional


def _manifests(base_path: str) -> List[str]:
    pattern = os.path.join(base_path, "*", "plugin.yaml")
    return sorted(glob.glob(pattern))


def _read_manifest(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    return data


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, Dict[str, Any]] = {}

    def discover_and_load(self, base_path: str = "plugins") -> None:
        for manifest_path in _manifests(base_path):
            manifest = _read_manifest(manifest_path) or {}
            plugin_dir = os.path.dirname(manifest_path)
            module_root = ".".join([p for p in os.path.relpath(plugin_dir, os.getcwd()).split(os.sep) if p])
            plugin_id = manifest.get("id") or module_root
            info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

            entrypoint = manifest.get("entrypoint")
            if entrypoint:
                if ":" in entrypoint:
                    mod_name, cls_name = entrypoint.split(":", 1)
                else:
                    mod_name, cls_name = entrypoint, None
                full_mod = module_root + "." + mod_name
                try:
                    mod = importlib.import_module(full_mod)
                    """Very small plugin loader used by tests.

                    This loader is intentionally tiny. It searches plugins/*/plugin.yaml, reads a
                    very small manifest (key: value lines) and imports the declared Python
                    entrypoint (module or module:Class).
                    """

                    import glob
                    import os
                    import importlib
                    from typing import Dict, Any, List, Optional


                    def _manifests(base_path: str) -> List[str]:
                        pattern = os.path.join(base_path, "*", "plugin.yaml")
                        return sorted(glob.glob(pattern))


                    def _read_manifest(path: str) -> Dict[str, Any]:
                        data: Dict[str, Any] = {}
                        with open(path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                if ":" in line:
                                    k, v = line.split(":", 1)
                                    data[k.strip()] = v.strip()
                        return data


                    class PluginRegistry:
                        def __init__(self) -> None:
                            self._plugins: Dict[str, Dict[str, Any]] = {}

                        def discover_and_load(self, base_path: str = "plugins") -> None:
                            for manifest_path in _manifests(base_path):
                                manifest = _read_manifest(manifest_path) or {}
                                plugin_dir = os.path.dirname(manifest_path)
                                module_root = ".".join([p for p in os.path.relpath(plugin_dir, os.getcwd()).split(os.sep) if p])
                                plugin_id = manifest.get("id") or module_root
                                info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

                                entrypoint = manifest.get("entrypoint")
                                if entrypoint:
                                    if ":" in entrypoint:
                                        mod_name, cls_name = entrypoint.split(":", 1)
                                    else:
                                        mod_name, cls_name = entrypoint, None
                                    full_mod = module_root + "." + mod_name
                                    try:
                                        mod = importlib.import_module(full_mod)
                                        info["module"] = mod
                                        if cls_name:
                                            cls = getattr(mod, cls_name, None)
                                            if cls:
                                                try:
                                                    inst = cls(manifest.get("config", {}))
                                                except Exception:
                                                    try:
                                                        inst = cls()
                                                    except Exception:
                                                        inst = None
                                                info["instance"] = inst
                                    except Exception as e:
                                        info["load_error"] = str(e)

                                self._plugins[plugin_id] = info

                        def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
                            return self._plugins.get(plugin_id)

                        def all(self) -> Dict[str, Dict[str, Any]]:
                            return dict(self._plugins)


                    __all__ = ["PluginRegistry"]
                                    info["instance"] = instance
                        except Exception as e:
                            info["load_error"] = str(e)

                self._plugins[plugin_id] = info
            except Exception as e:
                self._plugins[manifest_path] = {"error": str(e)}

    def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        return self._plugins.get(plugin_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._plugins)


__all__ = ["PluginRegistry"]
"""Clean plugin loader implementation.

Scans plugins/<type>/<name>/plugin.yaml, parses manifests (PyYAML if
available, small fallback otherwise), imports module:Class entrypoints and
records results. Focused on simplicity and testability.
"""

from typing import Dict, Any, List, Optional
import glob
import os
import importlib
import re


def _find_manifests(base_path: str) -> List[str]:
    if os.path.isdir(base_path):
        pattern = os.path.join(base_path, "*", "plugin.yaml")
    """Plugin loader (clean, minimal).

    Small, test-focused plugin registry. Scans for plugins/*/plugin.yaml and
    imports the module specified in the manifest's `entrypoint` (module:Class
    or module). If a class is provided, it attempts to instantiate it.
    """

    from typing import Dict, Any, List, Optional
    import glob
    import os
    import importlib
    import re


    def _find_manifests(base_path: str) -> List[str]:
        if os.path.isdir(base_path):
            pattern = os.path.join(base_path, "*", "plugin.yaml")
        """Plugin loader (clean, minimal).

        Small, test-focused plugin registry. Scans for plugins/*/plugin.yaml and
        imports the module specified in the manifest's `entrypoint` (module:Class
        or module). If a class is provided, it attempts to instantiate it.
        """

        from typing import Dict, Any, List, Optional
        import glob
        import os
        import importlib
        import re


        def _find_manifests(base_path: str) -> List[str]:
            if os.path.isdir(base_path):
                pattern = os.path.join(base_path, "*", "plugin.yaml")
            else:
                pattern = base_path
            return sorted(glob.glob(pattern))


        def _read_manifest(path: str) -> Dict[str, Any]:
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                data: Dict[str, Any] = {}
                key_re = re.compile(r"^(?P<k>[a-zA-Z0-9_\-]+):\s*(?P<v>.+)$")
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        m = key_re.match(line.strip())
                        if not m:
                            continue
                        k = m.group("k")
                        v = m.group("v").strip()
                        if v.startswith("[") and v.endswith("]"):
                            items = [s.strip().strip('"\'') for s in v[1:-1].split(",") if s.strip()]
                            data[k] = items
                        else:
                            data[k] = v
                return data


        def _module_root_from_dir(plugin_dir: str) -> str:
            # Turn a path like plugins/samples/onvif into plugins.samples.onvif
            rel = os.path.relpath(plugin_dir, os.getcwd())
            parts = [p for p in rel.split(os.sep) if p and p != "."]
            return ".".join(parts)


        class PluginRegistry:
            """Registry that discovers plugin manifests and loads entrypoints."""

            def __init__(self) -> None:
                self._plugins: Dict[str, Dict[str, Any]] = {}

            def discover_and_load(self, base_path: str = "plugins") -> None:
                manifests = _find_manifests(base_path)
                for manifest_path in manifests:
                    try:
                        manifest = _read_manifest(manifest_path) or {}
                        plugin_dir = os.path.dirname(manifest_path)
                        module_root = _module_root_from_dir(plugin_dir)
                        plugin_id = manifest.get("id") or module_root
                        info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

                        entrypoint = manifest.get("entrypoint")
                        if entrypoint and isinstance(entrypoint, str):
                            if ":" in entrypoint:
                                mod_name, cls_name = entrypoint.split(":", 1)
                            else:
                                mod_name, cls_name = entrypoint, None

                            if mod_name and mod_name.lower() != "readme":
                                full_mod = module_root + "." + mod_name
                                try:
                                    module = importlib.import_module(full_mod)
                                    info["module"] = module
                                    if cls_name:
                                        cls = getattr(module, cls_name, None)
                                        if cls:
                                            try:
                                                instance = cls(manifest.get("config", {}))
                                            except Exception:
                                                try:
                                                    instance = cls()
                                                except Exception:
                                                    instance = None
                                            info["instance"] = instance
                                        else:
                                            info["load_error"] = f"attribute {cls_name} not found in {full_mod}"
                                except Exception as e:
                                    info["load_error"] = str(e)
                        else:
                            info.setdefault("load_error", "no entrypoint")

                        self._plugins[plugin_id] = info
                    except Exception as e:
                        # record failure keyed by manifest path
                        self._plugins[manifest_path] = {"error": str(e)}

            def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
                return self._plugins.get(plugin_id)

            def all(self) -> Dict[str, Dict[str, Any]]:
                return dict(self._plugins)


        __all__ = ["PluginRegistry"]
                                                                        if not m:
                                                                            continue
                                                                        k = m.group("k")
                                                                        v = m.group("v").strip()
                                                                        if v.startswith("[") and v.endswith("]"):
                                                                            items = [s.strip().strip('"\'') for s in v[1:-1].split(",") if s.strip()]
                                                                            data[k] = items
                                                                        else:
                                                                            data[k] = v
                                                                return data


                                                        def _module_root_from_dir(plugin_dir: str) -> str:
                                                            # Turn a path like plugins/samples/onvif into plugins.samples.onvif
                                                            rel = os.path.relpath(plugin_dir, os.getcwd())
                                                            parts = [p for p in rel.split(os.sep) if p and p != "."]
                                                            return ".".join(parts)


                                                        class PluginRegistry:
                                                            """Registry that discovers plugin manifests and loads entrypoints."""

                                                            def __init__(self) -> None:
                                                                self._plugins: Dict[str, Dict[str, Any]] = {}

                                                            def discover_and_load(self, base_path: str = "plugins") -> None:
                                                                manifests = _find_manifests(base_path)
                                                                for manifest_path in manifests:
                                                                    try:
                                                                        manifest = _read_manifest(manifest_path) or {}
                                                                        plugin_dir = os.path.dirname(manifest_path)
                                                                        module_root = _module_root_from_dir(plugin_dir)
                                                                        plugin_id = manifest.get("id") or module_root
                                                                        info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

                                                                        entrypoint = manifest.get("entrypoint")
                                                                        if entrypoint and isinstance(entrypoint, str):
                                                                            if ":" in entrypoint:
                                                                                mod_name, cls_name = entrypoint.split(":", 1)
                                                                            else:
                                                                                mod_name, cls_name = entrypoint, None

                                                                            if mod_name and mod_name.lower() != "readme":
                                                                                full_mod = module_root + "." + mod_name
                                                                                try:
                                                                                    module = importlib.import_module(full_mod)
                                                                                    info["module"] = module
                                                                                    if cls_name:
                                                                                        cls = getattr(module, cls_name, None)
                                                                                        if cls:
                                                                                            try:
                                                                                                instance = cls(manifest.get("config", {}))
                                                                                            """Plugin loader (clean, minimal).

                                                                                            Small, test-focused plugin registry. Scans for plugins/*/plugin.yaml and
                                                                                            imports the module specified in the manifest's `entrypoint` (module:Class
                                                                                            or module). If a class is provided, it attempts to instantiate it.
                                                                                            """

                                                                                            from typing import Dict, Any, List, Optional
                                                                                            import glob
                                                                                            import os
                                                                                            import importlib
                                                                                            import re


                                                                                            def _find_manifests(base_path: str) -> List[str]:
                                                                                                if os.path.isdir(base_path):
                                                                                                    pattern = os.path.join(base_path, "*", "plugin.yaml")
                                                                                                else:
                                                                                                    pattern = base_path
                                                                                                return sorted(glob.glob(pattern))


                                                                                            def _read_manifest(path: str) -> Dict[str, Any]:
                                                                                                try:
                                                                                                    import yaml
                                                                                                    with open(path, "r", encoding="utf-8") as f:
                                                                                                        return yaml.safe_load(f) or {}
                                                                                                except Exception:
                                                                                                    data: Dict[str, Any] = {}
                                                                                                    key_re = re.compile(r"^(?P<k>[a-zA-Z0-9_\-]+):\s*(?P<v>.+)$")
                                                                                                    with open(path, "r", encoding="utf-8") as f:
                                                                                                        for line in f:
                                                                                                            m = key_re.match(line.strip())
                                                                                                            if not m:
                                                                                                                continue
                                                                                                            k = m.group("k")
                                                                                                            v = m.group("v").strip()
                                                                                                            if v.startswith("[") and v.endswith("]"):
                                                                                                                items = [s.strip().strip('"\'') for s in v[1:-1].split(",") if s.strip()]
                                                                                                                data[k] = items
                                                                                                            else:
                                                                                                                data[k] = v
                                                                                                    return data


                                                                                            def _module_root_from_dir(plugin_dir: str) -> str:
                                                                                                # Turn a path like plugins/samples/onvif into plugins.samples.onvif
                                                                                                rel = os.path.relpath(plugin_dir, os.getcwd())
                                                                                                parts = [p for p in rel.split(os.sep) if p and p != "."]
                                                                                                return ".".join(parts)


                                                                                            class PluginRegistry:
                                                                                                """Registry that discovers plugin manifests and loads entrypoints."""

                                                                                                def __init__(self) -> None:
                                                                                                    self._plugins: Dict[str, Dict[str, Any]] = {}

                                                                                                def discover_and_load(self, base_path: str = "plugins") -> None:
                                                                                                    manifests = _find_manifests(base_path)
                                                                                                    for manifest_path in manifests:
                                                                                                        try:
                                                                                                            manifest = _read_manifest(manifest_path) or {}
                                                                                                            plugin_dir = os.path.dirname(manifest_path)
                                                                                                            module_root = _module_root_from_dir(plugin_dir)
                                                                                                            plugin_id = manifest.get("id") or module_root
                                                                                                            info: Dict[str, Any] = {"manifest": manifest, "path": plugin_dir}

                                                                                                            entrypoint = manifest.get("entrypoint")
                                                                                                            if entrypoint and isinstance(entrypoint, str):
                                                                                                                if ":" in entrypoint:
                                                                                                                    mod_name, cls_name = entrypoint.split(":", 1)
                                                                                                                else:
                                                                                                                    mod_name, cls_name = entrypoint, None

                                                                                                                if mod_name and mod_name.lower() != "readme":
                                                                                                                    full_mod = module_root + "." + mod_name
                                                                                                                    try:
                                                                                                                        module = importlib.import_module(full_mod)
                                                                                                                        info["module"] = module
                                                                                                                        if cls_name:
                                                                                                                            cls = getattr(module, cls_name, None)
                                                                                                                            if cls:
                                                                                                                                try:
                                                                                                                                    instance = cls(manifest.get("config", {}))
                                                                                                                                except Exception:
                                                                                                                                    try:
                                                                                                                                        instance = cls()
                                                                                                                                    except Exception:
                                                                                                                                        instance = None
                                                                                                                                info["instance"] = instance
                                                                                                                            else:
                                                                                                                                info["load_error"] = f"attribute {cls_name} not found in {full_mod}"
                                                                                                                    except Exception as e:
                                                                                                                        info["load_error"] = str(e)
                                                                                                            else:
                                                                                                                info.setdefault("load_error", "no entrypoint")

                                                                                                            self._plugins[plugin_id] = info
                                                                                                        except Exception as e:
                                                                                                            # record failure keyed by manifest path
                                                                                                            self._plugins[manifest_path] = {"error": str(e)}

                                                                                                def get(self, plugin_id: str) -> Optional[Dict[str, Any]]:
                                                                                                    return self._plugins.get(plugin_id)

                                                                                                def all(self) -> Dict[str, Dict[str, Any]]:
                                                                                                    return dict(self._plugins)


                                                                                            __all__ = ["PluginRegistry"]
