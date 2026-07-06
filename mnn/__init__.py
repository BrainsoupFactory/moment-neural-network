# -*- coding: utf-8 -*-

from importlib import import_module


__version__ = "0.2.0"

_LAZY_IMPORTS = {
    "mnn_core": ".mnn_core",
    "utils": ".utils",
    "snn": ".snn",
    "models": ".models",
    "nn": ".mnn_core",
    "training_tools": ".utils",
}

__all__ = ["__version__", *tuple(_LAZY_IMPORTS)]


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_LAZY_IMPORTS[name], __name__)
    value = getattr(module, name) if name in {"nn", "training_tools"} else module
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
