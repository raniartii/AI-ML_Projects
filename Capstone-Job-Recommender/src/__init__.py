# src/__init__.py
"""
Lightweight init. Do NOT import heavy modules here.
This avoids import cycles and lets rebuild scripts run even if some deps are broken.
"""

__version__ = "0.1.0"

# If you want convenient imports later, use lazy access:
def __getattr__(name):
    if name == "paths":
        from . import paths as _paths
        return _paths
    raise AttributeError(name)
