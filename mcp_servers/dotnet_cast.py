"""Python.NET: cast managed objects to .NET interface types.

Python.NET 3 removed ``clr.Convert``; use ``iface_type(device)`` (see pythonnet
``test_interface_conversion``). If ``clr.Convert`` exists (older stacks), it is used.
"""

from __future__ import annotations

from typing import Any


def cast_clr_iface(device: Any, iface_type: Any) -> Any:
    import clr  # type: ignore[import-not-found]

    conv = getattr(clr, "Convert", None)
    if conv is not None:
        return conv(device, iface_type)
    return iface_type(device)
