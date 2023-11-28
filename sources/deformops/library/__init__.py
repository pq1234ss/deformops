r"""Library of operations."""

from ._build import define_library, locate_build

__all__ = [
    "locate_build",
    "define_library",
    "load",
]


def load(name: str):
    r"""Load library with the given name, or all libraries if name is None."""

    import importlib  # noqa: PLC0415

    return importlib.import_module(f".{name}", __name__)
