r"""
DeformOps
=========

Deformable operations for PyTorch, implemented in C++ and CUDA.
"""

from . import library, nn, version

__all__ = [
    "version",
    "library",
    "nn",
    "deform2d",  # pyright: ignore[reportUnsupportedDunderAll]
    "deform2d_multiscale",  # pyright: ignore[reportUnsupportedDunderAll]
    "deform2d_multiscale_fused",  # pyright: ignore[reportUnsupportedDunderAll]
]


def __getattr__(name: str):  # pyright: ignore[reportUnknownParameterType]
    r"""Generic import hook.

    When importing deformops.<name>, this function will be called to load
    the library with the given name, and return its forward method.
    """

    if name in ("deform2d", "deform2d_multiscale", "deform2d_multiscale_fused"):
        return library.load(name).forward
    if name == "__version__":
        return version.local

    msg = f"module {__name__} has no attribute {name}"
    raise AttributeError(msg)
