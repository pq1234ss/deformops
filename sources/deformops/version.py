"""This module defines the versions of this module. Used for compatability checks."""

from typing import TYPE_CHECKING


def _get_package() -> str:
    r"""Read the version of the installed package."""
    import importlib.metadata  # noqa: PLC0415

    return importlib.metadata.version("deformops")


def _get_environment() -> str:
    r"""Describes the current build environment in a oneline string.

    Notes
    -----
    This is agnostic of the Python version currently installed.
    """
    import platform  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # Machine platform and architecture
    abi_str = f"{sys.platform}_{platform.machine()}"

    # PyTorch version
    v_torch, *extra_torch = torch.__version__.split("+")
    torch_str = f"torch{v_torch}"

    # Backend: we use either CUDA or no-CUDA (i.e. CPU)
    if getattr(torch.version, "cuda", None) is None:
        extra_expected = "cpu"
        backend_str = extra_expected
    else:
        # Get CUDA version
        v_cuda = f"cu{torch.version.cuda}"
        extra_expected = v_cuda.replace(".", "")

        # Get cuDNN version if it's available and enabled
        if torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled:
            backend_str = f"{v_cuda}-cudnn{torch.backends.cudnn.version()}"
        else:
            # Handle case where CUDA is present but cuDNN is not
            backend_str = f"{v_cuda}-nocudnn"

    # If a canonical (extra) version is given, assert its expected value matches
    assert len(extra_torch) == 0 or extra_torch[0] == extra_expected, (
        extra_torch,
        extra_expected,
    )

    return "-".join((abi_str, torch_str, backend_str)).replace(".", "_")


def _get_install():
    return f"{_get_package()}+{_get_environment()}"


if TYPE_CHECKING:
    package: str
    environment: str
    local: str

__cache__: dict[str, str] = {}


def __getattr__(name: str):
    match name:
        case "package":
            return __cache__.setdefault(name, _get_package())
        case "environment":
            return __cache__.setdefault(name, _get_environment())
        case "local":
            return __cache__.setdefault(name, _get_install())
        case _:
            pass
    msg = f"Unsupported version attribute: {name}"
    raise AttributeError(msg)
