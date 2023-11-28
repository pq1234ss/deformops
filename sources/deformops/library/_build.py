r"""
C++/CUDA extensions for deformable operations.
"""

import importlib.resources
import logging
import os
import pathlib
import platform
import sys
import warnings
from collections.abc import Iterable
from typing import Literal

import torch.utils.cpp_extension
import torch.version
import deformops.version

__all__: list[str] = ["locate_build", "define_library"]

logger = logging.getLogger(__name__)


def locate_build(name: str) -> pathlib.Path:
    """Get the build directory for an extension.

    Parameters
    ----------
    name:
        The name of the extension

    Returns
    -------
    str
        The path to the build directory for the specified extension name.
    """

    try:
        ext_root = pathlib.Path(os.environ["TORCH_EXTENSIONS_DIR"])
    except KeyError:
        ext_root = (
            pathlib.Path(torch.utils.cpp_extension.get_default_build_root())
            / deformops.version.package
            / deformops.version.environment
        )

        warnings.warn(
            (
                "TORCH_EXTENSIONS_DIR is not set. "
                "This might cause PyTorch extensions to be cached globally, "
                "leading to compatability issues between PyTorch projects "
                "that have slightly different installations. "
            ),
            stacklevel=1,
        )

    logger.debug("Using %s as Deformops build directory...", str(ext_root))

    build_directory = ext_root / name
    if not build_directory.is_dir():
        build_directory.mkdir(exist_ok=True, parents=True)

    return build_directory


def define_library(
    name: str,
    build_dir: str | pathlib.Path | None = None,
    force_build: bool = False,
    *,
    with_cuda: bool = True,
    opt_level: Literal[0, 1, 2, 3] = 2,
    cflags: Iterable[str] = (),
    cflags_cuda: Iterable[str] = (),
) -> str:
    r"""Build an extension just-in-time (JIT) using the provided arguments."""
    build_dir = locate_build(name) if build_dir is None else pathlib.Path(build_dir)

    lib_file = build_dir / f"{name}.so"

    if not force_build:
        if lib_file.is_file():
            # If the library file already exists, we can skip the build process.
            torch.ops.load_library(str(lib_file))
            return str(lib_file)
        msg = (
            f"Deformops static library f{lib_file.name} not found. "
            "Building from source required. This may take a while."
        )
        warnings.warn(msg, stacklevel=2)

    root = importlib.resources.files("deformops.library")
    sources: list[str] = [str(root / f"{name}.cpp")]
    extra_include_paths: list[str] = []
    extra_cflags: list[str] = [
        "-fdiagnostics-color=always",
        "-std=c++17",
        "-DPy_LIMITED_API=0x03012000",
        f"-O{opt_level}",
        *cflags,
    ]
    extra_cuda_cflags: list[str] = [
        "--std=c++17",
        f"-O{opt_level}",
        *cflags_cuda,
    ]

    if with_cuda:
        # if CUDA_HOME is None or not pathlib.Path(CUDA_HOME).is_dir():
        #     msg = f"CUDA is not available. Found {CUDA_HOME=}"
        #     raise RuntimeError(msg)
        sources.append(str(root / "cuda" / f"{name}.cu"))
        extra_include_paths.append(str(root / "cuda"))

    torch.utils.cpp_extension.load(
        name,
        build_directory=str(build_dir),
        with_cuda=with_cuda,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        keep_intermediates=False,
        is_python_module=False,
        is_standalone=False,
        verbose=True,
    )

    if not lib_file.is_file():
        warnings.warn(
            f"Expected extension build to yield {lib_file!r}, but it does not exist.",
            stacklevel=2,
        )

    return str(lib_file)
