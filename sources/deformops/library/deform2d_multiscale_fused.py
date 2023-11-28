r"""
Fused Multi-Scale Deform Op
===========================

Implements the multi-scale deformable sampling operator, with softmax fused in at the
kernel level.
"""

import functools
import typing

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._build import define_library
from ._factors import find_factors
from .deform2d_multiscale import reference as _base_reference

__all__ = [
    "backward",
    "forward",
    "reference",
]


OP_NAME = "deform2d_multiscale_fused"
OP_LIBRARY: typing.Final[typing.LiteralString] = f"deformops::{OP_NAME}"

extension_path = define_library(
    OP_NAME,
    cflags_cuda=(
        "-DCUDA_HAS_FP16=1",
        "-DCUDA_HAS_BF16=1",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--use_fast_math",
    ),
)
forward_op = typing.cast(
    typing.Callable[..., Tensor],
    torch.ops.deformops.deform2d_multiscale_fused_forward,  # type: ignore[attr-defined]
)
backward_op = typing.cast(
    typing.Callable[..., tuple[Tensor, Tensor]],
    torch.ops.deformops.deform2d_multiscale_fused_backward.default,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
)


# ----------------- #
# R E F E R E N C E #
# ----------------- #


def reference(
    values: Tensor,
    shapes: Tensor | typing.Sequence[tuple[int, int]],
    locs: Tensor,
    attn: Tensor,
    points: int = 8,
):
    """
    Implements the forward pass using base PyTorch operators, used as a fallback
    mechanism for CPU. Skips the custom op entirely.
    """

    attn = torch.nn.functional.softmax(attn, -1)  # fused in CUDA
    attn = attn.unflatten(-1, (len(shapes), points))
    return _base_reference(values, shapes, locs, attn)


# ------------- #
# T U N I N G S #
# ------------- #


@functools.cache
def _forward_stridethread(B, Q, G, C):
    r"""
    Heuristic for choosing the forward stride and number of threads.
    """
    d_stride = 8
    multiplier = 1
    for m in find_factors(B * Q):
        if m <= 64 and (m * G * C // d_stride) <= 512:  # noqa: PLR2004
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


@functools.cache
def _backward_stridethread(B, Q, G, C):
    r"""
    Heuristic for choosing the backward stride and number of threads.
    """
    d_stride = 2 if C >= 64 else 1  # noqa: PLR2004
    multiplier = 1
    for m in find_factors(B * Q):
        if m <= 64 and (m * G * C // d_stride) <= 256:  # noqa: PLR2004
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


# ------------- #
# F O R W A R D #
# ------------- #


def forward(  # noqa: PLR0913
    values: Tensor,
    shapes: Tensor,
    start_index: Tensor,
    loc: Tensor,
    attn: Tensor,
    im2col_step: int = 64,
    points: int = 8,
) -> Tensor:
    r"""
    Multi-scale deformable attention operator.
    """
    if values.device.type != "cuda" or forward_op is None:
        return reference(values, shapes, loc, attn, points)

    # Concatenate locs and attn
    loc = loc.flatten(-3)
    loc_attn = torch.cat([loc, attn], dim=-1)

    # Lookup strides and block/thread
    stride_fwd, blkthd_fwd = _forward_stridethread(
        values.shape[0], loc_attn.shape[1], values.shape[2], values.shape[3]
    )

    # CUDA forward
    with torch.autocast("cuda", enabled=False):
        result = forward_op(
            values,
            shapes,
            start_index,
            loc_attn.type_as(values),
            im2col_step,
            points,
            stride_fwd,
            blkthd_fwd,
        )
        result = result.type_as(values)
    return result  # noqa: RET504


# --------------- #
# A U T O G R A D #
# --------------- #


def setup_context(ctx, inputs, output):
    r"""
    Setup the context for the multi-scale deformable attention module, for use
    in autograd.
    """
    (
        values,
        shapes,
        start_index,
        loc_attn,
        im2col_step,
        points,
        stride_fwd,
        blkthd_fwd,
    ) = inputs
    ctx.im2col_step = im2col_step
    ctx.points = points
    stride_bwd, blkthd_bwd = _backward_stridethread(
        values.shape[0], loc_attn.shape[1], values.shape[2], values.shape[3]
    )
    ctx.stride_bwd = stride_bwd
    ctx.blkthd_bwd = blkthd_bwd
    ctx.save_for_backward(values, shapes, start_index, loc_attn)


@once_differentiable
def backward(ctx, grad_output):
    r"""
    Wrapper for the backward pass of the multi-scale deformable attention module.
    """
    assert backward_op is not None, "Deformable ops not compiled"

    (
        value,
        spatial_shapes,
        level_index,
        loc_attn,
    ) = ctx.saved_tensors
    with torch.autocast("cuda", enabled=False):
        grad_value, grad_sampling_loc_attn = backward_op(
            value.float(),
            spatial_shapes,
            level_index,
            loc_attn.float(),
            grad_output.contiguous().float(),
            ctx.im2col_step,
            ctx.points,
            ctx.stride_bwd,
            ctx.blkthd_bwd,
        )
        grad_value = grad_value.type_as(value)
        grad_sampling_loc_attn = grad_sampling_loc_attn.type_as(loc_attn)

    return grad_value, None, None, grad_sampling_loc_attn, None, None, None, None


torch.library.register_autograd(
    f"{OP_LIBRARY}_forward", backward, setup_context=setup_context
)

# --------------------- #
# F A K E T E N S O R S #
# --------------------- #


@torch.library.register_fake(f"{OP_LIBRARY}_forward")
def _(  # noqa: PLR0913
    value: Tensor,
    spatial_shapes: Tensor,
    start_index: Tensor,
    loc_attn: Tensor,
    im2col_step: int,
    points: int,
    stride_fwd: int,
    blkthd_fwd: int,
):
    """
    Fake function for the multi-scale deformable attention module.

    See: https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#adding-torch-compile-support-for-an-operator
    """

    N, S, M, D = value.shape  # (N, S = H * W, M, D)
    _, Q, _, Z = loc_attn.shape  # (N, Q, M, Z = L * P * 2)
    (L,) = start_index.shape  # (L)
    P = points

    assert loc_attn.shape == (N, Q, M, L * P * 2), loc_attn.shape  # noqa: PLR2004
    assert spatial_shapes.shape == (L, 2), spatial_shapes.shape

    return value.new_empty(N, Q, M * D)


@torch.library.register_fake(f"{OP_LIBRARY}_backward")
def _(
    values: Tensor,
    shapes: Tensor,
    start_index: Tensor,
    loc_attn: Tensor,
    grad_output: Tensor,
    im2col_step: int,
) -> tuple[Tensor, Tensor]:
    N, S, M, D = values.shape  # [N, S = H * W, M, D]
    _, Q, _, Z = loc_attn.shape  # [N, Q, M, L * P * 3]

    assert grad_output.shape == (N, Q, M * D)

    return (
        grad_output.new_empty((N, S, M, D)),  # values
        grad_output.new_empty((N, Q, M, Z)),  # loc_attn
    )
