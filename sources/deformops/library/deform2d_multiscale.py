r"""
Multi-Scale Deform Op
=====================

Implements the multi-scale deformable sampling operator.
"""

import typing
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.nn.functional import grid_sample

from ._build import define_library

__all__ = [
    "reference",
    "backward",
    "forward",
]

OP_NAME = "deform2d_multiscale"
OP_LIBRARY: typing.Final[typing.LiteralString] = f"deformops::{OP_NAME}"

extension_path = define_library(
    OP_NAME,
    cflags_cuda=(
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--use_fast_math",
    ),
)
forward_op = typing.cast(
    typing.Callable[..., Tensor],
    torch.ops.deformops.deform2d_multiscale_forward,  # type: ignore[attr-defined]
)
backward_op = typing.cast(
    typing.Callable[..., tuple[Tensor, Tensor]],
    torch.ops.deformops.deform2d_multiscale_backward.default,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
)


# ----------------- #
# R E F E R E N C E #
# ----------------- #


def reference(
    values: Tensor,
    shapes: Tensor | Sequence[tuple[int, int]],
    locs: Tensor,
    attn: Tensor,
):
    """
    Implements the forward pass using base PyTorch operators, used as a fallback
    mechanism for CPU. Skips the custom op entirely.
    """

    N, _, M, D = values.shape
    _, Q, _, L, P, _ = locs.shape

    assert locs.shape == (N, Q, M, L, P, 2), f"{locs.shape} != {(N, Q, M, L, P, 2)}"  # noqa: PLR2004
    assert attn.shape == (N, Q, M, L, P), f"{attn.shape} != {(N, Q, M, L, P)}"  # noqa: PLR2004

    locs = 2 * locs - 1
    attn = attn.transpose(1, 2).reshape(N * M, 1, Q, L * P)
    values = torch.stack(
        [
            grid_sample(
                v.flatten(2)
                .transpose(1, 2)
                .view(N * M, D, -1)
                .unflatten(
                    -1, shape.tolist() if isinstance(shape, torch.Tensor) else shape
                ),
                g.transpose(1, 2).flatten(0, 1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            for v, g, shape in zip(
                values.split([H * W for H, W in shapes], dim=1),  # split values
                locs.unbind(
                    3
                ),  # unbind locs in L dimension to get L grids (N, Q, M, P, 2)
                shapes,
                strict=True,
            )
        ],
        dim=-2,
    ).flatten(-2)
    return (values * attn).sum(-1).view(N, M * D, Q).transpose(1, 2).contiguous()


# ------------- #
# F O R W A R D #
# ------------- #


def forward(  # noqa: PLR0913
    values: Tensor,
    shapes: Tensor,
    start_index: Tensor,
    locs: Tensor,
    attn: Tensor,
    im2col_step: int = 128,
) -> Tensor:
    r"""
    Multi-scale deformable attention operator.
    """
    if values.device.type == "cuda" and forward_op is not None:
        result = forward_op(
            values.float(),
            shapes,
            start_index,
            locs.float(),
            attn.float(),
            im2col_step,
        )
        return result.type_as(values)
    return reference(values, shapes, locs, attn)


# --------------- #
# A U T O G R A D #
# --------------- #


def setup_context(ctx, inputs, output):
    r"""
    Setup the context for the multi-scale deformable attention module, for use
    in autograd.
    """
    (
        value,
        spatial_shapes,
        start_index,
        loc,
        attn,
        im2col_step,
    ) = inputs
    ctx.im2col_step = im2col_step
    ctx.save_for_backward(value, spatial_shapes, start_index, loc, attn)


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
        loc,
        attn,
    ) = ctx.saved_tensors

    assert value.device.type == "cuda", value.device

    with torch.autocast("cuda", enabled=False):
        grad_values, grad_locs, grad_attn = backward_op(
            value.float(),
            spatial_shapes,
            level_index,
            loc.float(),
            attn.float(),
            grad_output.contiguous().float(),
            ctx.im2col_step,
        )

    return (
        grad_values.type_as(value),
        None,
        None,
        grad_locs.type_as(value),
        grad_attn.type_as(value),
        None,
    )


torch.library.register_autograd(
    f"{OP_LIBRARY}_forward", backward, setup_context=setup_context
)

# --------------------- #
# F A K E T E N S O R S #
# --------------------- #


@torch.library.register_fake(f"{OP_LIBRARY}_forward")
def _(  # noqa: PLR0913
    values: Tensor,
    shapes: Tensor,
    start_index: Tensor,
    locs: Tensor,
    attn: Tensor,
    im2col_step: int,
):
    """
    Implements the forward pass using base PyTorch operators.
    """

    N, S, M, D = values.shape  # [N, S = H * W, M, D]
    _, Q, _, L, P, _ = locs.shape  # [N, Q, M, L, P, 2]

    assert shapes.shape == (L, 2), shapes.shape  # noqa: PLR2004
    assert start_index.shape == (L,), start_index.shape  # noqa: PLR2004
    assert locs.shape == (N, Q, M, L, P, 2), locs.shape  # noqa: PLR2004
    assert attn.shape == (N, Q, M, L, P), attn.shape

    return values.new_empty(N, Q, M * D)


@torch.library.register_fake(f"{OP_LIBRARY}_backward")
def _(
    values: Tensor,
    shapes: Tensor,
    start_index: Tensor,
    locs: Tensor,
    attn: Tensor,
    grad_output: Tensor,
    im2col_step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    N, S, M, D = values.shape  # [N, S = H * W, M, D]
    _, Q, _, L, P, _ = locs.shape  # [N, Q, M, L, P, 2]

    assert grad_output.shape == (N, Q, M * D)
    assert shapes.shape == (L, 2), shapes.shape  # noqa: PLR2004
    assert start_index.shape == (L,), start_index.shape  # noqa: PLR2004
    assert locs.shape == (N, Q, M, L, P, 2), locs.shape  # noqa: PLR2004
    assert attn.shape == (N, Q, M, L, P), attn.shape

    return (
        grad_output.new_empty((N, S, M, D)),  # values
        grad_output.new_empty((N, Q, M, L, P, 2)),  # locs
        grad_output.new_empty((N, Q, M, L, P)),  # attns
    )
