r"""
Deform2d Op
===========

Implements the multi-scale deformable sampling operator.
"""

import functools
import typing

import torch
import torch.nn
from torch import Tensor
from torch.autograd.function import FunctionCtx, once_differentiable

from ._build import define_library
from ._factors import find_factors

__all__ = [
    "backward",
    "forward",
    "setup_context",
]

OP_NAME = "deform2d"
OP_LIBRARY: typing.Final[typing.LiteralString] = "deformops::deform2d"

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
    torch.ops.deformops.deform2d_forward,  # type: ignore[attr-defined]
)
backward_op = typing.cast(
    typing.Callable[..., tuple[Tensor, Tensor]],
    torch.ops.deformops.deform2d_backward.default,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
)

# ----------------- #
# R E F E R E N C E #
# ----------------- #


def reference(
    input: Tensor,
    offset: Tensor,
    mask: Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    group_channels: int,
    offset_scale: int,
    remove_center: bool = False,
) -> Tensor:
    if remove_center:
        msg = f"Keyword argument {remove_center=} is not supported!"
        raise NotImplementedError(msg)
    input = torch.nn.functional.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_IN, W_IN, _ = input.shape
    _, H_OUT, W_OUT, _ = offset.shape

    kernel_size = (kernel_h, kernel_w)
    dilation = (dilation_h, dilation_w)
    stride = (stride_h, stride_w)

    ref = _get_reference_points(
        input.shape,
        kernel_size,
        dilation,
        stride,
        device=input.device,
        dtype=input.dtype,
    )
    grid = _generate_dilation_grids(
        input.shape,
        kernel_size,
        dilation,
        groups,
        device=input.device,
        dtype=input.dtype,
    )
    spatial_norm = (
        torch.tensor([W_IN, H_IN])
        .reshape(1, 1, 1, 2)
        .repeat(1, 1, 1, groups * kernel_h * kernel_w)
        .to(input.device)
    )

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(
        3, 4
    ) + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    input_ = (
        input.view(N_, H_IN * W_IN, groups * group_channels)
        .transpose(1, 2)
        .reshape(N_ * groups, group_channels, H_IN, W_IN)
    )
    sampling_grid_ = (
        sampling_grids.view(N_, H_OUT * W_OUT, groups, P_, 2)
        .transpose(1, 2)
        .flatten(0, 1)
    )
    sampling_input_ = torch.nn.functional.grid_sample(
        input_,
        sampling_grid_,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    mask = (
        mask.view(N_, H_OUT * W_OUT, groups, P_)
        .transpose(1, 2)
        .reshape(N_ * groups, 1, H_OUT * W_OUT, P_)
    )
    output = (
        (sampling_input_ * mask)
        .sum(-1)
        .view(N_, groups * group_channels, H_OUT * W_OUT)
    )

    return output.transpose(1, 2).reshape(N_, H_OUT, W_OUT, -1).contiguous()


def _get_reference_points(
    spatial_shapes: tuple[int, int, int, int] | torch.Size,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    stride: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | torch.types.Device | str = None,
) -> Tensor:
    K_H, K_W = kernel_size
    D_H, D_W = dilation
    S_H, S_W = stride

    _, H_, W_, _ = spatial_shapes
    H_OUT = (H_ - (D_H * (K_H - 1) + 1)) // S_H + 1
    W_OUT = (W_ - (D_W * (K_W - 1) + 1)) // S_W + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (D_H * (K_H - 1)) // 2 + 0.5,
            (D_H * (K_H - 1)) // 2 + 0.5 + (H_OUT - 1) * S_H,
            H_OUT,
            dtype=dtype,
            device=device,
        ),
        torch.linspace(
            (D_W * (K_W - 1)) // 2 + 0.5,
            (D_W * (K_W - 1)) // 2 + 0.5 + (W_OUT - 1) * S_W,
            W_OUT,
            dtype=dtype,
            device=device,
        ),
        indexing="ij",
    )
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    return torch.stack((ref_x, ref_y), -1).reshape(1, H_OUT, W_OUT, 1, 2)


def _generate_dilation_grids(
    spatial_shapes: tuple[int, int, int, int] | torch.Size,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | torch.types.Device | str = None,
) -> Tensor:
    K_H, K_W = kernel_size
    D_H, D_W = dilation
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((D_W * (K_W - 1)) // 2),
            -((D_W * (K_W - 1)) // 2) + (K_W - 1) * D_W,
            K_W,
            dtype=dtype,
            device=device,
        ),
        torch.linspace(
            -((D_H * (K_H - 1)) // 2),
            -((D_H * (K_H - 1)) // 2) + (K_H - 1) * D_H,
            K_H,
            dtype=dtype,
            device=device,
        ),
        indexing="ij",
    )

    points_list.extend([x / W_, y / H_])
    grid = (
        torch.stack(points_list, -1)
        .reshape(-1, 1, 2)
        .repeat(1, groups, 1)
        .permute(1, 0, 2)
    )
    return grid.reshape(1, 1, 1, groups * K_H * K_W, 2)


torch.fx.wrap("_get_reference_points")
torch.fx.wrap("_generate_dilation_grids")

# ------------- #
# T U N I N G S #
# ------------- #


@functools.cache
def _forward_stridethread(B: int, H: int, W: int, G: int, C: int) -> tuple[int, int]:
    r"""
    Heuristic for choosing the forward stride and number of threads.
    """
    d_stride = 8
    multiplier = 1
    for m in find_factors(B * H * W):
        if m <= 64 and (m * G * C // d_stride) <= 512:  # noqa: PLR2004
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    # n_block = (B * H * W + n_thread - 1) // n_thread
    return d_stride, n_thread


@functools.cache
def _backward_stridethread(B: int, H: int, W: int, G: int, C: int) -> tuple[int, int]:
    """
    Heuristic for choosing the backward stride and number of threads.
    """
    d_stride = 2 if C >= 64 else 1  # noqa: PLR2004
    multiplier = 1
    for m in find_factors(B * H * W):
        if m <= 64 and (m * G * C // d_stride) <= 256:  # noqa: PLR2004
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    # n_block = (B * H * W + n_thread - 1) // n_thread
    return d_stride, n_thread


# ------------- #
# F O R W A R D #
# ------------- #


def forward(
    value: Tensor,  # N, H, W, C
    offset_mask: Tensor,  # N, H, W, 2*G*K
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    group: int,
    group_dims: int,
    offset_scale: float,
    im2col_step: int,
    remove_center: bool = False,
    softmax: bool = False,
) -> Tensor:
    assert forward_op is not None, "Deformable ops not compiled"

    fwd_stride, fwd_block_thread = _forward_stridethread(
        *value.shape[:3], group, group_dims
    )

    with torch.autocast("cuda", enabled=False):
        return forward_op(
            value,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_dims,
            offset_scale,
            im2col_step,
            remove_center,
            fwd_stride,
            fwd_block_thread,
            softmax,
        )


# --------------- #
# A U T O G R A D #
# --------------- #
class Deform2dContext(FunctionCtx):
    r"""
    Dummy class that defines the function context for autograd. Instantiation will
    always yield the base class :class:`FunctionCtx`.
    """

    def __new__(cls, *args, **kwargs) -> FunctionCtx:  # pyright: ignore[reportMissingParameterType]
        return FunctionCtx(*args, **kwargs)

    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    group: int
    group_dims: int
    offset_scale: float
    im2col_step: int
    remove_center: bool
    backward_d_stride: int
    backward_block_thread: int
    softmax: bool
    saved_tensors: tuple[Tensor, Tensor]


def setup_context(
    ctx: Deform2dContext,
    inputs: tuple[typing.Any, ...],
    output: Tensor,  # noqa: ARG001
) -> None:
    (
        value,
        offset_mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_dims,
        offset_scale,
        im2col_step,
        remove_center,
        fwd_stride,
        fwd_block_thread,
        softmax,
    ) = inputs
    bck_stride, bck_block_thread = _backward_stridethread(
        *value.shape[:3], group, group_dims
    )

    ctx.kernel_h = kernel_h
    ctx.kernel_w = kernel_w
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.dilation_h = dilation_h
    ctx.dilation_w = dilation_w
    ctx.group = group
    ctx.group_dims = group_dims
    ctx.offset_scale = offset_scale
    ctx.im2col_step = im2col_step
    ctx.remove_center = remove_center
    ctx.backward_d_stride = bck_stride
    ctx.backward_block_thread = bck_block_thread
    ctx.softmax = softmax
    ctx.save_for_backward(value, offset_mask)


@once_differentiable
def backward(
    ctx: Deform2dContext, grad: Tensor
) -> tuple[
    Tensor,
    Tensor,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    assert backward_op is not None, "Deformable ops not compiled"

    input, offset_mask = ctx.saved_tensors
    with torch.autocast("cuda", enabled=False):
        grad_input, grad_offset_mask = backward_op(
            input.float(),
            offset_mask.float(),
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.group,
            ctx.group_dims,
            ctx.offset_scale,
            ctx.im2col_step,
            grad.float().contiguous(),
            ctx.remove_center,
            ctx.backward_d_stride,
            ctx.backward_block_thread,
            ctx.softmax,
        )
    return (
        grad_input.type_as(input),
        grad_offset_mask.type_as(offset_mask),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    f"{OP_LIBRARY}_forward",
    backward,
    setup_context=setup_context,
)

# --------------------- #
# F A K E T E N S O R S #
# --------------------- #


@torch.library.register_fake(f"{OP_LIBRARY}_forward")
def _(
    value: Tensor,
    offset_mask: Tensor,  # noqa: ARG001
    kernel_h: int,  # noqa: ARG001
    kernel_w: int,  # noqa: ARG001
    stride_h: int,  # noqa: ARG001
    stride_w: int,  # noqa: ARG001
    pad_h: int,  # noqa: ARG001
    pad_w: int,  # noqa: ARG001
    dilation_h: int,  # noqa: ARG001
    dilation_w: int,  # noqa: ARG001
    group: int,  # noqa: ARG001
    group_dims: int,  # noqa: ARG001
    offset_scale: float,  # noqa: ARG001
    im2col_step: int,  # noqa: ARG001
    remove_center: bool,  # noqa: ARG001
    fwd_stride: int,  # noqa: ARG001
    fwd_block_thread: int,  # noqa: ARG001
    softmax: bool,  # noqa: ARG001
) -> Tensor:
    return value.new_empty(*value.shape)


@torch.library.register_fake(f"{OP_LIBRARY}_backward")
def _(
    value: Tensor,
    offset_mask: Tensor,
    kernel_h: int,  # noqa: ARG001
    kernel_w: int,  # noqa: ARG001
    stride_h: int,  # noqa: ARG001
    stride_w: int,  # noqa: ARG001
    pad_h: int,  # noqa: ARG001
    pad_w: int,  # noqa: ARG001
    dilation_h: int,  # noqa: ARG001
    dilation_w: int,  # noqa: ARG001
    group: int,  # noqa: ARG001
    group_dims: int,  # noqa: ARG001
    offset_scale: float,  # noqa: ARG001
    im2col_step: int,  # noqa: ARG001
    grad: Tensor,  # noqa: ARG001
    remove_center: bool,  # noqa: ARG001
    fwd_stride: int,  # noqa: ARG001
    fwd_block_thread: int,  # noqa: ARG001
    softmax: bool,  # noqa: ARG001
) -> tuple[Tensor, Tensor]:
    return value.new_empty(*value.shape), offset_mask.new_empty(*offset_mask.shape)
