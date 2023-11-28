r"""Base and implementations for multi-scale deformable attention layers."""

import abc
import enum as E
from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, cast, override

import torch
from torch import Tensor, nn

import deformops

from ._sanitize import CHECK_2POWER, CHECK_DIVISIBLE

__all__ = ["MSDeformAttn2d", "FusedMSDeformAttn2d"]


class MSDeformAttn2dBase(torch.nn.Module, abc.ABC):
    r"""Base class for multi-scale deformable attention layers."""

    dim: Final[int]
    dim_value: Final[int]
    dim_output: Final[int]
    num_heads: Final[int]
    num_levels: Final[int]
    num_points: Final[int]

    proj_offset: nn.Linear
    proj_weights: nn.Linear
    proj_value: nn.Linear
    proj_output: nn.Linear

    def __init__(
        self,
        dim: int,
        dim_value: int | None = None,
        dim_output: int | None = None,
        *,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ):
        r"""
        Parameters
        ----------
        dim
            Size of hidden dimension.
        dim_value
            Size of value dimension, projected to `dim`.
        dim_output
            Size of output dimensions, projected from `dim`.
        num_levels
            Number of feature levels.
        num_heads
            Number of attention num_heads.
        num_points
            Number of sampling points per attention head per feature level.
        """
        super().__init__()

        d_per_head = dim // num_heads

        CHECK_DIVISIBLE(dim, num_heads)
        CHECK_2POWER(d_per_head)

        if dim_value is None:
            dim_value = dim
        if dim_output is None:
            dim_output = dim

        self.dim = dim
        self.dim_value = dim_value
        self.dim_output = dim_output
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.proj_offset = torch.nn.Linear(dim, num_heads * num_levels * num_points * 2)
        self.proj_weights = torch.nn.Linear(dim, num_heads * num_levels * num_points)
        self.proj_value = torch.nn.Linear(dim_value, dim)
        self.proj_output = torch.nn.Linear(dim, dim_output)

        self.reset_parameters()

    def reset_parameters(self):
        H = self.num_heads
        P = self.num_points
        L = self.num_levels

        # Project to offsets
        _ = nn.init.constant_(self.proj_offset.weight.data, 0.0)

        # Initialize the bias for sampling offsets though a circular grid
        # of increasing radii.
        t = torch.arange(H, dtype=torch.float32) * (2.0 * torch.pi / H)
        grid = torch.stack([t.cos(), t.sin()], -1)
        grid = (
            (grid / grid.abs().max(-1, keepdim=True)[0])
            .view(H, 1, 1, 2)
            .repeat(1, L, P, 1)
        )
        for i in range(P):
            grid[:, :, i, :] *= i + 1

        grid = grid.reshape(-1)
        with torch.no_grad():
            _ = self.proj_offset.bias.data.copy_(grid)

        # Initialize the projection layers
        _ = nn.init.constant_(self.proj_weights.weight.data, 0.0)
        _ = nn.init.constant_(self.proj_weights.bias.data, 0.0)
        _ = nn.init.xavier_uniform_(self.proj_value.weight.data)
        _ = nn.init.constant_(self.proj_value.bias.data, 0.0)
        _ = nn.init.xavier_uniform_(self.proj_output.weight.data)
        _ = nn.init.constant_(self.proj_output.bias.data, 0.0)

    @override
    def forward(
        self,
        q: Tensor,
        p: Tensor,
        v: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        q: Tensor[B, Q, C]
            Query tensor
        p: Tensor[B, Q, S, 2] | Tensor[B, Q, S, 4]
            Reference points for each query point, in the format of
            (top-left, bottom-right) or (top, left, h, w).
        v: Tensor[B, S, C]
            Flattened input tensor
        shapes: Tensor[L, 2]
            Spatial shapes of each level
        level_index: Tensor[L]
            Start index of each level in the flattened input tensor
        padding_mask: Tensor[B, S]
            Padding mask for the input tensor

        Returns
        -------
        output : Tensor[N, Q, C]
            The output tensor.
        """
        B, Q, _ = q.shape
        _, _, L, P = p.shape
        _, N, _ = v.shape
        assert B == v.size(0), (q.shape, p.shape, v.shape)
        assert L == self.num_levels, (q.shape, p.shape, v.shape)

        # Sanitization: shapes
        assert (shapes[:, 0] * shapes[:, 1]).sum() == N, (N, shapes)

        # Project values
        v = cast(Tensor, self.proj_value(v))
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], float(0))
        v = v.view(B, N, self.num_heads, self.dim // self.num_heads)

        # Compute attentions
        attn = cast(Tensor, self.proj_weights(q))
        attn = attn.view(B, Q, self.num_heads, self.num_levels * self.num_points)

        # Predict deformable offsets
        loc_off = cast(Tensor, self.proj_offset(q))
        loc_off = loc_off.view(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )
        if P == 2:  # noqa: PLR2004
            loc_norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
            loc = p[:, :, None, :, None, :]
            loc = loc + loc_off / loc_norm[None, None, None, :, None, :]
        elif P == 4:  # noqa: PLR2004
            loc = (
                p[:, :, None, :, None, :2]
                + loc_off / self.num_points * p[:, :, None, :, None, 2:] * 0.5
            )
        else:
            msg = f"Last dim of points must be 2 or 4. Got shape {tuple(p.shape)}."
            raise ValueError(msg)

        # Run sampling operation
        out = self._forward_op(v, shapes, level_index, loc, attn)

        # Project output
        return cast(Tensor, self.proj_output(out))

    @abc.abstractmethod
    def _forward_op(
        self,
        values: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        loc: Tensor,
        attn: Tensor,
    ) -> Tensor:
        r"""
        The core operation of the deformable attention layer.

        Parameters
        ----------
        value: Tensor[B, S, M, D]
            The value tensor.
        shapes: Tensor[L, 2]
            The spatial shapes of each level.
        level_index: Tensor[L]
            The start index of each level in the flattened input tensor.
        loc: Tensor[B, Q, M, L, P, 2]
            The sampling locations.
        attn: Tensor[B, Q, M, L * P]
            The attention weights.

        Returns
        -------
        output : Tensor[N, Q, M * D]
            The output tensor.
        """
        ...

    if TYPE_CHECKING:
        __call__ = forward


class FusedMSDeformAttn2d(MSDeformAttn2dBase):
    r"""Multi-scale deformable attention, with fused softmax."""

    im2col_step: int

    def __init__(
        self,
        dim: int,
        dim_value: int | None = None,
        dim_output: int | None = None,
        *,
        num_heads: int,
        num_levels: int,
        num_points: int = 4,
        im2col_step: int = 64,
    ):
        r"""
        Parameters
        ----------
        dim
            Size of hidden dimension.
        dim_value
            Size of value dimension, projected to `dim`.
        dim_output
            Size of output dimensions, projected from `dim`.
        num_levels
            Number of feature levels.
        num_heads
            Number of attention num_heads.
        num_points
            Number of sampling points per attention head per feature level.
        im2col_step
            Step size for im2col operation. Larger values will result in lower
            memory consumption, but may be slower.
        """
        super().__init__(
            dim,
            dim_value,
            dim_output,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.im2col_step = im2col_step

        deformops.library.load("deform2d_multiscale_fused")

    @override
    def _forward_op(
        self,
        values: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        loc: Tensor,
        attn: Tensor,
    ) -> Tensor:
        return deformops.deform2d_multiscale_fused(
            values,
            shapes,
            level_index,
            loc,
            attn,
            self.im2col_step,
            self.num_points,
        )


class MSDeformAttn2d(MSDeformAttn2dBase):
    r"""Multi-scale deformable attention layer with configurable attention modes."""

    class Method(E.StrEnum):
        r"""
        Attention modes for the deformable attention layer.
        """

        LINEAR = E.auto()
        SOFTMAX = E.auto()
        RECTIFIED = E.auto()

    type MethodType = (
        Method
        | Literal["linear", "softmax", "rectified"]
        | Callable[[Tensor], Tensor]
        | torch.nn.Module
    )

    method: Final[MethodType]
    im2col_step: Final[int]

    def __init__(
        self,
        dim: int,
        dim_value: int | None = None,
        dim_output: int | None = None,
        *,
        num_heads: int,
        num_levels: int,
        num_points: int = 4,
        method: MethodType = Method.SOFTMAX,
        im2col_step: int = 128,
    ):
        r"""
        Parameters
        ----------
        dim
            Size of hidden dimension.
        dim_value
            Size of value dimension, projected to `dim`.
        dim_output
            Size of output dimensions, projected from `dim`.
        num_levels
            Number of feature levels.
        num_heads
            Number of attention num_heads.
        num_points
            Number of sampling points per attention head per feature level.
        im2col_step
            Step size for im2col operation. Larger values will result in lower
            memory consumption, but may be slower.
        """
        super().__init__(
            dim,
            dim_value,
            dim_output,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
        )
        self.im2col_step = im2col_step
        self.method = method

        deformops.library.load("deform2d_multiscale")

    def _generate_attn(self, attn: Tensor) -> Tensor:
        match self.method:
            case self.Method.LINEAR:
                pass
            case self.Method.SOFTMAX:
                attn = torch.nn.functional.softmax(attn, -1)
            case self.Method.RECTIFIED:
                attn = torch.nn.functional.relu(attn)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            case fn if callable(fn):
                attn = fn(attn)
            case _:
                msg = f"Unsupported attention mode {self.method!r}!"
                raise NotImplementedError(msg)
        return attn.unflatten(-1, (self.num_levels, self.num_points))  # type: ignore[return-value]

    @override
    def _forward_op(
        self,
        values: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        loc: Tensor,
        attn: Tensor,
    ) -> Tensor:
        return deformops.deform2d_multiscale(
            values,
            shapes,
            level_index,
            loc,
            self._generate_attn(attn),
            self.im2col_step,
        )
