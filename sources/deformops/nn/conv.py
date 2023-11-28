"""Implementation of deformable convolutional layers."""

import math  # pyright: ignore[reportShadowedImports]
from typing import TYPE_CHECKING, Final, Literal, cast, override

import torch
from torch import Tensor, nn

import deformops

__all__ = ["DeformConv2d"]


class ScaleLinear(nn.Module):
    r"""Linear scaling layer.

    Simple wrapper around a linear layer for the purpose of having weight and bias
    parameters that are not named "weight" and "bias" to prevent these parameters
    from being picked up by the optimizer for applying weight decay or other
    transformations.

    The output is passed through a sigmoid function to ensure that the scale is
    between 0 and 1.
    """

    scale_weight: nn.Parameter
    scale_bias: nn.Parameter

    def __init__(self, dims: int, group: int):
        super().__init__()

        self.scale_weight = nn.Parameter(
            torch.zeros((group, dims), dtype=torch.float32)
        )
        self.scale_bias = nn.Parameter(torch.zeros((group,), dtype=torch.float32))

    @override
    def forward(self, query: Tensor) -> Tensor:
        return nn.functional.linear(
            query,
            weight=self.scale_weight,
            bias=self.scale_bias,
        ).sigmoid()


class DeformConv2d[Project: nn.Module](nn.Module):
    r"""Deformable convolutional layer."""

    offset_scale: Final[float]
    dims: Final[int]
    kernel_size: Final[int]
    stride: Final[int]
    dilation: Final[int]
    padding: Final[int]
    groups: Final[int]
    group_dims: Final[int]
    center_feature_scale: Final[int]
    remove_center: Final[bool]
    softmax: Final[bool]
    offset_depthwise: nn.Conv2d
    offset_pointwise: nn.Linear
    proj_input: Project | None
    proj_output: Project | None
    norm: nn.Module | None
    activation: nn.Module | None
    center_scale: ScaleLinear | None

    def __init__(
        self,
        dims: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | Literal["same"] = "same",
        dilation: int = 1,
        groups: int = 4,
        offset_scale: float = 1.0,
        center_feature_scale: bool = False,
        remove_center: bool = False,
        project: type[Project] | None = nn.Linear,
        softmax: bool = False,
        norm: nn.Module | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dims : int
            Number of input dims
        kernel_size : int
            Size of the convolving kernel
        stride : int
            Stride of the convolution
        padding : int
            Padding added to both sides of the input
        padding_mode : str
            Padding mode for the convolutions, currently only "zeros" is supported.
        dilation : int
            Spacing between kernel elements
        groups : int
            Number of blocked connections from input dims to output dims
        offset_scale : float
            Scale of the offset
        center_feature_scale : bool
            Whether to use center feature scale
        remove_center : bool
            Whether to remove the center of the kernel
        bias : bool
            Whether to use bias in the output projection
        project : nn.Module
            Projection layer. Defaults to ``nn.Linear``.
        softmax : bool
            Whether to use softmax in the deformable convolution, defaults to False.
        """
        super().__init__()

        if dims % groups != 0:
            msg = f"{dims=} must be divisible by {groups=}"
            raise ValueError(msg)
        if (dims_per_group := dims // groups) % 16 != 0:  # noqa: PLR2004
            msg = f"dims per group ({dims_per_group}) must be divisible by 16"
            raise ValueError(msg)

        self.offset_scale = offset_scale
        self.dims = dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding == "same":
            padding = kernel_size // 2
        self.padding = padding
        self.groups = groups
        self.group_dims = dims // groups
        self.center_feature_scale = center_feature_scale
        self.remove_center = remove_center
        self.softmax = softmax
        self.offset_depthwise = nn.Conv2d(
            dims,
            dims,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            padding_mode="zeros",
            groups=dims,
        )
        kernel_numel = self.groups * (
            self.kernel_size * self.kernel_size - int(self.remove_center)
        )
        self.offset_pointwise = nn.Linear(
            dims, int(math.ceil((kernel_numel * 3) / 8) * 8)
        )

        if project is not None:
            self.proj_input = project(dims, dims)
            self.proj_output = project(dims, dims)
        else:
            self.register_module("proj_input", None)
            self.register_module("proj_output", None)

        self.reset_parameters()

        if center_feature_scale:
            self.center_scale = ScaleLinear(dims, groups)
        else:
            self.register_module("center_scale", None)

        if norm is None:
            self.register_module("norm", None)
        else:
            self.norm = norm

        if activation is None:
            self.register_module("activation", None)
        else:
            self.activation = activation

    def reset_parameters(self):
        # Initialize the offset layers
        for mod in [self.offset_depthwise, self.offset_pointwise]:
            if mod is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue
            if mod.bias is not None:
                _ = nn.init.zeros_(mod.bias.data)
            _ = nn.init.zeros_(mod.weight.data)

        # Initialize the projection layers
        for mod in [self.proj_input, self.proj_output]:
            if mod is None:
                continue
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()  # pyright: ignore[reportCallIssue]
            else:
                if mod.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                    _ = nn.init.zeros_(mod.bias.data)  # pyright: ignore[reportArgumentType]
                _ = nn.init.xavier_uniform_(mod.weight.data)  # pyright: ignore[reportArgumentType]

    def _forward_deform(self, out: Tensor, offset_mask: Tensor) -> Tensor:
        return deformops.deform2d(
            out,
            offset_mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            self.group_dims,
            self.offset_scale,
            256,
            self.remove_center,
            self.softmax,
        )

    @override
    def forward(self, input: Tensor, shape: torch.Size | None = None) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor[N, L, C]
            The input tensor, where N is the batch size, L is the sequence length.
        shape : tuple[H,W]
            Shape of the input tensor if input is in the form of Tensor[N, L, C].
            Optional when the input is in the form of Tensor[N, C, H, W].

        Returns
        -------
        Tensor[N, L, C] or Tensor[N, C, H, W]
            Result of the deformable convolution
        """

        ndim_input = input.ndim
        if ndim_input == 4:  # noqa: PLR2004
            assert shape is None
            shape = input.shape[-2:]
            input = input.flatten(2).permute(0, 2, 1).contiguous()

        N, L, C = input.shape
        assert shape is not None, "shape must be provided"
        H, W = shape

        out = input
        if self.proj_input is not None:
            out = self.proj_input(out)
        out = out.reshape(N, H, W, -1)

        offset_mask_input = cast(
            Tensor, self.offset_depthwise(input.view(N, H, W, C).permute(0, 3, 1, 2))
        )
        offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        offset_mask = cast(Tensor, self.offset_pointwise(offset_mask_input)).reshape(
            N, H, W, -1
        )

        out_ante = out
        out = self._forward_deform(out, offset_mask)

        if self.center_scale is not None:
            center_feature_scale = self.center_scale(out)
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.dims // self.groups)
                .flatten(-2)
            )
            out = out * (1 - center_feature_scale) + out_ante * center_feature_scale
        out = out.view(N, L, -1)

        if self.proj_output is not None:
            out = self.proj_output(out)
        if ndim_input == 4:  # noqa: PLR2004
            out = out.permute(0, 2, 1).unflatten(2, shape).contiguous()
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

    if TYPE_CHECKING:
        __call__ = forward
