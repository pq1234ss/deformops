# Deformable grid sampling operations

This PyTorch extension implements efficient custom operators for deformable grid
sampling with bilinear interpolation. 
This is used in the [deformable convolutions](https://arxiv.org/abs/1703.06211) 
and [multi-scale deformable attention](https://arxiv.org/abs/2010.04159) modules commonly found in dense vision
networks.

The deformable sampling operation is implemented as a CUDA extension, which is 
JIT-compiled when the operation is first called.

## Installation

The recommended method for installing `deformops` is to simply install the package
from PyPI.

```bash
pip install deformops
```


## Usage

Basic modules are provided in `deformops.nn`:

- `DeformConv2d` (deformable convolution), and
- `DeformAttn2d` (multi-scale deformable attention).

The deformable convolution follows a similar interface to `torch.nn.Conv2d`.
The following initializes the module and passes random initialized data to it:

```python
from deformops.nn import DeformConv2d
from torch import randn

dconv = DeformConv2d(16)

x = randn((4, 16, 32, 64))  # B, C, H, W
y = dconv(x)  # run forward pass
```

Refer to [the documentation](#documentation) for more information.

## About

The core operation in this package -- deformable grid sampling -- was first 
proposed for convolutions in the paper *Deformable Convolutional Networks* (ICCV 2017)
by Dai et al. with Microsoft Research Asia. 
[The paper](https://arxiv.org/abs/1703.06211) proposed two key innovations:

1. **Deformable convolution**, which adds learnable 2D offsets to the regular grid 
    sampling locations in standard convolution, and
2. **Deformable RoI pooling**, which adds learnable offsets to each bin position in 
    regular RoI pooling.

The method uses bilinear interpolation to handle the fractional offsets that arise from
the deformed sampling grid. 
This enabled CNNs to better model geometric transformations by learning the spatial 
sampling locations adaptively from the target tasks.

Multi-scale deformable attention was first introduced in the paper *Deformable DETR: 
Deformable Transformers for End-to-End Object Detection* (ICLR 2020) by Zhu et al. 
[This paper](https://arxiv.org/abs/2010.04159) adapted the deformable convolution 
concept to the attention mechanism used in (vision) transformers, using offsets in the
attention matrix to handle multiple feature scales simultaneously.
