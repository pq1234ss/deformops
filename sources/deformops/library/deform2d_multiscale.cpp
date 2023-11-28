#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deformops, m) {
  m.def(
      "deform2d_multiscale_forward(Tensor value, Tensor "
      "spatial_shapes, Tensor "
      "level_start_index, Tensor sampling_loc,Tensor attn_weight, int "
      "im2col_step) -> Tensor");
  m.def(
      "deform2d_multiscale_backward(Tensor value, Tensor "
      "spatial_shapes, Tensor "
      "level_start_index, Tensor sampling_loc, Tensor attn_weight, Tensor "
      "grad_output, int im2col_step) -> Tensor[]");
}
