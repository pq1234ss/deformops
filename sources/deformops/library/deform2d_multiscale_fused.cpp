#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deformops, m) {
  m.def(
      "deform2d_multiscale_fused_forward(Tensor value, Tensor spatial_shapes, "
      "Tensor "
      "level_start_index, Tensor sampling_loc_attn, int im2col_step, int K, "
      "int d_stride, int block_thread) -> Tensor");

  m.def(
      "deform2d_multiscale_fused_backward(Tensor value, Tensor spatial_shapes, "
      "Tensor "
      "level_start_index, Tensor sampling_loc_attn, Tensor grad_output, int "
      "im2col_step, int K, int d_stride, int block_thread) -> Tensor[]");
}
