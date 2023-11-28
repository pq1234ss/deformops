#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(deformops, m) {
  m.def("deform2d_forward(Tensor value, Tensor p_offset, int kernel_h, int "
        "kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int "
        "dilation_h, int dilation_w, int group, int group_channels, float "
        "offset_scale, int im2col_step, bool remove_center, int d_stride, int "
        "block_thread, bool softmax) -> Tensor");

  m.def(
      "deform2d_backward(Tensor value,  Tensor p_offset,  int kernel_h, int "
      "kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int "
      "dilation_h, int dilation_w, int group, int group_channels, float "
      "offset_scale, int im2col_step, Tensor grad_output, bool remove_center, "
      "int d_stride, int block_thread, bool softmax) -> Tensor[]");
}
