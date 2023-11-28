#pragma once

#include <ATen/OpMathType.h>
#include <c10/util/Exception.h>

// Input validation macros
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIVISIBLE(x, y) \
  TORCH_CHECK(x % y == 0, #x " (", x, ") must be divisible by " #y " (", y, ")")

#define CHECK_INPUT_DIM(x, d) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "-dimensional tensor")

#define CHECK_EQUALS(x, y) \
  TORCH_CHECK(x == y, #x " (", x, ") must be equal to " #y " (", y, ")")

#define CHECK_TENSORCORE(d) \
  TORCH_CHECK(d % 8 == 0,   \
              #d " must be a multiple of 8 for Tensor Core operations")

// Copied from Caffe2
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// Define a type for opmath-enabled scalars
#define opmath_t at::opmath_type<scalar_t>
