// See: ../deform2d_multiscale_fused.cpp

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <THC/THCAtomics.cuh>
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "interpolate.cuh"
#include "utils.cuh"

namespace deform2d_multiscale_fused {

// Utility functions
constexpr int64_t kWarpSize = 32;

inline int64_t get_blocks(const int64_t N, const int64_t num_threads) {
  return (N + num_threads - 1) / num_threads;
}

inline bool check_backward_warp(int d_stride, int64_t D) {
  int64_t n_group_threads = D / d_stride;
  return (n_group_threads <= kWarpSize) && (kWarpSize % n_group_threads == 0);
}

// Backwards kernel for deformable multi-scale attention sampling
template <typename scalar_t, int64_t d_stride, typename transfer_t, int64_t L,
          int64_t K>
__global__ void
backward_kernel(const scalar_t *p_value, const int64_t *data_spatial_shapes,
                const int64_t *data_level_start_index, const scalar_t *p_offset,
                const scalar_t *grad_output, const int64_t N, const int64_t G,
                const int64_t D, const int64_t Q,
                const int64_t block_multiplier, opmath_t *grad_im,
                opmath_t *grad_offset) {
  extern __shared__ char _s[];

  const int64_t &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int64_t &bi = blockIdx.x * block_multiplier / Q;

  const int64_t &di_s = threadIdx.x * d_stride;
  const int64_t &gi = threadIdx.y;

  opmath_t *cache_g_mask_before_softmax =
      (opmath_t *)(_s); // (block_multiplier*G) * (L * K)
  opmath_t *cache_grad_offset =
      (opmath_t *)(cache_g_mask_before_softmax +
                   block_multiplier * G * L *
                       K); // (block_multiplier*G*D/d_stride*3)
  opmath_t *const p_mask_shm =
      ((opmath_t *)(cache_grad_offset +
                    block_multiplier * G * D / d_stride * 3)) +
      (threadIdx.z * G + gi) * L * K; // G*block_multiplier * L * K

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;
  const int64_t mask_length = L * K;
  const int64_t num_thread = (D / d_stride);
  const int64_t num_iter = mask_length / num_thread;
  const int64_t remainder = mask_length - num_iter * num_thread;
  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * num_iter +
                      threadIdx.x);
  }

  // Calculate softmax over L and K
  __syncthreads();
  if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
    opmath_t softmax_max = -1e100;
    opmath_t softmax_sum = 0.0;
    for (int j = 0; j < L * K; j++) {
      softmax_max = max(softmax_max, p_mask_shm[j]);
    }
    for (int j = 0; j < L * K; j++) {
      opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
      p_mask_shm[j] = exp_results;
      softmax_sum += exp_results;
    }
    for (int j = 0; j < L * K; j++) {
      p_mask_shm[j] /= softmax_sum;
    }
  }

  __syncthreads();

  int64_t offset_idx = 0;
  int64_t mask_idx = 0;
  const int64_t w_stride = G * D;
  const int64_t base_ptr = gi * D + di_s;

  for (int li = 0; li < L; li++) {
    const int64_t spatial_h = data_spatial_shapes[li * 2];
    const int64_t spatial_w = data_spatial_shapes[li * 2 + 1];
    const int64_t level_start_id = data_level_start_index[li];
    const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;
    opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

    int64_t cache_grad_off_idx =
        ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 3;
    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;

      cache_grad_offset[cache_grad_off_idx] = 0;
      cache_grad_offset[cache_grad_off_idx + 1] = 0;
      cache_grad_offset[cache_grad_off_idx + 2] = 0;

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        deform_interpolate::col2im_bilinear<scalar_t, transfer_t, d_stride>(
            p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn, w_stride,
            base_ptr, spatial_h, spatial_w, top_grad, grad_im_ptr,
            cache_grad_offset + cache_grad_off_idx);

        __syncthreads();

        if (threadIdx.x == 0) {
          int64_t _didx = (threadIdx.z * G + threadIdx.y) * blockDim.x * 3;
          opmath_t _grad_w = cache_grad_offset[_didx];
          opmath_t _grad_h = cache_grad_offset[_didx + 1];
          opmath_t _grad_a = cache_grad_offset[_didx + 2];
          for (int c_id = 1; c_id < blockDim.x; ++c_id) {
            _grad_w += cache_grad_offset[_didx + 3 * c_id];
            _grad_h += cache_grad_offset[_didx + 3 * c_id + 1];
            _grad_a += cache_grad_offset[_didx + 3 * c_id + 2];
          }

          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2] = _grad_w;
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2 + 1] = _grad_h;
          cache_g_mask_before_softmax
              [((threadIdx.y + threadIdx.z * G) * L + li) * K + ki] = _grad_a;
        }
      }
      __syncthreads();

      offset_idx += 2;
      mask_idx += 1;
    }
  }

  // Softmax backward
  if (threadIdx.x == 0) {
    for (int i = 0; i < L * K; ++i) {
      opmath_t grad_i = 0.;
      const opmath_t *group_g_mask =
          cache_g_mask_before_softmax + (threadIdx.y + threadIdx.z * G) * L * K;
      for (int j = 0; j < L * K; ++j) {
        if (i != j) {
          grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
        } else {
          grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
        }
      }
      grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + L * K * 2 + i] =
          grad_i;
    }
  }
  __syncthreads();
}

template <typename scalar_t, int64_t d_stride, typename transfer_t, int64_t L,
          int64_t K>
__global__ void backward_kernel_warp_primitive(
    const scalar_t *p_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const scalar_t *p_offset,
    const scalar_t *grad_output, const int64_t N, const int64_t G,
    const int64_t D, const int64_t Q, const int64_t block_multiplier,
    opmath_t *grad_im, opmath_t *grad_offset) {
  extern __shared__ char _s[];

  const int64_t &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int64_t &bi = blockIdx.x * block_multiplier / Q;

  const int64_t &di_s = threadIdx.x * d_stride;
  const int64_t &gi = threadIdx.y;

  // const int64_t tid =
  //     (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  //  const int64_t lane_id = tid % kWarpSize;
  const int64_t group_per_warp = kWarpSize / blockDim.x;
  const int64_t group_in_warp_id =
      (threadIdx.z * G + threadIdx.y) % group_per_warp;
  const unsigned lane_mask = ((1 << blockDim.x) - 1)
                             << (group_in_warp_id * blockDim.x);

  opmath_t *cache_g_mask_before_softmax =
      (opmath_t *)(_s); // (block_multiplier*G) * (L * K)

  opmath_t *const p_mask_shm =
      ((opmath_t *)(cache_g_mask_before_softmax +
                    block_multiplier * G * L * K)) +
      (threadIdx.z * G + gi) * L * K; // G*block_multiplier * L * K

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;
  const int64_t mask_length = L * K;
  const int64_t num_thread = (D / d_stride);
  const int64_t num_iter = mask_length / num_thread;
  const int64_t remainder = mask_length - num_iter * num_thread;
  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * num_iter +
                      threadIdx.x);
  }

  __syncthreads();
  // Calculate softmax over L and K
  if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
    opmath_t softmax_max = -1e100;
    opmath_t softmax_sum = 0.0;

    // get max
    for (int j = 0; j < L * K; j++) {
      softmax_max = max(softmax_max, p_mask_shm[j]);
    }

    // get sumexp
    for (int j = 0; j < L * K; j++) {
      opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
      p_mask_shm[j] = exp_results;
      softmax_sum += exp_results;
    }

    // normalize
    for (int j = 0; j < L * K; j++) {
      p_mask_shm[j] /= softmax_sum;
    }
  }

  __syncthreads();

  int64_t offset_idx = 0;
  int64_t mask_idx = 0;
  const int64_t w_stride = G * D;
  const int64_t base_ptr = gi * D + di_s;

  for (int li = 0; li < L; li++) {
    const int64_t spatial_h = data_spatial_shapes[li * 2];
    const int64_t spatial_w = data_spatial_shapes[li * 2 + 1];
    const int64_t level_start_id = data_level_start_index[li];
    const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;
    opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

    // int64_t cache_grad_off_idx = ((threadIdx.z * G + threadIdx.y) *
    // blockDim.x + threadIdx.x) * 3;

    opmath_t reg_grad_offset[3] = {0.};
    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      reg_grad_offset[0] = 0;
      reg_grad_offset[1] = 0;
      reg_grad_offset[2] = 0;

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        deform_interpolate::col2im_bilinear<scalar_t, transfer_t, d_stride>(
            p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn, w_stride,
            base_ptr, spatial_h, spatial_w, top_grad, grad_im_ptr,
            reg_grad_offset);

        // aggregate across different channel for offset
        for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
          reg_grad_offset[0] +=
              __shfl_down_sync(lane_mask, reg_grad_offset[0], offset);
          reg_grad_offset[1] +=
              __shfl_down_sync(lane_mask, reg_grad_offset[1], offset);
          reg_grad_offset[2] +=
              __shfl_down_sync(lane_mask, reg_grad_offset[2], offset);
        }

        if (threadIdx.x == 0) {
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2] = reg_grad_offset[0];
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2 + 1] = reg_grad_offset[1];
          cache_g_mask_before_softmax
              [((threadIdx.y + threadIdx.z * G) * L + li) * K + ki] =
                  reg_grad_offset[2];
        }
      }
      __syncthreads();

      offset_idx += 2;
      mask_idx += 1;
    }
  }
  // backward for softmax
  if (threadIdx.x == 0) {
    for (int i = 0; i < L * K; ++i) {
      opmath_t grad_i = 0.;
      const opmath_t *group_g_mask =
          cache_g_mask_before_softmax + (threadIdx.y + threadIdx.z * G) * L * K;
      for (int j = 0; j < L * K; ++j) {
        if (i != j) {
          grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
        } else {
          grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
        }
      }
      grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + L * K * 2 + i] =
          grad_i;
    }
  }
  __syncthreads();
}

template <typename scalar_t, typename stride_type, int64_t K, int64_t d_stride>
void _col2im_cuda(cudaStream_t stream,
                  const scalar_t *value,                 // B, N, G, D
                  const int64_t *data_spatial_shapes,    // L * 2
                  const int64_t *data_level_start_index, // L
                  const scalar_t *offset,                // B, N, G, L, K, 3
                  const scalar_t *grad_output,           // B, N, G, D
                  const int64_t B, const int64_t N, const int64_t G,
                  const int64_t D, const int64_t L, const int64_t Q,
                  opmath_t *grad_im, opmath_t *grad_offset,
                  const int64_t block_thread) {
  CHECK_DIVISIBLE(D, d_stride);

  const int64_t block_multiplier = block_thread / (D / d_stride) / G;

  CHECK_DIVISIBLE((B * Q), block_multiplier);

  dim3 num_blocks(B * Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  int64_t shm_size;
  if (check_backward_warp(d_stride, D)) {
    shm_size = sizeof(opmath_t) * (block_multiplier * G * L * K) +
               sizeof(opmath_t) * (G * block_multiplier * L * K);
  } else {
    shm_size = sizeof(opmath_t) * (block_multiplier * G * L * K) +
               sizeof(opmath_t) * (G * block_multiplier * L * K) +
               sizeof(opmath_t) * (G * block_multiplier * D / d_stride * 3);
  }

  auto kernel =
      backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;

  switch (L) {
  case 1:
    if (check_backward_warp(d_stride, D)) {
      kernel =
          backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 1, K>;
    }
    break;
  case 2:
    if (check_backward_warp(d_stride, D)) {
      kernel =
          backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 2, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 2, K>;
    }
    break;
  case 3:
    if (check_backward_warp(d_stride, D)) {
      kernel =
          backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 3, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 3, K>;
    }
    break;
  case 4:
    if (check_backward_warp(d_stride, D)) {
      kernel =
          backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 4, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 4, K>;
    }
    break;
  case 5:
    if (check_backward_warp(d_stride, D)) {
      kernel =
          backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 5, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 5, K>;
    }
    break;
  default:
    printf("L=%ld\n", L);
    throw std::invalid_argument("invalid number of scales");
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, data_spatial_shapes, data_level_start_index, offset, grad_output,
      N, G, D, Q, block_multiplier, grad_im, grad_offset);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in im2col_cuda: %s\n", cudaGetErrorString(err));
    printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d, Q=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size, Q);
    TORCH_CHECK(false, "kernel launch error");
  }
}

template <typename scalar_t, int64_t K>
void col2im_cuda_inner(cudaStream_t stream,
                       const scalar_t *value,                 // B, N, G, D
                       const int64_t *data_spatial_shapes,    // L * 2
                       const int64_t *data_level_start_index, // L
                       const scalar_t *offset,      // B, N, G, L, K, 3
                       const scalar_t *grad_output, // B, N, G, D
                       const int64_t B, const int64_t N, const int64_t G,
                       const int64_t D, const int64_t L, const int64_t Q,
                       opmath_t *grad_im, opmath_t *grad_offset,
                       const int64_t d_stride, const int64_t block_thread) {
  CHECK_DIVISIBLE(D, d_stride);

  if (sizeof(scalar_t) == 2) {
    switch (d_stride) {
    case 1:
      _col2im_cuda<scalar_t, scalar_t, K, 1>(stream,
                                             value,               // B, N, G, D
                                             data_spatial_shapes, // L * 2
                                             data_level_start_index, // L
                                             offset,      // B, N, G, L, K, 3
                                             grad_output, // B, N, G, D
                                             B, N, G, D, L, Q, grad_im,
                                             grad_offset, block_thread);
      break;
    case 2:
      _col2im_cuda<scalar_t, uint, K, 2>(stream,
                                         value,                  // B, N, G, D
                                         data_spatial_shapes,    // L * 2
                                         data_level_start_index, // L
                                         offset,      // B, N, G, L, K, 3
                                         grad_output, // B, N, G, D
                                         B, N, G, D, L, Q, grad_im, grad_offset,
                                         block_thread);
      break;
    case 4:
      _col2im_cuda<scalar_t, uint2, K, 4>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset,      // B, N, G, L, K, 3
                                          grad_output, // B, N, G, D
                                          B, N, G, D, L, Q, grad_im,
                                          grad_offset, block_thread);
      break;
    case 8:
      _col2im_cuda<scalar_t, uint4, K, 8>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset,      // B, N, G, L, K, 3
                                          grad_output, // B, N, G, D
                                          B, N, G, D, L, Q, grad_im,
                                          grad_offset, block_thread);
      break;
    case 16:
      _col2im_cuda<scalar_t, ulonglong4, K, 16>(stream,
                                                value, // B, N, G, D
                                                data_spatial_shapes,    // L * 2
                                                data_level_start_index, // L
                                                offset,      // B, N, G, L, K, 3
                                                grad_output, // B, N, G, D
                                                B, N, G, D, L, Q, grad_im,
                                                grad_offset, block_thread);
      break;
    default:
      printf("not supported for d_stride > 16 for fp16");
      throw std::invalid_argument("invalid d_stride");
    }
  } else {
    TORCH_CHECK(sizeof(scalar_t) == 4, "Expected FP16, BF16 or FP32");

    switch (d_stride) {
    case 1:
      _col2im_cuda<scalar_t, scalar_t, K, 1>(stream,
                                             value,               // B, N, G, D
                                             data_spatial_shapes, // L * 2
                                             data_level_start_index, // L
                                             offset,      // B, N, G, L, K, 3
                                             grad_output, // B, N, G, D
                                             B, N, G, D, L, Q, grad_im,
                                             grad_offset, block_thread);
      break;
    case 2:
      _col2im_cuda<scalar_t, uint2, K, 2>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset,      // B, N, G, L, K, 3
                                          grad_output, // B, N, G, D
                                          B, N, G, D, L, Q, grad_im,
                                          grad_offset, block_thread);
      break;
    case 4:
      _col2im_cuda<scalar_t, uint4, K, 4>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset,      // B, N, G, L, K, 3
                                          grad_output, // B, N, G, D
                                          B, N, G, D, L, Q, grad_im,
                                          grad_offset, block_thread);
      break;
    case 8:
      _col2im_cuda<scalar_t, ulonglong4, K, 8>(stream,
                                               value, // B, N, G, D
                                               data_spatial_shapes,    // L * 2
                                               data_level_start_index, // L
                                               offset,      // B, N, G, L, K, 3
                                               grad_output, // B, N, G, D
                                               B, N, G, D, L, Q, grad_im,
                                               grad_offset, block_thread);
      break;
    default:
      printf("not supported for d_stride > 8 for fp32");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}

template <typename scalar_t>
void col2im_cuda(cudaStream_t stream,
                 const scalar_t *value,                 // B, N, G, D
                 const int64_t *data_spatial_shapes,    // L * 2
                 const int64_t *data_level_start_index, // L
                 const scalar_t *offset,                // B, N, G, L, K, 3
                 const scalar_t *grad_output,           // B, N, G, D
                 const int64_t B, const int64_t N, const int64_t G,
                 const int64_t D, const int64_t L, const int64_t Q,
                 const int64_t K, opmath_t *grad_im, opmath_t *grad_offset,
                 const int64_t d_stride, const int64_t block_thread) {
  switch (K) {
  case 4:
    col2im_cuda_inner<scalar_t, 4>(stream,
                                   value,                  // B, N, G, D
                                   data_spatial_shapes,    // L * 2
                                   data_level_start_index, // L
                                   offset,                 // B, N, G, L, K, 3
                                   grad_output,            // B, N, G, D
                                   B, N, G, D, L, Q, grad_im, grad_offset,
                                   d_stride, block_thread);
    break;
  case 8:
    col2im_cuda_inner<scalar_t, 8>(stream,
                                   value,                  // B, N, G, D
                                   data_spatial_shapes,    // L * 2
                                   data_level_start_index, // L
                                   offset,                 // B, N, G, L, K, 3
                                   grad_output,            // B, N, G, D
                                   B, N, G, D, L, Q, grad_im, grad_offset,
                                   d_stride, block_thread);
    break;
  default:
    printf("not supported for K not in [4, 8]");
    throw std::invalid_argument("invalid K");
  }
}

template <typename scalar_t, int64_t d_stride, typename transfer_t, int64_t L,
          int64_t K>
__global__ void
forward_kernel(const scalar_t *p_value, const int64_t *data_spatial_shapes,
               const int64_t *data_level_start_index, const scalar_t *p_offset,
               scalar_t *p_output, const int64_t N, const int64_t G,
               const int64_t D, const int64_t Q,
               const int64_t block_multiplier) {
  const int64_t &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int64_t &bi = blockIdx.x * block_multiplier / Q;

  const int64_t &di_s = threadIdx.x * d_stride;
  const int64_t &gi = threadIdx.y;

  opmath_t p_out_shm[d_stride] = {0.};
  opmath_t p_mask_shm[L * K] = {0.};

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;

  for (int i = 0; i < L * K; i++) {
    p_mask_shm[i] = *(p_offset_ptr + L * K * 2 + i);
  }

  // Calculate softmax over L and K
  opmath_t softmax_max = -1e100;
  opmath_t softmax_sum = 0.0;

  // get max
  for (int j = 0; j < L * K; j++) {
    softmax_max = max(softmax_max, p_mask_shm[j]);
  }

  // get sumexp
  for (int j = 0; j < L * K; j++) {
    opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
    p_mask_shm[j] = exp_results;
    softmax_sum += exp_results;
  }

  // normalize
  for (int j = 0; j < L * K; j++) {
    p_mask_shm[j] /= softmax_sum;
  }

  int64_t offset_idx = 0;
  int64_t mask_idx = 0;
  const int64_t w_stride = G * D;
  const int64_t base_ptr = gi * D + di_s;

  for (int li = 0; li < L; li++) {
    const int64_t spatial_h = data_spatial_shapes[li * 2];
    const int64_t spatial_w = data_spatial_shapes[li * 2 + 1];
    const int64_t level_start_id = data_level_start_index[li];
    const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        deform_interpolate::im2col_bilinear<scalar_t, transfer_t, d_stride>(
            p_out_shm, p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn,
            w_stride, base_ptr);
      }
      offset_idx += 2;
      mask_idx += 1;
    }
  }

  int64_t out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }

  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, typename stride_type, int64_t K, int64_t d_stride>
void _im2col_cuda(cudaStream_t stream,
                  const scalar_t *value,                 // B, N, G, D
                  const int64_t *data_spatial_shapes,    // L * 2
                  const int64_t *data_level_start_index, // L
                  const scalar_t *offset,                // B, N, G, L, K, 3
                  scalar_t *output,                      // B, N, G, D
                  const int64_t B, const int64_t N, const int64_t G,
                  const int64_t D, const int64_t L, const int64_t Q,
                  const int64_t block_thread) {
  CHECK_DIVISIBLE(D, d_stride);

  const int64_t block_multiplier = block_thread / (D / d_stride) / G;
  CHECK_DIVISIBLE((B * Q), block_multiplier);

  dim3 num_blocks(B * Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  const int64_t shm_size = 0;

  auto kernel = forward_kernel<scalar_t, d_stride, stride_type, 1, K>;

  switch (L) {
  case 1:
    kernel = forward_kernel<scalar_t, d_stride, stride_type, 1, K>;
    break;
  case 2:
    kernel = forward_kernel<scalar_t, d_stride, stride_type, 2, K>;
    break;
  case 3:
    kernel = forward_kernel<scalar_t, d_stride, stride_type, 3, K>;
    break;
  case 4:
    kernel = forward_kernel<scalar_t, d_stride, stride_type, 4, K>;
    break;
  case 5:
    kernel = forward_kernel<scalar_t, d_stride, stride_type, 5, K>;
    break;
  default:
    printf("L=%ld\n", L);
    throw std::invalid_argument("invalid number of scales");
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, data_spatial_shapes, data_level_start_index, offset, output, N, G,
      D, Q, block_multiplier);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in im2col_cuda: %s\n", cudaGetErrorString(err));
    printf("Launch parameters: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d, Q=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size, Q);
    TORCH_CHECK(false, "kernel launch error");
  }
}

template <typename scalar_t, int64_t K>
void im2col_cuda_inner(cudaStream_t stream,
                       const scalar_t *value,                 // B, N, G, D
                       const int64_t *data_spatial_shapes,    // L * 2
                       const int64_t *data_level_start_index, // L
                       const scalar_t *offset, // B, N, G, L, K, 3
                       scalar_t *output,       // B, N, G, D
                       const int64_t B, const int64_t N, const int64_t G,
                       const int64_t D, const int64_t L, const int64_t Q,
                       const int64_t d_stride, const int64_t block_thread) {
  CHECK_DIVISIBLE(D, d_stride);

  if (sizeof(scalar_t) == 2) { // FP16 or BF16
    switch (d_stride) {
    case 1:
      _im2col_cuda<scalar_t, scalar_t, K, 1>(stream,
                                             value,               // B, N, G, D
                                             data_spatial_shapes, // L * 2
                                             data_level_start_index, // L
                                             offset, // B, N, G, L, K, 3
                                             output, // B, N, G, D
                                             B, N, G, D, L, Q, block_thread);
      break;
    case 2:
      _im2col_cuda<scalar_t, uint, K, 2>(stream,
                                         value,                  // B, N, G, D
                                         data_spatial_shapes,    // L * 2
                                         data_level_start_index, // L
                                         offset, // B, N, G, L, K, 3
                                         output, // B, N, G, D
                                         B, N, G, D, L, Q, block_thread);
      break;
    case 4:
      _im2col_cuda<scalar_t, uint2, K, 4>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset, // B, N, G, L, K, 3
                                          output, // B, N, G, D
                                          B, N, G, D, L, Q, block_thread);
      break;
    case 8:
      _im2col_cuda<scalar_t, uint4, K, 8>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset, // B, N, G, L, K, 3
                                          output, // B, N, G, D
                                          B, N, G, D, L, Q, block_thread);
      break;
    case 16:
      _im2col_cuda<scalar_t, ulonglong4, K, 16>(stream,
                                                value, // B, N, G, D
                                                data_spatial_shapes,    // L * 2
                                                data_level_start_index, // L
                                                offset, // B, N, G, L, K, 3
                                                output, // B, N, G, D
                                                B, N, G, D, L, Q, block_thread);
      break;
    default:
      printf("Half-precision requires d_stride in {1, 2, 4, 8, 16}");
      throw std::invalid_argument("invalid d_stride");
    }
  } else { // FP32
    TORCH_CHECK(sizeof(scalar_t) == 4, "Expected FP16, BF16 or FP32");

    switch (d_stride) {
    case 1:
      _im2col_cuda<scalar_t, scalar_t, K, 1>(stream,
                                             value,               // B, N, G, D
                                             data_spatial_shapes, // L * 2
                                             data_level_start_index, // L
                                             offset, // B, N, G, L, K, 3
                                             output, // B, N, G, D
                                             B, N, G, D, L, Q, block_thread);
      break;
    case 2:
      _im2col_cuda<scalar_t, uint2, K, 2>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset, // B, N, G, L, K, 3
                                          output, // B, N, G, D
                                          B, N, G, D, L, Q, block_thread);
      break;
    case 4:
      _im2col_cuda<scalar_t, uint4, K, 4>(stream,
                                          value,                  // B, N, G, D
                                          data_spatial_shapes,    // L * 2
                                          data_level_start_index, // L
                                          offset, // B, N, G, L, K, 3
                                          output, // B, N, G, D
                                          B, N, G, D, L, Q, block_thread);
      break;
    case 8:
      _im2col_cuda<scalar_t, ulonglong4, K, 8>(stream,
                                               value, // B, N, G, D
                                               data_spatial_shapes,    // L * 2
                                               data_level_start_index, // L
                                               offset, // B, N, G, L, K, 3
                                               output, // B, N, G, D
                                               B, N, G, D, L, Q, block_thread);
      break;
    default:
      printf("Full precision requires d_stride in {1, 2, 4, 8}");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}

template <typename scalar_t>
void im2col_cuda(cudaStream_t stream,
                 const scalar_t *value,                 // B, N, G, D
                 const int64_t *data_spatial_shapes,    // L * 2
                 const int64_t *data_level_start_index, // L
                 const scalar_t *offset,                // B, N, G, L, K, 3
                 scalar_t *output,                      // B, N, G, D
                 const int64_t B, const int64_t N, const int64_t G,
                 const int64_t D, const int64_t L, const int64_t Q,
                 const int64_t K, const int64_t d_stride,
                 const int64_t block_thread) {
  switch (K) {
  case 4:
    im2col_cuda_inner<scalar_t, 4>(stream,
                                   value,                  // B, N, G, D
                                   data_spatial_shapes,    // L * 2
                                   data_level_start_index, // L
                                   offset,                 // B, N, G, L, K, 3
                                   output,                 // B, N, G, D
                                   B, N, G, D, L, Q, d_stride, block_thread);
    break;
  case 8:
    im2col_cuda_inner<scalar_t, 8>(stream,
                                   value,                  // B, N, G, D
                                   data_spatial_shapes,    // L * 2
                                   data_level_start_index, // L
                                   offset,                 // B, N, G, L, K, 3
                                   output,                 // B, N, G, D
                                   B, N, G, D, L, Q, d_stride, block_thread);
    break;
  default:
    printf("not supported for K not in [4, 8]");
    throw std::invalid_argument("invalid K");
  }
}

at::Tensor forward_cuda(const at::Tensor &value,
                        const at::Tensor &spatial_shapes,
                        const at::Tensor &level_start_index,
                        const at::Tensor &sampling_loc_attn,
                        const int64_t im2col_step = 64, const int64_t K = 8,
                        const int64_t d_stride = 8,
                        const int64_t block_thread = 0) {
  CHECK_INPUT(value)
  CHECK_INPUT(spatial_shapes)
  CHECK_INPUT(level_start_index)
  CHECK_INPUT(sampling_loc_attn)

  // value (B, H*W, H, C)
  const int64_t batch = value.size(0);
  const int64_t spatial_size = value.size(1);
  const int64_t num_heads = value.size(2);
  const int64_t num_channels = value.size(3);

  // spatial_shapes (L, Q, ...)
  const int64_t num_levels = spatial_shapes.size(0);
  const int64_t num_query = sampling_loc_attn.size(1);
  const int64_t num_point = K;

  // im2col_step
  const int64_t im2col_step_ = std::min(batch, im2col_step);
  CHECK_DIVISIBLE(batch, im2col_step_);

  auto output =
      at::zeros({batch, num_query, num_heads, num_channels}, value.options());

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "deform2d_multiscale_fused::forward_cuda", ([&] {
          im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc_attn.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_offset_size,
              output.data_ptr<scalar_t>() + n * im2col_step_ * per_out_size,
              im2col_step_, spatial_size, num_heads, num_channels, num_levels,
              num_query, num_point, d_stride, block_thread);
        }));
  }
  output = output.view({batch, num_query, num_heads * num_channels});
  return output;
}

std::vector<at::Tensor>
backward_cuda(const at::Tensor &value, const at::Tensor &spatial_shapes,
              const at::Tensor &level_start_index,
              const at::Tensor &sampling_loc_attn,
              const at::Tensor &grad_output, const int64_t im2col_step = 64,
              const int64_t K = 8, const int64_t d_stride = 2,
              const int64_t block_thread = 0) {
  CHECK_INPUT(value)
  CHECK_INPUT(spatial_shapes)
  CHECK_INPUT(level_start_index)
  CHECK_INPUT(sampling_loc_attn)
  CHECK_INPUT(grad_output)

  const int64_t batch = value.size(0);
  const int64_t spatial_size = value.size(1);
  const int64_t num_heads = value.size(2);
  const int64_t num_channels = value.size(3);

  const int64_t num_levels = spatial_shapes.size(0);
  const int64_t num_query = sampling_loc_attn.size(1);
  const int64_t num_point = K;

  const int64_t im2col_step_ = std::min(batch, im2col_step);

  TORCH_CHECK(batch % im2col_step_ == 0, "batch(", batch,
              ") must divide im2col_step(", im2col_step_, ")");

  // Backward is always cast to float32
  auto dtype = at::kFloat;
  auto grad_input = at::zeros_like(value, dtype);
  auto grad_offset = at::zeros_like(sampling_loc_attn, dtype);

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "deform2d_multiscale_fused::backward_cuda", ([&] {
          col2im_cuda(at::cuda::getCurrentCUDAStream(),
                      value.data_ptr<scalar_t>() +
                          n * im2col_step_ * per_value_size,
                      spatial_shapes.data_ptr<int64_t>(),
                      level_start_index.data_ptr<int64_t>(),
                      sampling_loc_attn.data_ptr<scalar_t>() +
                          n * im2col_step_ * per_offset_size,
                      grad_output.data_ptr<scalar_t>() +
                          n * im2col_step_ * per_out_size,
                      im2col_step_, spatial_size, num_heads, num_channels,
                      num_levels, num_query, num_point,
                      grad_input.data_ptr<opmath_t>() +
                          n * im2col_step_ * per_value_size,
                      grad_offset.data_ptr<opmath_t>() +
                          n * im2col_step_ * per_offset_size,
                      d_stride, block_thread);
        }));
  }

  // Cast back to original dtype
  if (value.dtype() != dtype) {
    grad_input = grad_input.to(value.dtype());
    grad_offset = grad_offset.to(value.dtype());
  }
  return {grad_input, grad_offset};
}

} // namespace deform2d_multiscale_fused

// Register CUDA implementation with the PyTorch custom operation dispatcher
TORCH_LIBRARY_IMPL(deformops, CUDA, m) {
  m.impl("deform2d_multiscale_fused_forward",
         &deform2d_multiscale_fused::forward_cuda);
  m.impl("deform2d_multiscale_fused_backward",
         &deform2d_multiscale_fused::backward_cuda);
}
