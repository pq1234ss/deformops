// See: ../deform2d_multiscale.cpp

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <THC/THCAtomics.cuh>
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "utils.cuh"

namespace deform2d_multiscale {

const int64_t CUDA_NUM_THREADS = 1024;
inline int64_t GET_BLOCKS(const int64_t N, const int64_t num_threads) {
  return (N + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t im2col_bilinear(const scalar_t *&bottom_data,
                                    const int64_t &height, const int64_t &width,
                                    const int64_t &nheads,
                                    const int64_t &channels, const scalar_t &h,
                                    const scalar_t &w, const int64_t &m,
                                    const int64_t &c) {
  const int64_t h_low = floor(h);
  const int64_t w_low = floor(w);
  const int64_t h_high = h_low + 1;
  const int64_t w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh;
  const scalar_t hw = 1 - lw;

  const int64_t w_stride = nheads * channels;
  const int64_t h_stride = width * w_stride;
  const int64_t h_low_ptr_offset = h_low * h_stride;
  const int64_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int64_t w_low_ptr_offset = w_low * w_stride;
  const int64_t w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int64_t base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int64_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int64_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int64_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int64_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ void
col2im_bilinear(const scalar_t *&bottom_data, const int64_t &height,
                const int64_t &width, const int64_t &nheads,
                const int64_t &channels, const scalar_t &h, const scalar_t &w,
                const int64_t &m, const int64_t &c, const scalar_t &top_grad,
                const scalar_t &attn_weight, scalar_t *&grad_value,
                scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int64_t h_low = floor(h);
  const int64_t w_low = floor(w);
  const int64_t h_high = h_low + 1;
  const int64_t w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int64_t w_stride = nheads * channels;
  const int64_t h_stride = width * w_stride;
  const int64_t h_low_ptr_offset = h_low * h_stride;
  const int64_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int64_t w_low_ptr_offset = w_low * w_stride;
  const int64_t w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int64_t base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int64_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int64_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int64_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int64_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename scalar_t>
__device__ void
col2im_bilinear_gm(const scalar_t *&bottom_data, const int64_t &height,
                   const int64_t &width, const int64_t &nheads,
                   const int64_t &channels, const scalar_t &h,
                   const scalar_t &w, const int64_t &m, const int64_t &c,
                   const scalar_t &top_grad, const scalar_t &attn_weight,
                   scalar_t *&grad_value, scalar_t *grad_sampling_loc,
                   scalar_t *grad_attn_weight) {
  const int64_t h_low = floor(h);
  const int64_t w_low = floor(w);
  const int64_t h_high = h_low + 1;
  const int64_t w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int64_t w_stride = nheads * channels;
  const int64_t h_stride = width * w_stride;
  const int64_t h_low_ptr_offset = h_low * h_stride;
  const int64_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int64_t w_low_ptr_offset = w_low * w_stride;
  const int64_t w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int64_t base_ptr = m * channels + c;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_h_weight = 0, grad_w_weight = 0;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int64_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int64_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int64_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int64_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

template <typename scalar_t>
__global__ void deform_im2col_gpu_kernel(
    const int64_t n, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col +=
              im2col_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                              channels, h_im, w_im, m_col, c_col) *
              weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                          channels, h_im, w_im, m_col, c_col, top_grad, weight,
                          grad_value_ptr,
                          cache_grad_sampling_loc + (threadIdx.x << 1),
                          cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int64_t sid = 2;
          for (unsigned int tid = 1; tid < blockSize; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                          channels, h_im, w_im, m_col, c_col, top_grad, weight,
                          grad_value_ptr,
                          cache_grad_sampling_loc + (threadIdx.x << 1),
                          cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deform_col2im_gpu_kernel_shm_reduce_v1(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int64_t _s[];
    scalar_t *cache_grad_sampling_loc = (scalar_t *)_s;
    scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                          channels, h_im, w_im, m_col, c_col, top_grad, weight,
                          grad_value_ptr,
                          cache_grad_sampling_loc + (threadIdx.x << 1),
                          cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          scalar_t _grad_w = cache_grad_sampling_loc[0],
                   _grad_h = cache_grad_sampling_loc[1],
                   _grad_a = cache_grad_attn_weight[0];
          int64_t sid = 2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deform_col2im_gpu_kernel_shm_reduce_v2(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int64_t _s[];
    scalar_t *cache_grad_sampling_loc = (scalar_t *)_s;
    scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                          channels, h_im, w_im, m_col, c_col, top_grad, weight,
                          grad_value_ptr,
                          cache_grad_sampling_loc + (threadIdx.x << 1),
                          cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deform_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int64_t _s[];
    scalar_t *cache_grad_sampling_loc = (scalar_t *)_s;
    scalar_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear(data_value_ptr, spatial_h, spatial_w, num_heads,
                          channels, h_im, w_im, m_col, c_col, top_grad, weight,
                          grad_value_ptr,
                          cache_grad_sampling_loc + (threadIdx.x << 1),
                          cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void deform_col2im_gpu_kernel_gm(
    const int64_t n, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t _temp = index;
    const int64_t c_col = _temp % channels;
    _temp /= channels;
    const int64_t sampling_index = _temp;
    const int64_t m_col = _temp % num_heads;
    _temp /= num_heads;
    // const int64_t q_col = _temp % num_query;
    _temp /= num_query;
    const int64_t b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int64_t data_weight_ptr = sampling_index * num_levels * num_point;
    int64_t data_loc_w_ptr = data_weight_ptr << 1;
    const int64_t grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int64_t grad_weight_stride = 1;
    const int64_t grad_loc_stride = 2;
    const int64_t qid_stride = num_heads * channels;
    const int64_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int64_t level_start_id = data_level_start_index[l_col];
      const int64_t spatial_h_ptr = l_col << 1;
      const int64_t spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int64_t spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int64_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col2im_bilinear_gm(data_value_ptr, spatial_h, spatial_w, num_heads,
                             channels, h_im, w_im, m_col, c_col, top_grad,
                             weight, grad_value_ptr, grad_sampling_loc,
                             grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
void deform_im2col_cuda(cudaStream_t stream,
                        const scalar_t *data_value,            // (B, N, G, D)
                        const int64_t *data_spatial_shapes,    // (L, 2)
                        const int64_t *data_level_start_index, // (L,)
                        const scalar_t *data_sampling_loc,     // (N, L, P, 2)
                        const scalar_t *data_attn_weight,      // (N, L, P)
                        const int64_t batch_size, const int64_t spatial_size,
                        const int64_t num_heads, const int64_t channels,
                        const int64_t num_levels, const int64_t num_query,
                        const int64_t num_point, scalar_t *data_col) {
  const int64_t num_kernels = batch_size * num_query * num_heads * channels;
  const int64_t num_actual_kernels =
      batch_size * num_query * num_heads * channels;
  const int64_t num_threads = CUDA_NUM_THREADS;
  deform_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_level_start_index,
          data_sampling_loc, data_attn_weight, batch_size, spatial_size,
          num_heads, channels, num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deform_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void deform_col2im_cuda(
    cudaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t batch_size, const int64_t spatial_size,
    const int64_t num_heads, const int64_t channels, const int64_t num_levels,
    const int64_t num_query, const int64_t num_point, scalar_t *grad_value,
    scalar_t *grad_sampling_loc, scalar_t *grad_attn_weight) {
  const int64_t num_threads =
      (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
  const int64_t num_kernels = batch_size * num_query * num_heads * channels;
  const int64_t num_actual_kernels =
      batch_size * num_query * num_heads * channels;
  if (channels > 1024) {
    if ((channels & 1023) == 0) {
      deform_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
             num_threads * 3 * sizeof(scalar_t), stream>>>(
              num_kernels, grad_col, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight,
              batch_size, spatial_size, num_heads, channels, num_levels,
              num_query, num_point, grad_value, grad_sampling_loc,
              grad_attn_weight);
    } else {
      deform_col2im_gpu_kernel_gm<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
    }
  } else {
    switch (channels) {
    case 1:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 2:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 4:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 8:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 16:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 32:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 64:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 128:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 256:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 512:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    case 1024:
      deform_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
      break;
    default:
      if (channels < 64) {
        deform_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
               num_threads * 3 * sizeof(scalar_t), stream>>>(
                num_kernels, grad_col, data_value, data_spatial_shapes,
                data_level_start_index, data_sampling_loc, data_attn_weight,
                batch_size, spatial_size, num_heads, channels, num_levels,
                num_query, num_point, grad_value, grad_sampling_loc,
                grad_attn_weight);
      } else {
        deform_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
               num_threads * 3 * sizeof(scalar_t), stream>>>(
                num_kernels, grad_col, data_value, data_spatial_shapes,
                data_level_start_index, data_sampling_loc, data_attn_weight,
                batch_size, spatial_size, num_heads, channels, num_levels,
                num_query, num_point, grad_value, grad_sampling_loc,
                grad_attn_weight);
      }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deform_col2im_cuda: %s\n", cudaGetErrorString(err));
  }
}

at::Tensor forward_cuda(const at::Tensor &value,
                        const at::Tensor &spatial_shapes,
                        const at::Tensor &level_start_index,
                        const at::Tensor &sampling_loc,
                        const at::Tensor &attn_weight,
                        const int64_t im2col_step) {
  CHECK_INPUT(value)
  CHECK_INPUT(spatial_shapes)
  CHECK_INPUT(level_start_index)
  CHECK_INPUT(sampling_loc)
  CHECK_INPUT(attn_weight)

  // value (B, H*W, H, C)
  const int64_t batch = value.size(0);
  const int64_t spatial_size = value.size(1);
  const int64_t num_heads = value.size(2);
  const int64_t channels = value.size(3);

  // spatial_shapes (L, Q, ..., P)
  const int64_t num_levels = spatial_shapes.size(0);
  const int64_t num_query = sampling_loc.size(1);
  const int64_t num_point = sampling_loc.size(4);

  const int64_t im2col_step_ = std::min(batch, im2col_step);

  CHECK_DIVISIBLE(batch, im2col_step_);

  auto output =
      at::zeros({batch, num_query, num_heads, channels}, value.options());

  const int64_t batch_n = im2col_step_;
  auto output_n = output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});

  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "deform2d_multiscale_forward_cuda", ([&] {
          deform_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              batch_n, spatial_size, num_heads, channels, num_levels, num_query,
              num_point, columns.data_ptr<scalar_t>());
        }));
  }

  output = output.view({batch, num_query, num_heads * channels});
  return output;
}

std::vector<at::Tensor>
backward_cuda(const at::Tensor &value, const at::Tensor &spatial_shapes,
              const at::Tensor &level_start_index,
              const at::Tensor &sampling_loc, const at::Tensor &attn_weight,
              const at::Tensor &grad_output, const int64_t im2col_step) {
  CHECK_INPUT(value)
  CHECK_INPUT(spatial_shapes)
  CHECK_INPUT(level_start_index)
  CHECK_INPUT(sampling_loc)
  CHECK_INPUT(attn_weight)
  CHECK_INPUT(grad_output)

  const int64_t batch = value.size(0);
  const int64_t spatial_size = value.size(1);
  const int64_t num_heads = value.size(2);
  const int64_t channels = value.size(3);

  const int64_t num_levels = spatial_shapes.size(0);
  const int64_t num_query = sampling_loc.size(1);
  const int64_t num_point = sampling_loc.size(4);

  const int64_t im2col_step_ = std::min(batch, im2col_step);

  CHECK_DIVISIBLE(batch, im2col_step_);

  auto grad_value = at::zeros_like(value);
  auto grad_sampling_loc = at::zeros_like(sampling_loc);
  auto grad_attn_weight = at::zeros_like(attn_weight);

  const int64_t batch_n = im2col_step_;

  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto grad_output_n = grad_output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto grad_output_g = grad_output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "deform2d_multiscale_backward_cuda", ([&] {
          deform_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                             grad_output_g.data_ptr<scalar_t>(),
                             value.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_value_size,
                             spatial_shapes.data_ptr<int64_t>(),
                             level_start_index.data_ptr<int64_t>(),
                             sampling_loc.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_sample_loc_size,
                             attn_weight.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_attn_weight_size,
                             batch_n, spatial_size, num_heads, channels,
                             num_levels, num_query, num_point,
                             grad_value.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_value_size,
                             grad_sampling_loc.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_sample_loc_size,
                             grad_attn_weight.data_ptr<scalar_t>() +
                                 n * im2col_step_ * per_attn_weight_size);
        }));
  }

  return {grad_value, grad_sampling_loc, grad_attn_weight};
}
} // namespace deform2d_multiscale

// Register CUDA implementation with the PyTorch custom operation dispatcher
TORCH_LIBRARY_IMPL(deformops, CUDA, m) {
  m.impl("deform2d_multiscale_forward", &deform2d_multiscale::forward_cuda);
  m.impl("deform2d_multiscale_backward", &deform2d_multiscale::backward_cuda);
}
