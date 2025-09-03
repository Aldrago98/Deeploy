/* =====================================================================
 * Title:        Conv.c
 * Description:  Float32 version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         05.06.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Conv2d_fp32_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                    uint32_t H, uint32_t W, uint32_t C,
                                    const float32_t *__restrict__ pSrcB,
                                    uint32_t F_total, uint32_t P, uint32_t Q,
                                    uint32_t SP, uint32_t SQ,
                                    float32_t *__restrict__ pDstC,
                                    uint32_t pad_top, uint32_t pad_bottom,
                                    uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h = 0; h < H_out; ++h) {
    for (uint32_t w = 0; w < W_out; ++w) {
      for (uint32_t f = 0; f < ch_out_count; ++f) {
        float32_t sum = 0.0f;

        for (uint32_t p = 0; p < P; ++p) {
          for (uint32_t q = 0; q < Q; ++q) {
            for (uint32_t c = 0; c < C; ++c) {
              int32_t h_in = h * SP + p - pad_top;
              int32_t w_in = w * SQ + q - pad_left;

              if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
                  w_in >= (int32_t)W) {
                continue;
              }

              uint32_t input_idx = (h_in * W + w_in) * C + c;
              uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

              sum += pSrcA[input_idx] * weight_ptr[weight_idx];
            }
          }
        }

        uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
        pDstC[output_idx] = sum;
      }
    }
  }
}

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ, float32_t *__restrict__ pDstC,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
    uint32_t pad_right, float32_t *__restrict__ pContextBuffer) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  for (uint32_t h_out = 0; h_out < H_out; h_out++) {
    for (uint32_t w_out = 0; w_out < W_out; w_out++) {
      int32_t h_in_start = h_out * SP - pad_top;
      int32_t w_in_start = w_out * SQ - pad_left;

      for (uint32_t p = 0; p < P; p++) {
        int32_t h_in = h_in_start + p;

        for (uint32_t q = 0; q < Q; q++) {
          int32_t w_in = w_in_start + q;

          for (uint32_t c = 0; c < C; c++) {
            if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                w_in < (int32_t)W) {
              uint32_t in_idx = (h_in * W + w_in) * C + c;
              im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
            } else {
              im2col_buffer[p * Q * C + q * C + c] = 0.0f;
            }
          }
        }
      }

      for (uint32_t f = 0; f < ch_out_count; f++) {
        float32_t sum = 0.0f;
        const float32_t *local_weight_ptr = weight_ptr + f * kernel_size;

        for (uint32_t k = 0; k < kernel_size; k++) {
          sum += im2col_buffer[k] * local_weight_ptr[k];
        }

        uint32_t out_idx =
            (h_out * W_out + w_out) * F_total + (ch_out_start + f);
        pDstC[out_idx] = sum;
      }
    }
  }
}

// standard Conv1d

/* void PULP_Conv1d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, // Input: [L, C]
    uint32_t L,                          // Input length
    uint32_t C,                          // Input channels
    const float32_t *__restrict__ pSrcB, // Weights: [F_total, C, K]
    uint32_t F_total,                    // Output channels
    uint32_t K,                          // Kernel size
    uint32_t S,                          // Stride
    const float32_t *__restrict__ pBias, // può essere NULL
    float32_t *__restrict__ pDstC,       // Output: [L_out, F_total]
    uint32_t pad_left,                   // Padding left
    uint32_t pad_right)                  // Padding right
{

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const float32_t *weight_ptr = pSrcB + ch_out_start * C * K;

  // Output length
  uint32_t L_out = (L + pad_left + pad_right - K) / S + 1;

  for (uint32_t l = 0; l < L_out; ++l) {
    for (uint32_t f = 0; f < ch_out_count; ++f) {
      float32_t sum = 0.0f;

      // convoluzione
      for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t c = 0; c < C; ++c) {
          int32_t l_in = l * S + k - pad_left;

          if (l_in < 0 || l_in >= (int32_t)L) {
            continue;
          }

          uint32_t input_idx = l_in * C + c;
          uint32_t weight_idx = f * (K * C) + k * C + c;

          sum += pSrcA[input_idx] * weight_ptr[weight_idx];
        }
      }

      // aggiunta del bias se disponibile
      if (pBias != NULL) {
        sum += pBias[ch_out_start + f];
      }

      uint32_t output_idx = l * F_total + (ch_out_start + f);
      pDstC[output_idx] = sum;
    }
  }
}
 */
// efficient Conv1d
// sfrutta un unroll a 4/8 su C

void PULP_Conv1d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, // [L, C]
    uint32_t L, uint32_t C,
    const float32_t *__restrict__ pSrcB, // [F_total, K, C] o [F_total, C, K] ->
                                         // qui usiamo [F_total, K, C]
    uint32_t F_total, uint32_t K, uint32_t S,
    const float32_t *__restrict__ pBias, // può essere NULL
    float32_t *__restrict__ pDstC,       // [L_out, F_total]
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {
  const int core_id = pi_core_id();
  const int log2Core = log2(NUM_CORES);

  const uint32_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  const uint32_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  const uint32_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  const uint32_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0)
    return;

  const uint32_t L_out = (L + pad_left + pad_right - K) / S + 1;
  const float32_t *__restrict__ weight_ptr = pSrcB + ch_out_start * (K * C);

  for (uint32_t l = 0; l < L_out; ++l) {

    // range k valido per questo l (niente branch dentro c)
    const int32_t base_in = (int32_t)(l * S) - (int32_t)pad_left;
    uint32_t k_start = 0;
    if (base_in < 0)
      k_start = (uint32_t)(-base_in); // primo k che entra in [0, L)
    uint32_t k_end = K;
    if (base_in + (int32_t)K > (int32_t)L)
      k_end = (uint32_t)(L - base_in);

    for (uint32_t f = 0; f < ch_out_count; ++f) {
      float32_t acc = (pBias ? pBias[ch_out_start + f] : 0.0f);

      // porzione di pesi del filtro f
      const float32_t *__restrict__ w_f = weight_ptr + f * (K * C);

      // k valido
      for (uint32_t k = k_start; k < k_end; ++k) {
        const float32_t *__restrict__ xk =
            pSrcA + (base_in + (int32_t)k) * C;         // [C]
        const float32_t *__restrict__ wk = w_f + k * C; // [C]

        // dot product su C (unroll a 4/8 se vuoi)
        uint32_t c = 0;
        for (; c + 3 < C; c += 4) {
          acc += xk[c + 0] * wk[c + 0];
          acc += xk[c + 1] * wk[c + 1];
          acc += xk[c + 2] * wk[c + 2];
          acc += xk[c + 3] * wk[c + 3];
        }
        for (; c < C; ++c) {
          acc += xk[c] * wk[c];
        }
      }

      pDstC[l * F_total + (ch_out_start + f)] = acc;
    }
  }
}

/* void PULP_Conv1d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, // [L, C]
    uint32_t L, uint32_t C,
    const float32_t *__restrict__ pSrcB, // [F_total, C, K]
    uint32_t F_total, uint32_t K, uint32_t S,
    const float32_t *__restrict__ pBias, // può essere NULL
    float32_t *__restrict__ pDstC,
    // [L_out, F_total]
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer // Im2Col buffer (per-core)
) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  const float32_t *weight_ptr = pSrcB + ch_out_start * C * K;

  uint32_t im2col_size_per_core = C * K;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // uint32_t H_out = 1;
  int32_t W_out = (L + pad_left + pad_right - K) / S + 1;
  uint32_t kernel_size = K * C;
  for (uint32_t i = 0; i < im2col_size_per_core; ++i) {
    im2col_buffer[i] = 0.0f;
  }
  __sync_synchronize(); // barriera di memoria
  for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
    // int32_t h_in_start = h_out * 1;
    int32_t w_in_start = w_out * S - pad_left;
    __sync_synchronize(); // barriera di memoria

    // int32_t h_in = h_in_start + p;

    for (uint32_t q = 0; q < K; ++q) {
      int32_t w_in = w_in_start + q;
      // printf("l_out: %d, w_out: %d, w_in_start: %d, q: %d, w_in: %d\n",
      // w_out, w_out, w_in_start, q, w_in); printf("p: %d, q: %d, w_in: %d
      __sync_synchronize(); // barriera di memoria
      for (uint32_t c = 0; c < C; ++c) {
        if (w_in >= 0 && w_in < (int32_t)L) {

          uint32_t in_idx = (w_in)*C + c;
          float32_t buff_in = pSrcA[in_idx];
          im2col_buffer[q * C + c] = buff_in;

        } else {

          im2col_buffer[q * C + c] = 0.0f;
        }
      }
      __sync_synchronize(); // barriera di memoria
    }

    __sync_synchronize(); // barriera di memoria
    for (uint32_t f = 0; f < ch_out_count; ++f) {

      float32_t sum = (pBias != NULL) ? pBias[ch_out_start + f] : 0.0f;

      const float32_t *local_weight_ptr = weight_ptr + f * kernel_size;
      // Debug: stampiamo solo per i primi 2 w_out e primo filtro

      for (uint32_t k = 0; k < kernel_size; k++) {

        sum += im2col_buffer[k] * local_weight_ptr[k];
        __sync_synchronize(); // barriera di memoria
      }

      uint32_t out_idx = (w_out)*F_total + (ch_out_start + f);
      if (out_idx == 0) {
        // printf("bias[%d]: %f\n", ch_out_start + f, pBias[ch_out_start + f]);
      }
      pDstC[out_idx] = sum;
    }
  }
  __sync_synchronize(); // barriera di memoria
} */