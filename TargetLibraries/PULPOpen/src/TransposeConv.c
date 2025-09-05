#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_ConvTranspose1d_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA,    // Input: [L, C]
    uint32_t L,                             // Input length
    uint32_t C,                             // Input channels
    const float32_t *__restrict__ pWeights, // Weights: [C, F_total, K]
    uint32_t F_total,                       // Output channels
    uint32_t K,                             // Kernel size
    uint32_t stride,                        // Stride
    const float32_t *__restrict__ pBias,    // Bias, può essere NULL
    float32_t *__restrict__ pDstC,          // Output: [L_out, F_total]
    uint32_t L_out,                         // Output length
    uint32_t pad_left,                      // Padding left
    uint32_t pad_right)                     // Padding right
{
  int core_id = pi_core_id();
  const int num_cores = NUM_CORES;
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  for (uint32_t c = 0; c < F_total; ++c) {
    for (uint32_t w = 0; w < L_out; ++w) {
      pDstC[c * L_out + w] = 0.0f;
    }
  }
  // Convoluzione trasposta
  printf("Starting ConvTranspose1D: L=%u C=%u F_total=%u K=%u L_out=%u "
         "pad_left=%u pad_right=%u stride=%u\n",
         L, C, F_total, K, L_out, pad_left, pad_right, stride);
  for (uint32_t cin = 0; cin < C; ++cin) {
    for (uint32_t l_in = 0; l_in < L; ++l_in) {
      uint32_t in_idx = cin * L + l_in;
      float32_t val = pSrcA[in_idx];

      for (uint32_t k = 0; k < K; ++k) {
        int l_out = l_in * stride + k - pad_left;
        if (l_out < 0 || l_out >= (int)L_out)
          continue;

        for (uint32_t cout = 0; cout < F_total; ++cout) {
          uint32_t wgt_idx =
              cout * (F_total * K) + cout * K + k; // [Cout, Cin, K]
          uint32_t out_idx = cout * L_out + l_out; // [Cout, L_out]

          float32_t wgt = pWeights[wgt_idx];
          pDstC[out_idx] += val * wgt;
        }
      }
    }
  }

  // Bias (una volta per output channel)
  if (pBias != NULL) {
    for (uint32_t cout = 0; cout < F_total; ++cout) {
      for (uint32_t l_out = 0; l_out < L_out; ++l_out) {
        #pragma nounroll
          for (int j = 0; j < 3; j++) {
            asm volatile("nop" ::);
          }
        uint32_t out_idx = cout * L_out + l_out;
        pDstC[out_idx] += pBias[cout];
        // if (l_out < 20) {
        //   printf("Adding bias: l_out=%u cout=%u bias=%f -> out[%u]=%f\n",
        //   l_out,
        //          cout, pBias[cout],out_idx,
        //          pDstC[0 + out_idx]);
        // }
        printf("out[%u]=%f\n", out_idx, pDstC[out_idx]);
      }
    }
  }
}
