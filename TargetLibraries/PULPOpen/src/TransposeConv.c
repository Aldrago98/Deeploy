#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_ConvTranspose1d_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA,    // Input: [W_in, C_in]
    uint32_t W_in,                          // Input width
    uint32_t C_in,                          // Input channels
    const float32_t *__restrict__ pWeights, // Weights: [C_in, C_out, K]
    uint32_t C_out,                         // Output channels
    uint32_t K,                             // Kernel size
    uint32_t stride,                        // Stride
    const float32_t *__restrict__ bias,     // Bias: [C_out] or NULL
    float32_t *__restrict__ pDstC,          // Output: [W_out, C_out]
    uint32_t W_out,                         // Output width
    uint32_t pad_left,                      // Padding left
    uint32_t pad_right                      // Padding right
) {
    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);

    uint16_t ch_out_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
    uint16_t ch_out_start = MIN(ch_out_chunk * core_id, C_out);
    uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, C_out);

    // Initialize output to zero
    for (uint32_t c = ch_out_start; c < ch_out_stop; ++c) {
        for (uint32_t w = 0; w < W_out; ++w) {
            pDstC[w * C_out + c] = 0.0f;
        }
    }

    // Main transposed convolution loop
    for (uint32_t c_out = ch_out_start; c_out < ch_out_stop; ++c_out) {
        for (uint32_t c_in = 0; c_in < C_in; ++c_in) {
            for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
                float32_t val = pSrcA[w_in * C_in + c_in];
                for (uint32_t k = 0; k < K; ++k) {
                    int32_t w_out = w_in * stride + k - pad_left;
                    if (w_out >= 0 && w_out < (int32_t)W_out) {
                        // weight index: [c_in, c_out, k]
                        float32_t wgt = pWeights[c_in * (C_out * K) + c_out * K + k];
                        pDstC[w_out * C_out + c_out] += val * wgt;
                    }
                }
            }
        }
        // Add bias if provided
        if (bias) {
            for (uint32_t w = 0; w < W_out; ++w) {
                pDstC[w * C_out + c_out] += bias[c_out];
            }
        }
    }
}