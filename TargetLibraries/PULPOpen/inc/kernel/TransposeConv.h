#include "DeeployPULPMath.h"

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
);