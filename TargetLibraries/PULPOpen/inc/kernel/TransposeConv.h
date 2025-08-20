#include "DeeployPULPMath.h"

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
    uint32_t pad_right);                    // Padding right