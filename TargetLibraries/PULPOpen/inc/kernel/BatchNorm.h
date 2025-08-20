#include "DeeployPULPMath.h"

void PULP_BatchNorm1d_fp32(const float32_t *input, const float32_t *gamma,
                           const float32_t *beta, const float32_t *mean,
                           const float32_t *var, float32_t *output, int N,
                           int L, int C);