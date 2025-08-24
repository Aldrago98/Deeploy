#include "DeeployBasicMath.h"

void BatchNorm_fp32(const float32_t *input, const float32_t *gamma,
                    const float32_t *beta, const float32_t *mean,
                    const float32_t *var, float32_t *output, int N, int C,
                    int L) {
  const float epsilon = 1e-5f;
#pragma omp parallel for
  for (int c = 0; c < C; ++c) {
    float32_t c_mean = mean[c];
    float32_t c_var = var[c];
    float32_t c_gamma = gamma[c];
    float32_t c_beta = beta[c];
    float32_t denom = sqrtf(c_var + epsilon);
    for (int n = 0; n < N; ++n) {
      for (int l = 0; l < L; ++l) {
        int index = n * C * L + c * L + l;
        float32_t x = input[index];
        float32_t norm = (x - c_mean) / denom;
        output[index] = c_gamma * norm + c_beta;
        printf("n=%d l=%d c=%d idx=%d\n",
                            n, l, c, index + c);
                    printf("   x=%.6f mean=%.6f var=%.6f denom=%.6f\n",
                           x, mean[c], var[c], denom);
                    printf("   norm=%.6f gamma=%.6f beta=%.6f -> y=%.6f\n",
                           norm, gamma[c], beta[c], output[index]);
      }
    }
  }
}
