
#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_BatchNorm1d_fp32(const float32_t *__restrict input,
                           const float32_t *__restrict gamma,
                           const float32_t *__restrict beta,
                           const float32_t *__restrict mean,
                           const float32_t *__restrict var,
                           float32_t *__restrict output, int N, int L, int C) {
  const float epsilon = 1e-5f;

// Parallelizza sui due indici esterni (N e L). Ogni (n,l) lavora
// su una riga contigua di C elementi (NHWC), minimizzando conflitti.
#pragma omp parallel for collapse(2)
  for (int n = 0; n < N; ++n) {
    for (int l = 0; l < L; ++l) {

      int base_index = n * L * C + l * C; // NHWC: (n,l) punta a C contigui

      for (int c = 0; c < C; ++c) {
        float x = input[base_index + c];
        float denom = sqrtf(var[c] + epsilon);
        float norm = (x - mean[c]) / denom;
        output[base_index + c] = gamma[c] * norm + beta[c];
      }
    }
  }
}
