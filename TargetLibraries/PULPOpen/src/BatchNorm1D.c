
#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_BatchNorm1d_fp32(const float32_t *input, const float32_t *gamma,
                               const float32_t *beta, const float32_t *mean,
                               const float32_t *var, float32_t *output, int N,
                               int C, int L) {
  const float epsilon = 1e-5f;
  
  // Parallelizziamo su batch e canali
  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  uint16_t ch_out_chunk =
      (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, C);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, C);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }
  for(int c = ch_out_start; c < ch_out_stop; ++c) {
    for (int n = 0; n < N; ++n) {
      for (int l = 0; l < L; ++l) {
        int idx = n * C * L + c * L + l; // NCHW
        output[idx] = 0.0f;
      }
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int c = ch_out_start; c < ch_out_stop; ++c) {
      
      int base_index = n * C * L + c * L; // NCHW: (n,c) punta a L contigui

      float denom = sqrtf(var[c] + epsilon);
      /* printf("\n[DEBUG] Batch %d, Canale %d -> base_index=%d, denom=%.6f\n",
             n, c, base_index, denom); */
      
      for (int l = 0; l < L; ++l) {
        float x = input[base_index + l];
        float norm = (x - mean[c]) / denom;
        int idx = base_index + l;
        float y = gamma[c] * norm + beta[c];
        output[base_index + l] = y;
        //printf("  (n=%d, c=%d, l=%d) idx=%d | x=%.6f "
           //    "-> norm=%.6f -> y=%.6f\n",
            //   n, c, l, idx, x, norm, y);
        if (pi_core_id() == 0) {
                    printf("[core %d] n=%d l=%d c=%d idx=%d\n",
                           pi_core_id(), n, l, c, idx);
                    printf("   x=%.6f mean=%.6f var=%.6f denom=%.6f\n",
                           x, mean[c], var[c], denom);
                    printf("   norm=%.6f gamma=%.6f beta=%.6f -> y=%.6f\n",
                           norm, gamma[c], beta[c], y);
                    }
        
      }
    }
  }


}