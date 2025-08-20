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
    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);

    uint16_t ch_out_chunk = (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
    uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
    uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
    uint16_t ch_out_count = ch_out_stop - ch_out_start;

    if (ch_out_count == 0) return;

    // Inizializza l'output locale a zero
    for (uint32_t l = 0; l < L_out; ++l) {
        for (uint32_t f = ch_out_start; f < ch_out_stop; ++f) {
            pDstC[l * F_total + f] = 0.0f;
        }
    }

    // Loop principale della transposed convolution
    for (uint32_t cin = 0; cin < C; ++cin) {
        for (uint32_t l_in = 0; l_in < L; ++l_in) {
            float32_t val = pSrcA[l_in * C + cin];

            for (uint32_t k = 0; k < K; ++k) {
                int32_t l_out = l_in * stride + k - pad_left;

                if (l_out < 0 || l_out >= (int32_t)L_out) continue;

                for (uint32_t cout = ch_out_start; cout < ch_out_stop; ++cout) {
                    // weight index: [cin, cout, k]
                    float32_t wgt = pWeights[cin * (F_total * K) + cout * K + k];
                    pDstC[l_out * F_total + cout] += val * wgt;
                }
            }
        }
    }

    // Aggiunta del bias se disponibile
    if (pBias != NULL) {
        for (uint32_t l = 0; l < L_out; ++l) {
            for (uint32_t f = ch_out_start; f < ch_out_stop; ++f) {
                pDstC[l * F_total + (ch_out_start + f)] += pBias[ch_out_start + f];
            }
        }
    }
}
