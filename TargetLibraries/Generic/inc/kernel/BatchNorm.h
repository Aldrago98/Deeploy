// SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <stdbool.h>
#include <stdint.h>

void BatchNorm_fp32(const float32_t *input, const float32_t *gamma,
                    const float32_t *beta, const float32_t *mean,
                    const float32_t *var, float32_t *output, int N, int C,
                    int L, float epsilon, int channels_first);

void BatchNorm_s8(const int8_t *input, const int8_t *gamma,
                  const int8_t *beta, const int8_t *mean, const int8_t *var,
                  int8_t *output, int N, int C, int L, float epsilon,
                  int channels_first);

#endif // BATCHNORM_H
