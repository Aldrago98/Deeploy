# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Integer ReLU (Name: ${nodeName}, Op: ${nodeOp})
for (uint32_t i = 0; i < ${size}; i++) {
    ${data_out}[i] = (${data_in}[i] > 0) ? ${data_in}[i] : 0;
}
""")
