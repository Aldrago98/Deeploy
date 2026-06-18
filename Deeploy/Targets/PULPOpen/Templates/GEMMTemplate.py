# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULPGEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['B'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['A'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []


PULPGEMM_8_Template = PULPGEMMTemplate("""
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
if (pi_core_id() == 0) {
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
for(int j=0;j<${M};j++){
for(int o=0;o<${O};o++){
int32_t sum = 0;
int8_t* ref_${data_out}_B_o = ref_${data_out}_${B} + (o * ${N});
for(int n=0;n<${N};n++){
sum += (int32_t)ref_${data_out}_${A}[n] * (int32_t)ref_${data_out}_B_o[n];
}
int32_t requant = (sum * ${mul}[o] + ${C}[o]) >> ${log2D};
ref_${data_out}_${data_out}[o] = (int8_t)CLAMP(requant, -128, 127);
}
ref_${data_out}_${A} += ${N};
ref_${data_out}_${data_out} += ${O};
}
% if W_batched:
ref_${data_out}_${B} += ${N} * ${O};
% endif
}
}
pi_cl_team_barrier();
""")


class _MatMulTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['B_offset'] = 0
        operatorRepresentation['C_offset'] = 0

        if hasattr(A, "nLevels"):
            operatorRepresentation['A_offset'] = (A._type.referencedType.typeMin == 0) * int(A.nLevels / 2)
        if hasattr(B, "nLevels"):
            operatorRepresentation['B_offset'] = (B._type.referencedType.typeMin == 0) * int(B.nLevels / 2)
        if hasattr(C, "nLevels"):
            operatorRepresentation['C_offset'] = -(C._type.referencedType.typeMin == 0) * int(C.nLevels / 2)

        return ctxt, operatorRepresentation, []


PULPMM_8_Template = _MatMulTemplate("""
// MatMul (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${A_type.typeName} ref_${data_out}_${A} = ${A};
    ${B_type.typeName} ref_${data_out}_${B} = ${B};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for(uint32_t i=0;i<${batch};i++){
        MatMul_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}(
            ref_${data_out}_${A},
            ref_${data_out}_${B},
            ref_${data_out}_${data_out},
            ${M},
            ${N},
            ${O},
            0, 0, ${C_offset}
        );

        ref_${data_out}_${A} += ${M} * ${N};
        ref_${data_out}_${B} += ${N} * ${O};
        ref_${data_out}_${data_out} += ${M} * ${O};
    }
END_SINGLE_CORE
""")
