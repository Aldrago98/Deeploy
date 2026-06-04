# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _Pad2DTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Align padding value to input signedness

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels") and not data_in._signed:
            operatorRepresentation['value'] = operatorRepresentation['value'] - int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


reference2DTemplate = _Pad2DTemplate("""
<%
    y_offset_out = dim_im_out_ch*(pad_y*dim_im_out_y)
    x_offset_out = dim_im_out_ch*(pad_x)
    width = dim_im_in_ch*dim_im_in_y

    addoffsetOut = dim_im_out_ch * dim_im_out_y
    addoffsetIn = dim_im_in_ch * dim_im_in_y

    startPosX = y_offset_out + x_offset_out

batchOffsetOut = dim_im_out_ch * dim_im_out_x * dim_im_out_y
%>

// 2D Pad (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    % if mode == "constant":
    memset(${data_out}, ${value}, ${data_out_size}*sizeof(${data_out_type.referencedType.typeName}));
    uint32_t xoffset_${data_out}_${data_in};
    uint32_t offset_in_${data_out}_${data_in} = 0;

    % if channels_first:
    // NCHW Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${pad_y}*${dim_im_out_y}+${pad_x};
        for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
            for(uint32_t h=0; h<${dim_im_in_x}; h++){
                memcpy(${data_out} + xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${dim_im_in_y}*sizeof(${data_out_type.referencedType.typeName}));
                xoffset_${data_out}_${data_in} += ${dim_im_out_y};
                offset_in_${data_out}_${data_in} += ${dim_im_in_y};
            }
            xoffset_${data_out}_${data_in} += 2*${pad_y}*${dim_im_out_y};
        }
    }
    % else:
    // NHWC Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${startPosX};
        for(uint32_t h=0; h<${dim_im_in_x}; h++){
            memcpy(${data_out}+xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${width}*sizeof(${data_out_type.referencedType.typeName}));
            xoffset_${data_out}_${data_in} += ${addoffsetOut};
            offset_in_${data_out}_${data_in} += ${addoffsetIn};
        }
    }
    %endif
    % else:
    % if channels_first:
    // NCHW Layout
    for(uint32_t n=0; n<${batch}; n++){
        for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
            for(uint32_t oh=0; oh<${dim_im_out_x}; oh++){
                int32_t ih = (int32_t)oh - ${pad_y};
                if (ih < 0) {
                    ih = 0;
                }
                if (ih >= ${dim_im_in_x}) {
                    ih = ${dim_im_in_x} - 1;
                }
                for(uint32_t ow=0; ow<${dim_im_out_y}; ow++){
                    int32_t iw = (int32_t)ow - ${pad_x};
                    if (iw < 0) {
                        iw = 0;
                    }
                    if (iw >= ${dim_im_in_y}) {
                        iw = ${dim_im_in_y} - 1;
                    }
                    uint32_t out_idx_${data_out}_${data_in} = n*${batchOffsetOut} + c*${dim_im_out_x}*${dim_im_out_y} + oh*${dim_im_out_y} + ow;
                    uint32_t in_idx_${data_out}_${data_in} = n*${dim_im_in_ch}*${dim_im_in_x}*${dim_im_in_y} + c*${dim_im_in_x}*${dim_im_in_y} + ih*${dim_im_in_y} + iw;
                    ${data_out}[out_idx_${data_out}_${data_in}] = ${data_in}[in_idx_${data_out}_${data_in}];
                }
            }
        }
    }
    % else:
    // NHWC Layout
    for(uint32_t n=0; n<${batch}; n++){
        for(uint32_t oh=0; oh<${dim_im_out_x}; oh++){
            int32_t ih = (int32_t)oh - ${pad_y};
            if (ih < 0) {
                ih = 0;
            }
            if (ih >= ${dim_im_in_x}) {
                ih = ${dim_im_in_x} - 1;
            }
            for(uint32_t ow=0; ow<${dim_im_out_y}; ow++){
                int32_t iw = (int32_t)ow - ${pad_x};
                if (iw < 0) {
                    iw = 0;
                }
                if (iw >= ${dim_im_in_y}) {
                    iw = ${dim_im_in_y} - 1;
                }
                for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
                    uint32_t out_idx_${data_out}_${data_in} = n*${batchOffsetOut} + oh*${dim_im_out_y}*${dim_im_out_ch} + ow*${dim_im_out_ch} + c;
                    uint32_t in_idx_${data_out}_${data_in} = n*${dim_im_in_x}*${dim_im_in_y}*${dim_im_in_ch} + ih*${dim_im_in_y}*${dim_im_in_ch} + iw*${dim_im_in_ch} + c;
                    ${data_out}[out_idx_${data_out}_${data_in}] = ${data_in}[in_idx_${data_out}_${data_in}];
                }
            }
        }
    }
    %endif
    %endif
END_SINGLE_CORE
""")


class _Pad1DTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        # Align padding value to input signedness

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels") and not data_in._signed:
            operatorRepresentation['value'] = operatorRepresentation['value'] - int(data_in.nLevels / 2)

        return ctxt, operatorRepresentation, []


reference1DTemplate = _Pad1DTemplate("""
<%
    x_offset_out = dim_im_out_ch*(pad_y)
    width = dim_im_in_ch*dim_im_in_y

    startPosX = x_offset_out

batchOffsetOut = dim_im_out_ch * dim_im_out_y
%>

// 1D Pad (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    memset(${data_out}, ${value}, ${data_out_size}*sizeof(${data_out_type.referencedType.typeName}));
    uint32_t xoffset_${data_out}_${data_in};
    uint32_t offset_in_${data_out}_${data_in} = 0;

    % if channels_first:
    // NCHW Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} +${pad_y};
        for (uint32_t c=0; c<${dim_im_in_ch}; ++c) {
            memcpy(${data_out} + xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${dim_im_in_y}*sizeof(${data_out_type.referencedType.typeName}));
            xoffset_${data_out}_${data_in} += ${dim_im_out_y};
            offset_in_${data_out}_${data_in} += ${dim_im_in_y};
        }
    }
    % else:
    // NHWC Layout
    for(uint32_t n=0; n<${batch}; n++){
        xoffset_${data_out}_${data_in} = n*${batchOffsetOut} + ${startPosX};
        memcpy(${data_out}+xoffset_${data_out}_${data_in}, ${data_in}+offset_in_${data_out}_${data_in}, ${width}*sizeof(${data_out_type.referencedType.typeName}));
        offset_in_${data_out}_${data_in} += ${width};
    }
    %endif
END_SINGLE_CORE
""")
