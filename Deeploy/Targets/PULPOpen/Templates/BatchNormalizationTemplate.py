from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// BatchNorm1D (Name: ${nodeName}, Op: ${nodeOp}), PULP
PULP_BatchNorm1d_fp32(
    ${data_in},
    ${scale},
    ${bias},
    ${mean},
    ${variance},
    ${data_out},
    ${batch_size},
    ${ch_im_in},
    ${dim_im_in_y}
    
);

""")
