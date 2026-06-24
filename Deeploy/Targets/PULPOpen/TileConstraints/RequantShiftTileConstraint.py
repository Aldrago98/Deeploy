# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RequantShiftTileConstraint(TileConstraint):

    @staticmethod
    def _channelDim(shape: Tuple[int, ...]) -> int:
        nonUnitDims = [idx for idx, dim in enumerate(shape) if dim != 1]
        if len(nonUnitDims) == 1:
            return nonUnitDims[0]
        return len(shape) - 1

    @staticmethod
    def _inputChannelDim(inputShape: Tuple[int, ...], rqShape: Tuple[int, ...], channelsFirst: bool) -> int:
        rqChannelDim = RequantShiftTileConstraint._channelDim(rqShape)
        rqChannels = rqShape[rqChannelDim]

        if len(inputShape) > 1 and inputShape[1] == rqChannels:
            return 1
        if inputShape[-1] == rqChannels:
            return len(inputShape) - 1
        return 1 if channelsFirst else len(inputShape) - 1

    @staticmethod
    def _requantCube(cube: HyperRectangle, rqShape: Tuple[int, ...], inputChannelDim: int) -> HyperRectangle:
        rqChannelDim = RequantShiftTileConstraint._channelDim(rqShape)
        rqOffset = [0] * len(rqShape)
        rqDims = list(rqShape)
        rqOffset[rqChannelDim] = cube.offset[inputChannelDim]
        rqDims[rqChannelDim] = cube.dims[inputChannelDim]
        return HyperRectangle(tuple(rqOffset), tuple(rqDims))

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = tuple(ctxt.lookup(inputBufferName).shape)
        mulShape = tuple(ctxt.lookup(mulBufferName).shape)
        addShape = tuple(ctxt.lookup(addBufferName).shape)

        mulChannelDim = RequantShiftTileConstraint._channelDim(mulShape)
        addChannelDim = RequantShiftTileConstraint._channelDim(addShape)

        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = mulChannelDim)
        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = addChannelDim)

        tilerModel.addConstraint(mulChannelVar == addChannelVar)

        channels_first = parseDict['channels_first']
        inputChannelDim = RequantShiftTileConstraint._inputChannelDim(inputShape, addShape, channels_first)
        inChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = inputChannelDim)

        tilerModel.addConstraint(mulChannelVar == inChannelVar)

        for dim in range(len(inputShape)):
            inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDimVar == outputDimVar)  # Batch

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputCubes = outputCubes
        inputShape = tuple(ctxt.lookup(operatorRepresentation['data_in']).shape)
        mulShape = tuple(ctxt.lookup(operatorRepresentation['mul']).shape)
        addShape = tuple(ctxt.lookup(operatorRepresentation['add']).shape)
        inputChannelDim = cls._inputChannelDim(inputShape, addShape, operatorRepresentation['channels_first'])

        rqMulCubes = []
        rqAddCubes = []

        replacements = {"size": [], "channel_width": [], "channels": []}
        replacementTypes = {
            "size": PointerClass(uint16_t),
            "channel_width": PointerClass(uint16_t),
            "channels": PointerClass(uint16_t)
        }

        for cube in inputCubes:

            rqMulCubes.append(cls._requantCube(cube, mulShape, inputChannelDim))
            rqAddCubes.append(cls._requantCube(cube, addShape, inputChannelDim))

            channelDim = cube.dims[inputChannelDim]
            size = np.prod(cube.dims[1:])
            channelWidth = size // channelDim
            channels = channelDim

            replacements['size'].append(size)
            replacements['channel_width'].append(channelWidth)
            replacements['channels'].append(channels)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, rqMul, rqAdd in zip(inputCubes, rqMulCubes, rqAddCubes):
            inputLoadSchedule.append({"data_in": a, "add": rqAdd, "mul": rqMul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
