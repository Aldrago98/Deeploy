from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme




class BatchNorm1DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(name=parseDict['data_in'])
        outputBuffer = ctxt.lookup(name=parseDict['data_out'])

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBuffer.name, outputBuffer.name]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName=inputBuffer.name, dimIdx=0)
        inputLengthVar = tilerModel.getTensorDimVar(tensorName=inputBuffer.name, dimIdx=1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName=inputBuffer.name, dimIdx=2)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName=outputBuffer.name, dimIdx=0)
        outputLengthVar = tilerModel.getTensorDimVar(tensorName=outputBuffer.name, dimIdx=1)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName=outputBuffer.name, dimIdx=2)

        # BatchNorm non cambia dimensioni
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputLengthVar == inputLengthVar)
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer = ctxt.lookup(name=parseDict['data_in'])
        inputLengthVar = tilerModel.getTensorDimVar(tensorName=inputBuffer.name, dimIdx=1)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName=inputBuffer.name, dimIdx=2)

        # BatchNorm richiede che i canali coincidano con parametri gamma/beta
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(inputLengthVar >= 1)  # almeno 1 valore

        return tilerModel

    @classmethod
    def serializeTilingSolution(
        cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str, ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        replacements: Dict[str, List[int]] = {
            "dim_im_in_y": [],
            "dim_im_out_y": [],
            "ch_im_in": []
        }

        replacementTypes = {
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t)
        }

        for cube in outputCubes:
            (BatchOffset, YOffset, COffset) = cube.offset
            (BatchSize, YSize, CSize) = cube.dims

            # In BatchNorm input e output hanno la stessa geometria
            InCube = HyperRectangle((BatchOffset, YOffset, COffset), (BatchSize, YSize, CSize))

            replacements['dim_im_in_y'].append(InCube.dims[1])
            replacements['dim_im_out_y'].append(YSize)
            replacements['ch_im_in'].append(CSize)

            inputInCubes.append(InCube)

        inputLoadSchedule = [{"data_in": a} for a in inputInCubes]
        outputLoadSchedule = [{"data_out": out} for out in outputCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
