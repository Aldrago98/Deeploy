from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


class TransposedConv1DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel, parseDict, ctxt):
        """
        Geometria 1D Transposed Conv:
          L_out = (L_in - 1) * s - 2*p + d*(k - 1) + output_padding + 1
        Tutto per la dimensione spaziale Y.
        """
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        # Parametri
        strides = parseDict.get("strides", [1])
        pads = parseDict.get("pads", [0, 0])  # (left, right)
        dilations = parseDict.get("dilations", [1])
        out_pad = parseDict.get("output_padding", [0])  # ONNX: lista per dim

        s = strides[0]
        pL, pR = pads
        d = dilations[0]
        op = out_pad[0] if isinstance(out_pad, (list, tuple)) else int(out_pad)

        # Registra le variabili di dimensione
        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Variabili tensori (N, C, Y)
        inN = tilerModel.getTensorDimVar(inputBufferName, 0)
        inC = tilerModel.getTensorDimVar(inputBufferName, 1)
        inY = tilerModel.getTensorDimVar(inputBufferName, 2)

        wOut = tilerModel.getTensorDimVar(weightBufferName, 0)  # Cout
        wIn = tilerModel.getTensorDimVar(weightBufferName, 1)  # Cin
        wK = tilerModel.getTensorDimVar(weightBufferName, 2)  # K

        outN = tilerModel.getTensorDimVar(outputBufferName, 0)
        outC = tilerModel.getTensorDimVar(outputBufferName, 1)
        outY = tilerModel.getTensorDimVar(outputBufferName, 2)
        print(f" TransposedConv1D Geom: inY={inY}, outY={outY}, wK={wK}, s={s}, pL={pL}, pR={pR}, d={d}, op={op}")

        # Mappature batch/canali
        tilerModel.addConstraint(outN == inN)
        tilerModel.addConstraint(outC == wOut)
        tilerModel.addConstraint(inC == wIn)

        # Formula dimensione spaziale per trasposta 1D
        # L_out = (L_in - 1)*s - (pL + pR) + d*(K - 1) + op + 1
        tilerModel.addConstraint(outY == ((inY - 1) * s) - (pL + pR) + (d * (wK - 1)) + op + 1)
        print(f"  Added geom constraint for TransposedConv1D, inY={inY}, outY={outY}")

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel, parseDict, ctxt):
        """
        Manteniamo la stessa policy principale: tiling sul canale di uscita.
        """
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        inYVar = tilerModel.getTensorDimVar(inputBuffer.name, 2)
        inCVar = tilerModel.getTensorDimVar(inputBuffer.name, 1)

        wKVar = tilerModel.getTensorDimVar(weightBuffer.name, 2)
        wInVar = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        wOutVar = tilerModel.getTensorDimVar(weightBuffer.name, 0)

        # Fissa dimensioni note (come nel tuo codice Conv)
        tilerModel.addConstraint(inYVar == parseDict['dim_im_in_y'])
        tilerModel.addConstraint(inCVar == parseDict['ch_im_in'])
        tilerModel.addConstraint(wKVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint(wInVar == parseDict['ch_im_in'])

        # Tiling minimo sul canale di uscita (Cout)
        if parseDict["ch_im_out"] >= 8:
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', wOutVar, 8)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel, parseDict, ctxt):
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)
        return {}  # coerente con la tua API

    # ===== Helpers matematici per gli indici =====
    @staticmethod
    def _ceil_div(a, b):
        # ceil(a/b) che funzioni anche per interi negativi
        return -((-a) // b)

    @staticmethod
    def _floor_div(a, b):
        # floor(a/b)
        return a // b

    @staticmethod
    def computeInputCube(kernelShape, pads, strides, dilations, output_padding, inputCSize, inputYLen, outputCube,
                         outputDims):
        """
        Dato un output tile (gather-based), calcola l'input tile minimo necessario.
        Formula di dipendenza (trasposta 1D):
           y_out = y_in * s - pL + m * d,  con 0 <= m < K
        Quindi, per un range di y_out in [y0, y1], l'intervallo di y_in che contribuisce è:
           y_in_min = ceil((y0 + pL - (K-1)*d) / s)
           y_in_max = floor((y1 + pL) / s)
        Poi si clippa a [0, L_in-1].
        Inoltre calcoliamo "crop_left/right" nel dominio output per il tile,
        da usare in serializzazione come padding effettivo del tile (chiavi riusate).
        """
        (outBN, outY0, outC0) = outputCube.offset
        (outB, outYSize, outC) = outputCube.dims

        K = kernelShape[0]
        pL, pR = pads
        s = strides[0]
        d = dilations[0]
        op = output_padding if isinstance(output_padding, int) else output_padding[0]

        y0 = outY0
        y1 = outY0 + outYSize - 1  # inclusivo

        # Intervallo di input che influenza [y0, y1]
        y_in_min = TransposedConv1DTileConstraint._ceil_div(y0 + pL - (K - 1) * d, s)
        y_in_max = TransposedConv1DTileConstraint._floor_div(y1 + pL, s)

        # Clip all'intervallo valido dell'input
        y_in_min_clip = max(0, y_in_min)
        y_in_max_clip = min(inputYLen - 1, y_in_max)

        if y_in_max_clip < y_in_min_clip:
            # Tile output interamente fuori copertura: nessun input richiesto
            InCube = HyperRectangle((outBN, 0, 0), (outB, 0, inputCSize))
            # Tutto crop lato output
            crop_left = outYSize
            crop_right = 0
            return InCube, (crop_left, crop_right)

        inYOffset = y_in_min_clip
        inYSize = (y_in_max_clip - y_in_min_clip + 1)

        # Costruisci l'input cube (N, Y, C)
        InCube = HyperRectangle((outBN, inYOffset, 0), (outB, inYSize, inputCSize))

        # ---- Calcolo crop/padding effettivo sul tile output ----
        # Ricostruiamo il primo e l’ultimo y_out generabili dal range input tile
        gen_y_out_min = inYOffset * s - pL
        gen_y_out_max = (inYOffset + inYSize - 1) * s - pL + (K - 1) * d

        # Limita al sottointervallo del tile [y0, y1]
        # "crop_left": quante posizioni all'inizio del tile non sono generate
        crop_left = max(0, y0 - gen_y_out_min)
        # "crop_right": quante posizioni alla fine del tile non sono generate
        crop_right = max(0, gen_y_out_max - y1)

        # Nota: op (output_padding) è già inglobato nel calcolo della dimensione out globale;
        # a livello di tile, il crop riflette solo i bordi effettivamente toccati.

        return InCube, (crop_left, crop_right)

    @classmethod
    def serializeTilingSolution(cls, tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation):
        """
        Serializzazione: spec analoga alla tua Conv.
        Attenzione: qui interpretiamo 'padding_y_left/right' come 'crop' nel dominio output
        per il singolo tile della trasposta (valori da passare al kernel tile-wise).
        """
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputWeightCubes = []

        replacements = {
            "dim_im_in_y": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
            "padding_y_left": [],
            "padding_y_right": []
        }

        replacementTypes = {
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_left": PointerClass(uint8_t),
            "padding_y_right": PointerClass(uint8_t)
        }

        weightK = ctxt.lookup(varWeight).shape[2]  # K
        weightCin = ctxt.lookup(varWeight).shape[1]  # Cin
        weightCout = ctxt.lookup(varWeight).shape[0]  # Cout

        pads = operatorRepresentation.get('pads', [0, 0])
        strides = operatorRepresentation.get('strides', [1])
        dilations = operatorRepresentation.get('dilations', [1])
        out_pad = operatorRepresentation.get('output_padding', [0])

        # Lunghezza input Y per il clipping nel computeInputCube
        inYLen = ctxt.lookup(operatorRepresentation['data_in']).shape[1]

        for cube in outputCubes:
            (BN, Y0, C0) = cube.offset
            (B, YS, CS) = cube.dims

            InCube, crop_tuple = cls.computeInputCube((weightK,), pads, strides, dilations, out_pad, weightCin, inYLen,
                                                      cube,
                                                      ctxt.lookup(varOut).shape)

            crop_left, crop_right = crop_tuple

            replacements['dim_im_in_y'].append(InCube.dims[1])  # input tile Y
            replacements['dim_im_out_y'].append(YS)  # output tile Y
            replacements['ch_im_out'].append(CS)
            replacements['padding_y_left'].append(crop_left)
            replacements['padding_y_right'].append(crop_right)

            inputInCubes.append(InCube)

            # Per i pesi: (Cout, Cin, K) limitati al tile dei canali di uscita
            WeightCube = HyperRectangle((C0, 0, 0), (CS, weightCin, weightK))
            inputWeightCubes.append(WeightCube)

        inputLoadSchedule = [{"data_in": a, "weight": b} for a, b in zip(inputInCubes, inputWeightCubes)]
        outputLoadSchedule = [{"data_out": out} for out in outputCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)
        return variableReplacementSchedule, tilingSchedule
