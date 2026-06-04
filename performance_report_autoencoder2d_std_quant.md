# Performance Report - Autoencoder2D STD_QUANT

Data: 2026-05-26  
Repo: Deeploy  
Commit testato: `e60ad2e6`  
Target modello: `DeeployTest/Tests/Models/Autoencoder2D/STD_QUANT`

## Obiettivo

Confrontare l'evoluzione delle performance del modello Autoencoder2D quantizzato passando da:

1. Generic
2. Siracusa senza tiling
3. Siracusa tiled
4. Siracusa tiled + Neureka
5. Siracusa tiled + Neureka + WMEM
6. Siracusa tiled + Neureka + Conv3x3
7. Siracusa tiled + Neureka + Conv3x3 + WMEM

La metrica principale disponibile nella pipeline Siracusa/GVSOC e il runtime in cicli. Il runner stampa anche il numero di errori numerici rispetto ai reference output. Non ho trovato una metrica energetica affidabile esposta da questa pipeline: il codice contiene plumbing per misure power, ma i run GVSOC usati qui espongono runtime, errori e, con `--profileMicrobenchmark`, contatori microarchitetturali.

Fonti esterne usate come riferimento generale:

- Deeploy repository: https://github.com/pulp-platform/Deeploy
- Deeploy documentation: https://pulp-platform.github.io/Deeploy/

## Risultati end-to-end

| Caso | Comando sintetico | Esito | Runtime | Delta vs Siracusa tiled | Neureka task | Conv PULP |
|---|---|---:|---:|---:|---:|---:|
| Generic | `deeployRunner_generic.py` | PASS, 0/160 errori | n/a | n/a | 0 | n/a |
| Siracusa non tiled | `deeployRunner_siracusa.py` | FAIL, 34/160 errori | 28,309,212 cicli | +3.13% | 0 | 5 |
| Siracusa tiled | `deeployRunner_tiled_siracusa.py` | PASS, 0/160 errori | 27,449,593 cicli | baseline | 0 | 5 |
| Siracusa tiled + Neureka | `deeployRunner_tiled_siracusa_w_neureka.py` | PASS, 0/160 errori | 28,158,224 cicli | +2.58% | 3 | 4 |
| Siracusa tiled + Neureka + WMEM | `... --neureka-wmem` | PASS, 0/160 errori | 27,084,149 cicli | -1.33% | 3 | 4 |
| Siracusa tiled + Neureka + Conv3x3 | `... --enable-3x3` | PASS, 0/160 errori | 27,524,387 cicli | +0.27% | 7 | 0 |
| Siracusa tiled + Neureka + Conv3x3 + WMEM | `... --enable-3x3 --neureka-wmem` | PASS, 0/160 errori | 26,246,258 cicli | -4.38% | 7 | 0 |

Nota importante: il caso Siracusa senza tiling e diagnostico, non una baseline numericamente valida, perche produce 34 errori su 160. Il runtime e comunque riportato per completezza.

## Comandi usati

```bash
cd /workspaces/Deeploy/DeeployTest
PYTHONPATH=/workspaces/Deeploy python deeployRunner_generic.py -t Tests/Models/Autoencoder2D/STD_QUANT
PYTHONPATH=/workspaces/Deeploy python deeployRunner_siracusa.py -t Tests/Models/Autoencoder2D/STD_QUANT
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa.py -t Tests/Models/Autoencoder2D/STD_QUANT --defaultMemLevel=L3 --l2=3000000
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D/STD_QUANT --defaultMemLevel=L3 --l2=3000000
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D/STD_QUANT --defaultMemLevel=L3 --l2=3000000 --neureka-wmem
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D/STD_QUANT --defaultMemLevel=L3 --l2=3000000 --enable-3x3
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py -t Tests/Models/Autoencoder2D/STD_QUANT --defaultMemLevel=L3 --l2=3000000 --enable-3x3 --neureka-wmem
```

## Interpretazione

Il miglior risultato misurato e:

```text
Siracusa tiled + Neureka + Conv3x3 + WMEM
Runtime: 26,246,258 cicli
Errori: 0/160
Speedup vs Siracusa tiled: 4.38%
```

Neureka senza Conv3x3 peggiora leggermente il runtime. In quel caso vengono accelerati solo pochi layer piccoli, principalmente Gemm/pointwise e la conv finale 1x1, quindi l'overhead di offload e gestione dati pesa piu del guadagno.

L'abilitazione delle Conv3x3 e necessaria per spostare il grosso del lavoro convoluzionale su Neureka. Tuttavia il beneficio diventa netto solo usando anche la weight memory dedicata (`--neureka-wmem`), che riduce il costo di trasferimento/riuso dei pesi.

Il caso Siracusa senza tiling fallisce numericamente. Questo suggerisce che, per questo network quantizzato, la pipeline tiled non e solo una scelta di performance: nel caso corrente e anche la configurazione affidabile per ottenere output corretti.

## Stima statica del lavoro

Dal grafo ONNX:

| Tipo operazione | MAC stimati |
|---|---:|
| Conv | 1,212,800 |
| Gemm | 320,000 |
| ConvTranspose | 128,000 |
| Totale | 1,660,800 |

Distribuzione qualitativa dell'offload:

| Caso | Lavoro su Neureka | Lavoro residuo software/PULP |
|---|---|---|
| Siracusa tiled | nessuno | Conv, Gemm, ConvTranspose, elementwise |
| Neureka senza 3x3 | Gemm/pointwise + conv 1x1 finale | Conv3x3, ConvTranspose, elementwise |
| Neureka con 3x3 | Conv quantizzate + Gemm/pointwise | ConvTranspose, quant/dequant, BatchNorm, Slice/Reshape, elementwise |

## Cicli per MAC approssimati

Questa metrica e solo indicativa, perche il runtime include anche quant/dequant, tiling, DMA, controllo, ConvTranspose, BatchNorm e operazioni elementwise.

| Caso | Cicli/MAC approx |
|---|---:|
| Siracusa non tiled | 17.05 |
| Siracusa tiled | 16.53 |
| Siracusa tiled + Neureka | 16.95 |
| Siracusa tiled + Neureka + WMEM | 16.31 |
| Siracusa tiled + Neureka + Conv3x3 | 16.57 |
| Siracusa tiled + Neureka + Conv3x3 + WMEM | 15.80 |

## Profiling microbenchmark

Con `--profileMicrobenchmark` vengono stampati contatori come:

- cycles
- instructions
- IPC
- load/store count
- branch count
- load stalls
- jump stalls
- instruction cache misses
- TCDM contentions
- external load/store counters

Attenzione: il runtime finale del run profilato non e confrontabile con il runtime end-to-end normale, perche l'instrumentation cambia il punto e la semantica della misura. In un run profilato del caso `--enable-3x3 --neureka-wmem`, il valore finale stampato era `12207 cicli`, quindi non va usato come latenza globale.

Esempi di regioni osservate nel profiling:

| Regione | Cicli | Istruzioni | IPC |
|---|---:|---:|---:|
| Dequant pass | 1,277,833 | 634,098 | 0.496 |
| ConvTranspose 1 | 1,343,673 | 826,773 | 0.615 |
| ConvTranspose 2 | 1,740,343 | 1,096,751 | 0.630 |
| Dequant pass finale | 4,105,544 | 2,283,431 | 0.556 |

Questi numeri confermano che una parte rilevante del tempo resta fuori da Neureka, soprattutto ConvTranspose, quant/dequant e operazioni accessorie.

## Conclusioni

La progressione consigliata da usare come riferimento e:

```text
Generic
-> Siracusa tiled
-> Siracusa tiled + Neureka
-> Siracusa tiled + Neureka + WMEM
-> Siracusa tiled + Neureka + Conv3x3
-> Siracusa tiled + Neureka + Conv3x3 + WMEM
```

Il caso Siracusa non tiled va tenuto nel report, ma non come configurazione valida: fallisce con `34/160` errori.

Il collo di bottiglia residuo non sembra piu essere il forwarding dei pesi 3x3 verso Neureka. Dopo la correzione del formato dei pesi e del tiling constraint, tutte le conv quantizzate vengono offloadate quando `--enable-3x3` e attivo. I limiti principali rimangono:

- ConvTranspose non accelerata da Neureka.
- Quant/dequant e passaggi float residui.
- BatchNorm/elementwise/accessory ops fuori dall'acceleratore.
- Overhead di tiling, DMA e scheduling dei task.

La configurazione migliore oggi e quindi:

```bash
PYTHONPATH=/workspaces/Deeploy python deeployRunner_tiled_siracusa_w_neureka.py \
  -t Tests/Models/Autoencoder2D/STD_QUANT \
  --defaultMemLevel=L3 \
  --l2=3000000 \
  --enable-3x3 \
  --neureka-wmem
```

