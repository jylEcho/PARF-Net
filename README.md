# PARF-Net: Phase-Adaptive Robust Fusion for Liver Tumor Segmentation with Missing-Phase CECT

Official code of PARF-Net (Published at ESWA2026)

PARFNet is a deep learning model designed for multi-phase image input tasks. This repository provides the core implementation of PARFNet, including the two-phase and three-phase input interfaces, the LGAI and PARM modules, and the Swin Transformer backbone used for global feature extraction.

## Project Structure

```text
PARFNet/
├── parfnet_common.py
├── parfnet_3p.py
├── parfnet_2p.py
├── swin_transformer_list.py
└── README.md
```

## Model Files

### `parfnet_common.py`

This file contains the core architecture of PARFNet.

It includes the main components used in the network, such as:

- Basic convolutional modules
- Swin Transformer backbone adapter
- LGAI module
- PARM module
- Main PARFNet architecture

The `SwinBackboneAdapter` defined in this file is responsible for adapting the Swin Transformer backbone for global feature extraction.

---

### `parfnet_3p.py`

This file provides the **three-phase input interface** for PARFNet.

It imports the PARFNet architecture from `parfnet_common.py` and provides the model entry point for experiments using complete three-phase inputs.

Use this file when conducting experiments with three-phase inputs.

---

### `parfnet_2p.py`

This file provides the **two-phase input interface** for PARFNet.

It calls the PARFNet implementation defined in `parfnet_common.py` and provides the corresponding model entry point for two-phase input experiments.

This version can also be used for comparative or ablation experiments with different numbers of input phases.

---

### `swin_transformer_list.py`

This file provides the **Swin Transformer** backbone used by PARFNet.

The `SwinBackboneAdapter` in `parfnet_common.py` calls the `SwinTransformer` implemented in this file as the global feature extraction backbone.

The relationship between the main files can be summarized as follows:

```text
parfnet_3p.py ─┐
               ├──> parfnet_common.py ───> swin_transformer_list.py
parfnet_2p.py ─┘
```

More specifically:

```text
parfnet_3p.py / parfnet_2p.py
        │
        ▼
     PARFNet
        │
        ▼
SwinBackboneAdapter
        │
        ▼
 SwinTransformer
```

## Input Settings

PARFNet currently provides two input configurations.

### Three-Phase Input

For experiments using three-phase inputs, use:

```text
parfnet_3p.py
```

### Two-Phase Input

For experiments using two-phase inputs, use:

```text
parfnet_2p.py
```

The two-phase version can also be used for comparative or ablation experiments.

## Data Processing

For data preprocessing, dataset organization, and related data processing procedures, please refer to the implementation provided in **CGS-Net**:

https://github.com/jylEcho/CGS-Net

The corresponding data loading and preprocessing pipeline can be adapted according to the input format and the number of phases required by PARFNet.


