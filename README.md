# StructFormer

This repo contains pytorch code for semantic rearrangement.

## Dependencies
- `h5py==2.10`: this specific version is needed.

see `conda_dependencies.yml` for other dependencies. 

## Environments
The code has been tested on ubuntu 18.04 with nvidia driver 460.91, cuda 11.0, python 3.6, and pytorch 1.7.

## Organization
Source code in `/semantic_rearrangement` are mainly organized as:
- data loaders `data`
- models `models`
- training scripts `training`
- inference scripts `evaluation`

Parameters for data loaders and models are defined in `OmegaConf` yaml files stored in `configs`.

Trained models are stored in `/experiments`

## Quick Start

1. Modify `{structure}_dirs.yaml` files (e.g., `circle_dirs.yaml`) in `/semantic_rearrangement/configs` to point to the correct location of data on
your computer. 
2. Add the python path: `export $PYTHONPATH=/path/to/semantic_rearrangement:$PYTHONPATH`

### Using pretrained models

Download pretrained models from ...

### Pose Generation Networks
```bash
python semantic_rearrangement/evaluation/test_{model_name}.py
```

### Object Selection Network
```bash
python semantic_rearrangement/evaluation/test_object_selection_network.py
```

## Training

### Pose Generation Networks
```bash
# train StructFormer
python semantic_rearrangement/training/train_{model_name}.py
```

### Object Selection Network
```bash
python semantic_rearrangement/training/train_object_selection_network.py
```

