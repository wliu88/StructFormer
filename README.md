# StructFormer

This repo contains pytorch code for semantic rearrangement.

## Dependencies
- `h5py==2.10`: this specific version is needed.
- `omegaconfg==2.1`: some functions used in this repo are from newer versions

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

## Quick Start with Pretrained Models
- Add the package: `export $PYTHONPATH=/path/to/semantic_rearrangement:$PYTHONPATH`
- Download and unzip pretrained models `/models` from [this link](https://drive.google.com/file/d/1EsptihJv_lPND902P6CYbe00QW-y-rA4/view?usp=sharing).
- Download and unzip test split of the data `/data_new_objects_test_split` from [this link](https://drive.google.com/file/d/1e76qJbBJ2bKYq0JzDSRWZjswySX1ftq_/view?usp=sharing).

### Run StructFormer
```bash
cd semantic_rearrangement/demo/
python run_full_pipeline.py \ 
  --dataset_base_dir /path/to/data_new_objects_test_split \
  --pose_generation_model_dir /path/to/models/object_selection_network/best_model \
  --pose_generation_model_dir /path/to/models/structformer_circle/best_model \
  --dirs_config ../configs/data/circle_dirs.yaml
```

### Evaluate Pose Generation Networks
```bash
cd semantic_rearrangement/evaluation/
python test_{model_name}.py \
  --dataset_base_dir /path/to/data_new_objects_test_split \
  --model_dir /path/to/model/best_model \
  --dirs_config ../configs/data/{structure}_dirs.yaml
```

### Evaluate Object Selection Network
```bash
cd semantic_rearrangement/evaluation/
python test_object_selection_network.py \
  --dataset_base_dir /path/to/data_new_objects_test_split \
  --model_dir /path/to/model/best_model \
  --dirs_config ../configs/data/{structure}_dirs.yaml
```

## Training

- Download and unzip test split of the data `/data_new_objects` from ...

### Pose Generation Networks
```bash
cd semantic_rearrangement/training/
python train_{model_name}.py \
  --dataset_base_dir /path/to/data_new_objects \
  --main_config ../configs/{model_name}.yaml \
  --dirs_config ../configs/data/{structure}_dirs.yaml
```

### Object Selection Network
```bash
cd semantic_rearrangement/training/
python train_object_selection_network.py \
  --dataset_base_dir /path/to/data_new_objects \
  --main_config ../configs/object_selection_network.yaml \
  --dirs_config ../configs/data/circle_dirs.yaml
```
