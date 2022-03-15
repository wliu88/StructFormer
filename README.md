# StructFormer

Pytorch implementation for ICRA 2022 paper _StructFormer: Learning Spatial Structure for Language-Guided Semantic Rearrangement of Novel Objects_. [[PDF]](https://arxiv.org/abs/2110.10189) [[Video]](https://youtu.be/6NPdpAtMawM) [[Website]](https://sites.google.com/view/structformer)

StructFormer rearranges unknown objects into semantically meaningful spatial structures based on high-level language instructions and partial-view
point cloud observations of the scene. The model use multi-modal transformers to predict both which objects to manipulate and where to place them.

<p align="center">
<img src="./doc/rearrange_mugs.gif" alt="drawing" width="500"/>
</p>

## License
The source code is released under the [NVIDIA Source Code License](LICENSE). The dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).


## Installation

```
pip install -r requirements.txt
pip install -e .
```

### Notes on Dependencies
- `h5py==2.10`: this specific version is needed.
- `omegaconfg==2.1`: some functions used in this repo are from newer versions

### Environments
The code has been tested on ubuntu 18.04 with nvidia driver 460.91, cuda 11.0, python 3.6, and pytorch 1.7.

## Organization
Source code in the StructFormer package is mainly organized as:
- data loaders `data`
- models `models`
- training scripts `training`
- inference scripts `evaluation`

Parameters for data loaders and models are defined in `OmegaConf` yaml files stored in `configs`.

Trained models are stored in `/experiments`

## Quick Start with Pretrained Models
- Set the package root dir: `export STRUCTFORMER=/path/to/StructFormer`
- Download pretrained models from [this link](https://drive.google.com/file/d/1EsptihJv_lPND902P6CYbe00QW-y-rA4/view?usp=sharing) and unzip to the `$STRUCTFORMER/models` folder
- Download the test split of the dataset from [this link](https://drive.google.com/file/d/1e76qJbBJ2bKYq0JzDSRWZjswySX1ftq_/view?usp=sharing) and unzip to the `$STRUCTFORMER/data_new_objects_test_split`

### Run StructFormer
```bash
cd $STRUCTFORMER/scripts/
python run_full_pipeline.py \
  --dataset_base_dir $STRUCTFORMER/data_new_objects_test_split \
  --object_selection_model_dir $STRUCTFORMER/models/object_selection_network/best_model \
  --pose_generation_model_dir $STRUCTFORMER/models/structformer_circle/best_model \
  --dirs_config $STRUCTFORMER/configs/data/circle_dirs.yaml
```

### Evaluate Pose Generation Networks

Where `{model_name}` is one of `structformer_no_encoder`, `structformer_no_structure`, `object_selection_network`, `structformer`, and `{structure}` is one of `circle`, `line`, `tower`, or `dinner`:

```bash
cd $STRUCTFORMER/src/structformer/evaluation/
python test_{model_name}.py \
  --dataset_base_dir $STRUCTFORMER/data_new_objects_test_split \
  --model_dir $STRUCTFORMER/models/{model_name}_{structure}/best_model \
  --dirs_config $STRUCTFORMER/configs/data/{structure}_dirs.yaml
```

### Evaluate Object Selection Network

Where `{structure}` is as above:

```bash
cd $STRUCTFORMER/src/structformer/evaluation/
python test_object_selection_network.py \
  --dataset_base_dir $STRUCTFORMER/data_new_objects_test_split \
  --model_dir $STRUCTFORMER/models/object_selection_network/best_model \
  --dirs_config $STRUCTFORMER/configs/data/{structure}_dirs.yaml
```

## Training

- Download vocabulary list `type_vocabs_coarse.json` from [this link](https://drive.google.com/file/d/1topawwqMSvwE8Ac-8OiwMApEqwYeR5rc/view?usp=sharing) and unzip to the `$STRUCTFORMER/data_new_objects`.
- Download all data for [circle](https://drive.google.com/file/d/1PTGFcAWBrQmlglygNiJz6p7s0rqe2rtP/view?usp=sharing) and unzip to the `$STRUCTFORMER/data_new_objects`.

### Pose Generation Networks

Where `{model_name}` is one of `structformer_no_encoder`, `structformer_no_structure`, `object_selection_network`, `structformer`, and `{structure}` is one of `circle`, `line`, `tower`, or `dinner`:

```bash
cd $STRUCTFORMER/src/structformer/training/
python train_{model_name}.py \
  --dataset_base_dir $STRUCTFORMER/data_new_objects \
  --main_config $STRUCTFORMER/configs/{model_name}.yaml \
  --dirs_config STRUCTFORMER/configs/data/{structure}_dirs.yaml
```

### Object Selection Network
```bash
cd $STRUCTFORMER/src/structformer/training/
python train_object_selection_network.py \
  --dataset_base_dir $STRUCTFORMER/data_new_objects \
  --main_config $STRUCTFORMER/configs/object_selection_network.yaml \
  --dirs_config $STRUCTFORMER/configs/data/circle_dirs.yaml
```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{structformer2022,
    title     = {StructFormer: Learning Spatial Structure for Language-Guided Semantic Rearrangement of Novel Objects},
    author    = {Liu, Weiyu and Paxton, Chris and Hermans, Tucker and Fox, Dieter},
    year      = {2022},
    booktitle = {ICRA 2022}
}
```