import copy
import time
import torch
import numpy as np
import os
import tqdm
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from semantic_rearrangement.data.tokenizer import Tokenizer
from semantic_rearrangement.evaluation.test_object_selection_network import ObjectSelectionInference
from semantic_rearrangement.evaluation.test_structformer import PriorInference



import semantic_rearrangement.data.sequence_dataset as generation_dataset
import semantic_rearrangement.training.train_structformer as generation_model

from semantic_rearrangement.data.object_set_refer_dataset import ObjectSetReferDataset
from semantic_rearrangement.training.train_object_selection_network import load_model, validate, infer_once
from semantic_rearrangement.utils.rearrangement import show_pcs_with_predictions, get_initial_scene_idxs, evaluate_target_object_predictions, save_img, show_pcs_with_labels, test_new_vis


from semantic_rearrangement.data.tokenizer import Tokenizer
import semantic_rearrangement.data.sequence_dataset as prior_dataset
import semantic_rearrangement.training.train_structformer as prior_model
from semantic_rearrangement.utils.rearrangement import show_pcs
from semantic_rearrangement.evaluation.inference import PointCloudRearrangement


def run_demo(object_selection_model_dir, pose_generation_model_dir, dirs_config, beam_size=3):

    object_selection_inference = ObjectSelectionInference(object_selection_model_dir, dirs_cfg)
    pose_generation_inference = PriorInference(pose_generation_model_dir, dirs_cfg)

    test_dataset = object_selection_inference.dataset
    initial_scene_idxs = get_initial_scene_idxs(test_dataset)

    for idx in range(len(test_dataset)):
        if idx not in initial_scene_idxs:
            continue

        if idx == 4:
            continue

        filename, _ = test_dataset.get_data_index(idx)
        scene_id = os.path.split(filename)[1][4:-3]
        print("-"*50)
        print("Scene No.{}".format(scene_id))

        # retrieve data
        init_datum = test_dataset.get_raw_data(idx)
        goal_specification = init_datum["goal_specification"]
        object_selection_structured_sentence = init_datum["sentence"][5:]
        structure_specification_structured_sentence = init_datum["sentence"][:5]
        object_selection_natural_sentence = object_selection_inference.tokenizer.convert_to_natural_sentence(
            object_selection_structured_sentence)
        structure_specification_natural_sentence = object_selection_inference.tokenizer.convert_structure_params_to_natural_language(structure_specification_structured_sentence)

        # object selection
        predictions, gts = object_selection_inference.predict_target_objects(init_datum)

        all_obj_xyzs = init_datum["xyzs"][:len(predictions)]
        all_obj_rgbs = init_datum["rgbs"][:len(predictions)]
        obj_idxs = [i for i, l in enumerate(predictions) if l == 1.0]
        other_obj_idxs = [i for i, l in enumerate(predictions) if l == 0.0]
        obj_xyzs = [all_obj_xyzs[i] for i in obj_idxs]
        obj_rgbs = [all_obj_rgbs[i] for i in obj_idxs]
        other_obj_xyzs = [all_obj_xyzs[i] for i in other_obj_idxs]
        other_obj_rgbs = [all_obj_rgbs[i] for i in other_obj_idxs]

        print("\nSelect objects to rearrange...")
        print("Instruction:", object_selection_natural_sentence)
        print("Visualize groundtruth (dot color) and prediction (ring color)")
        show_pcs_with_predictions(init_datum["xyzs"][:len(predictions)], init_datum["rgbs"][:len(predictions)],
                                  gts, predictions, add_table=True, side_view=True)
        print("Visualize object to rearrange")
        show_pcs(obj_xyzs, obj_rgbs, side_view=True, add_table=True)

        # pose generation
        max_num_objects = pose_generation_inference.cfg.dataset.max_num_objects
        max_num_other_objects = pose_generation_inference.cfg.dataset.max_num_other_objects
        if len(obj_xyzs) > max_num_objects:
            print("WARNING: reducing the number of \"query\" objects because this model is trained with a maximum of {} \"query\" objects. Train a new model if a larger number is needed.".format(max_num_objects))
            obj_xyzs = obj_xyzs[:max_num_objects]
            obj_rgbs = obj_rgbs[:max_num_objects]
        if len(other_obj_xyzs) > max_num_other_objects:
            print("WARNING: reducing the number of \"distractor\" objects because this model is trained with a maximum of {} \"distractor\" objects. Train a new model if a larger number is needed.".format(max_num_other_objects))
            other_obj_xyzs = other_obj_xyzs[:max_num_other_objects]
            other_obj_rgbs = other_obj_rgbs[:max_num_other_objects]

        pose_generation_datum = pose_generation_inference.dataset.prepare_test_data(obj_xyzs, obj_rgbs,
                                                                                    other_obj_xyzs, other_obj_rgbs,
                                                                                    goal_specification["shape"])
        beam_data = []
        beam_pc_rearrangements = []
        for b in range(beam_size):
            datum_copy = copy.deepcopy(pose_generation_datum)
            beam_data.append(datum_copy)
            beam_pc_rearrangements.append(PointCloudRearrangement(datum_copy))

        # autoregressive decoding
        num_target_objects = beam_pc_rearrangements[0].num_target_objects

        # first predict structure pose
        beam_goal_struct_pose, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
        for b in range(beam_size):
            datum = beam_data[b]
            datum["struct_x_inputs"] = [beam_goal_struct_pose[b][0]]
            datum["struct_y_inputs"] = [beam_goal_struct_pose[b][1]]
            datum["struct_z_inputs"] = [beam_goal_struct_pose[b][2]]
            datum["struct_theta_inputs"] = [beam_goal_struct_pose[b][3:]]

        # then iteratively predict pose of each object
        beam_goal_obj_poses = []
        for obj_idx in range(num_target_objects):
            struct_preds, target_object_preds = pose_generation_inference.limited_batch_inference(beam_data)
            beam_goal_obj_poses.append(target_object_preds[:, obj_idx])
            for b in range(beam_size):
                datum = beam_data[b]
                datum["obj_x_inputs"][obj_idx] = target_object_preds[b][obj_idx][0]
                datum["obj_y_inputs"][obj_idx] = target_object_preds[b][obj_idx][1]
                datum["obj_z_inputs"][obj_idx] = target_object_preds[b][obj_idx][2]
                datum["obj_theta_inputs"][obj_idx] = target_object_preds[b][obj_idx][3:]
        # concat in the object dim
        beam_goal_obj_poses = np.stack(beam_goal_obj_poses, axis=0)
        # swap axis
        beam_goal_obj_poses = np.swapaxes(beam_goal_obj_poses, 1, 0)  # batch size, number of target objects, pose dim

        # move pc
        for bi in range(beam_size):
            beam_pc_rearrangements[bi].set_goal_poses(beam_goal_struct_pose[bi], beam_goal_obj_poses[bi])
            beam_pc_rearrangements[bi].rearrange()

        print("\nRearrange \"query\" objects...")
        print("Instruction:", structure_specification_natural_sentence)
        for pi, pc_rearrangement in enumerate(beam_pc_rearrangements):
            print("Visualize rearranged scene sample {}".format(pi))
            pc_rearrangement.visualize("goal", add_other_objects=True, add_table=True, side_view=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--object_selection_model_dir", help='location for the saved object selection model', type=str)
    parser.add_argument("--pose_generation_model_dir", help='location for the saved pose generation model', type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories', type=str)
    args = parser.parse_args()

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.object_selection_model_dir = "/home/weiyu/Research/intern/StructFormer/models/object_selection_network/best_model"
    # args.pose_generation_model_dir = "/home/weiyu/Research/intern/StructFormer/models/structformer_circle/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/semantic_rearrangement/configs/data/circle_dirs.yaml"

    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None

    run_demo(args.object_selection_model_dir, args.pose_generation_model_dir, dirs_cfg)