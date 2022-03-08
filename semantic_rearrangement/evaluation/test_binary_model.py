import torch
import numpy as np
import os
import copy
import tqdm
import argparse
from omegaconf import OmegaConf
import trimesh
import time
from torch.utils.data import DataLoader

from semantic_rearrangement.data.tokenizer import Tokenizer
import semantic_rearrangement.data.binary_dataset as prior_dataset
import semantic_rearrangement.training.train_binary_model as prior_model
from semantic_rearrangement.utils.rearrangement import move_one_object_pc, make_gifs, \
    modify_language, sample_gaussians, fit_gaussians, get_initial_scene_idxs, show_pcs, save_pcs


def test_model(model_dir, dirs_cfg):
    prior_inference = PriorInference(model_dir, dirs_cfg, data_split="test")
    prior_inference.validate()


class PriorInference:

    def __init__(self, model_dir, dirs_cfg, data_split="test"):
        # load prior
        cfg, tokenizer, model, optimizer, scheduler, epoch = prior_model.load_model(model_dir, dirs_cfg)

        data_cfg = cfg.dataset

        dataset = prior_dataset.BinaryDataset(data_cfg.dirs, data_cfg.index_dirs, data_split, tokenizer,
                                              data_cfg.max_num_objects,
                                              data_cfg.max_num_other_objects,
                                              data_cfg.max_num_shape_parameters,
                                              data_cfg.max_num_rearrange_features,
                                              data_cfg.max_num_anchor_features,
                                              data_cfg.num_pts)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.epoch = epoch

    def validate(self):
        data_cfg = self.cfg.dataset
        data_iter = DataLoader(self.dataset, batch_size=data_cfg.batch_size, shuffle=False,
                               collate_fn=prior_dataset.BinaryDataset.collate_fn,
                               pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

        prior_model.validate(self.cfg, self.model, data_iter, self.epoch, self.cfg.device)

    def limited_batch_inference(self, data, t, verbose=False):
        """
        This function makes the assumption that scenes in the batch have the same number of objects that need to be
        rearranged

        :param data:
        :param model:
        :param test_dataset:
        :param tokenizer:
        :param cfg:
        :param num_samples:
        :param verbose:
        :return:
        """

        data_size = len(data)
        batch_size = self.cfg.dataset.batch_size
        if verbose:
            print("data size:", data_size)
            print("batch size:", batch_size)

        num_batches = int(data_size / batch_size)
        if data_size % batch_size != 0:
            num_batches += 1

        all_obj_preds = []
        all_binary_data = []
        for b in range(num_batches):
            if b + 1 == num_batches:
                # last batch
                batch = data[b * batch_size:]
            else:
                batch = data[b * batch_size: (b+1) * batch_size]

            binary_data = [self.dataset.convert_sequence_to_binary(d, t) for d in batch]
            data_tensors = [self.dataset.convert_to_tensors(d, self.tokenizer) for d in binary_data]
            data_tensors = self.dataset.collate_fn(data_tensors)
            predictions = prior_model.infer_once(self.cfg, self.model, data_tensors, self.cfg.device)

            obj_x_preds = torch.cat(predictions["obj_x_outputs"], dim=0)
            obj_y_preds = torch.cat(predictions["obj_y_outputs"], dim=0)
            obj_z_preds = torch.cat(predictions["obj_z_outputs"], dim=0)
            obj_theta_preds = torch.cat(predictions["obj_theta_outputs"], dim=0)
            obj_preds = torch.cat([obj_x_preds, obj_y_preds, obj_z_preds, obj_theta_preds], dim=1)  # batch_size * max num objects, output_dim

            all_obj_preds.append(obj_preds)
            all_binary_data.extend(binary_data)

        obj_preds = torch.cat(all_obj_preds, dim=0)  # data_size * max num objects, output_dim
        obj_preds = obj_preds.detach().cpu().numpy()
        obj_preds = obj_preds.reshape(data_size, obj_preds.shape[-1])  # batch_size, max num objects, output_dim

        return obj_preds, all_binary_data


def inference_beam_decoding(model_dir, dirs_cfg, beam_size=100, max_scene_decodes=30000,
                            visualize=True, visualize_action_sequence=False,
                            inference_visualization_dir=None):
    """
    This function decodes a scene with a single forward pass

    :param model_dir:
    :param discriminator_model_dir:
    :param inference_visualization_dir:
    :param visualize:
    :param num_samples: number of MDN samples drawn, in this case it's also the number of rearrangements
    :param keep_steps:
    :param initial_scenes_only:
    :param verbose:
    :return:
    """

    if inference_visualization_dir and not os.path.exists(inference_visualization_dir):
        os.makedirs(inference_visualization_dir)

    prior_inference = PriorInference(model_dir, dirs_cfg)
    test_dataset = prior_inference.dataset

    initial_scene_idxs = get_initial_scene_idxs(test_dataset)

    decoded_scene_count = 0
    with tqdm.tqdm(total=len(initial_scene_idxs)) as pbar:
        # for idx in np.random.choice(range(len(test_dataset)), len(test_dataset), replace=False):
        for idx in initial_scene_idxs:

            if decoded_scene_count == max_scene_decodes:
                break

            filename, step_t = test_dataset.get_data_index(idx)
            scene_id = os.path.split(filename)[1][4:-3]

            decoded_scene_count += 1

            ############################################
            # retrieve data
            beam_data = []
            num_target_objects = None
            for b in range(beam_size):
                datum = test_dataset.get_raw_sequence_data(idx)
                beam_data.append(datum)

                if num_target_objects is None:
                    num_target_objects = len(datum["xyzs"])

                # We can play with different language here
                # datum["sentence"] = modify_language(datum["sentence"], radius=0.5)

            if visualize:
                datum = beam_data[0]
                print("#"*50)
                print("sentence", datum["sentence"])
                print("num target objects", num_target_objects)
                show_pcs(datum["xyzs"] + datum["other_bg_xyzs"],
                         datum["rgbs"] + datum["other_bg_rgbs"],
                         add_coordinate_frame=False, side_view=True, add_table=True)

            ############################################

            beam_predicted_parameters = [[]] * beam_size
            for time_index in range(num_target_objects):

                # iteratively decoding
                target_object_preds, binary_data = prior_inference.limited_batch_inference(beam_data, time_index)

                for b in range(beam_size):
                    # a list of list, where each inside list contains xyz, 3x3 rotation

                    datum = beam_data[b]
                    binary_datum = binary_data[b]
                    obj_pred = target_object_preds[b]

                    #------------
                    goal_query_pc_translation_offset = obj_pred[:3]
                    goal_query_pc_rotation = np.eye(4)
                    goal_query_pc_rotation[:3, :3] = np.array(obj_pred[3:]).reshape(3, 3)

                    query_obj_xyz = binary_datum["query_xyz"]
                    anchor_obj_xyz = binary_datum["anchor_xyz"]


                    current_query_pc_center = torch.mean(query_obj_xyz, dim=0).numpy()[:3]
                    current_anchor_pc_center = torch.mean(anchor_obj_xyz, dim=0).numpy()[:3]

                    t = np.eye(4)
                    t[:3, 3] = current_anchor_pc_center + goal_query_pc_translation_offset - current_query_pc_center
                    new_query_obj_xyz = trimesh.transform_points(query_obj_xyz, t)

                    # rotating in place
                    # R = tra.euler_matrix(0, 0, obj_pc_rotations[i])
                    query_obj_center = np.mean(new_query_obj_xyz, axis=0)
                    centered_query_obj_xyz = new_query_obj_xyz - query_obj_center
                    new_centered_query_obj_xyz = trimesh.transform_points(centered_query_obj_xyz, goal_query_pc_rotation,
                                                                          translate=True)
                    new_query_obj_xyz = new_centered_query_obj_xyz + query_obj_center
                    new_query_obj_xyz = torch.tensor(new_query_obj_xyz, dtype=query_obj_xyz.dtype)

                    # vis_query_obj_rgb = np.tile(np.array([0, 1, 0], dtype=np.float), (query_obj_xyz.shape[0], 1))
                    # vis_anchor_obj_rgb = np.tile(np.array([1, 0, 0], dtype=np.float), (anchor_obj_xyz.shape[0], 1))
                    # vis_new_query_obj_rgb = np.tile(np.array([0, 0, 1], dtype=np.float), (new_query_obj_xyz.shape[0], 1))
                    # show_pcs([new_query_obj_xyz, query_obj_xyz, anchor_obj_xyz],
                    #          [vis_new_query_obj_rgb, vis_query_obj_rgb, vis_anchor_obj_rgb],
                    #          add_coordinate_frame=True)

                    datum["xyzs"][time_index] = new_query_obj_xyz

                    current_object_param = t[:3, 3].tolist() + goal_query_pc_rotation[:3, :3].flatten().tolist()
                    beam_predicted_parameters[b].append(current_object_param)

            for b in range(beam_size):
                datum = beam_data[b]

                pc_sizes = [xyz.shape[0] for xyz in datum["other_bg_xyzs"]]
                table_idx = np.argmax(pc_sizes)
                show_pcs(datum["xyzs"] + [xyz for i, xyz in enumerate(datum["other_bg_xyzs"]) if i != table_idx],
                         datum["rgbs"] + [rgb for i, rgb in enumerate(datum["other_bg_rgbs"]) if i != table_idx],
                         add_coordinate_frame=False, side_view=True, add_table=True)

            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--model_dir", help='location for the saved model', type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories', default="", type=str)
    args = parser.parse_args()

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.model_dir = "/home/weiyu/Research/intern/StructFormer/models/binary_model_tower/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/semantic_rearrangement/configs/data/tower_dirs.yaml"

    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None

    inference_beam_decoding(args.model_dir, dirs_cfg, beam_size=3, max_scene_decodes=30000,
                            visualize=True, visualize_action_sequence=False,
                            inference_visualization_dir=None)

