import torch
import numpy as np
import os
import tqdm
import time
import argparse
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from structformer.data.object_set_refer_dataset import ObjectSetReferDataset
from structformer.training.train_object_selection_network import load_model, validate, infer_once
from structformer.utils.rearrangement import show_pcs_with_predictions, get_initial_scene_idxs, evaluate_target_object_predictions, save_img, show_pcs_with_labels, test_new_vis


class ObjectSelectionInference:

    def __init__(self, model_dir, dirs_cfg, data_split="test"):
        # load prior
        cfg, tokenizer, model, optimizer, scheduler, epoch = load_model(model_dir, dirs_cfg)

        data_cfg = cfg.dataset
        test_dataset = ObjectSetReferDataset(data_cfg.dirs, data_cfg.index_dirs, data_split, tokenizer,
                                             data_cfg.max_num_all_objects,
                                             data_cfg.max_num_shape_parameters,
                                             data_cfg.max_num_rearrange_features,
                                             data_cfg.max_num_anchor_features,
                                             data_cfg.num_pts)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.dataset = test_dataset
        self.epoch = epoch

    def prepare_datum(self, obj_xyzs, obj_rgbs, goal_specification, structure_parameters, gt_num_rearrange_objects):
        datum = self.dataset.prepare_test_data(obj_xyzs, obj_rgbs, goal_specification, structure_parameters, gt_num_rearrange_objects)
        return datum

    def predict_target_objects(self, datum):

        batch = self.dataset.collate_fn([self.dataset.convert_to_tensors(datum, self.tokenizer)])

        gts, predictions = infer_once(self.model, batch, self.cfg.device)
        gts = gts["rearrange_obj_labels"][0].detach().cpu().numpy()
        predictions = predictions["rearrange_obj_labels"][0].detach().cpu().numpy()
        predictions = predictions > 0.5
        # remove paddings
        target_mask = gts != -100
        gts = gts[target_mask]
        predictions = predictions[target_mask]

        return predictions, gts

    def validate(self):
        """
        validate the pretrained model on the dataset

        :return:
        """
        data_cfg = self.cfg.dataset
        data_iter = DataLoader(self.dataset, batch_size=data_cfg.batch_size, shuffle=False,
                               collate_fn=ObjectSetReferDataset.collate_fn,
                               pin_memory=data_cfg.pin_memory, num_workers=data_cfg.num_workers)

        validate(self.model, data_iter, self.epoch, self.cfg.device)


def inference(model_dir, dirs_cfg, visualize=False, inference_visualization_dir=None):

    # object selection information
    # goal = {"rearrange": {"features": [], "objects": [], "combine_features_logic": None, "count": None},
    #         "anchor": {"features": [], "objects": [], "combine_features_logic": None},
    #         "distract": {"features": [], "objects": [], "combine_features_logic": None},
    #         "random_selection": {"varying_features": [], "nonvarying_features": []},
    #         "order": {"feature": None}
    #         }

    if inference_visualization_dir:
        if not os.path.exists(inference_visualization_dir):
            os.makedirs(inference_visualization_dir)

    object_selection_inference = ObjectSelectionInference(model_dir, dirs_cfg)
    test_dataset = object_selection_inference.dataset

    initial_scene_idxs = get_initial_scene_idxs(test_dataset)

    all_predictions = []
    all_gts = []
    all_goal_specifications = []
    all_sentences = []

    count = 0
    for idx in tqdm.tqdm(range(len(test_dataset))):

        if idx not in initial_scene_idxs:
            continue

        count += 1

        datum = test_dataset.get_raw_data(idx)
        goal_specification = datum["goal_specification"]
        object_selection_sentence = datum["sentence"][5:]
        reference_sentence = object_selection_inference.tokenizer.convert_to_natural_sentence(
            object_selection_sentence)

        predictions, gts = object_selection_inference.predict_target_objects(datum)

        if visualize:
            print(gts)
            print(predictions)
            print(object_selection_sentence)
            print(reference_sentence)
            print("Visualize groundtruth (dot color) and prediction (ring color)")
            show_pcs_with_predictions(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)],
                                      gts, predictions, add_coordinate_frame=False)

        if inference_visualization_dir:
            buffer = show_pcs_with_predictions(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)],
                                               gts, predictions, add_coordinate_frame=False, return_buffer=True)
            img = np.uint8(np.asarray(buffer) * 255)
            save_img(img, os.path.join(inference_visualization_dir, "scene_{}.png".format(idx)), text=reference_sentence)

        # try:
        #     batch = test_dataset.collate_fn([test_dataset.convert_to_tensors(datum, tokenizer)])
        # except KeyError:
        #     print("skipping this for now because we are using an outdated model with an old vocab")
        #     continue
        #
        # goal_specification = datum["goal_specification"]
        #
        # gts, predictions = infer_once(model, batch, cfg.device)
        # gts = gts["rearrange_obj_labels"][0].detach().cpu().numpy()
        # predictions = predictions["rearrange_obj_labels"][0].detach().cpu().numpy()
        # predictions = predictions > 0.5
        # # remove paddings
        # target_mask = gts != -100
        # gts = gts[target_mask]
        # predictions = predictions[target_mask]
        #
        # object_selection_sentence = datum["sentence"][5:]
        #
        # if visualize:
        #     print(gts)
        #     print(predictions)
        #     print(object_selection_sentence)
        #     print(tokenizer.convert_to_natural_sentence(object_selection_sentence))
        #
        #     if inference_visualization_dir is None:
        #         show_pcs_with_predictions(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)],
        #                               gts, predictions, add_coordinate_frame=False)
        #         # show_pcs_with_only_predictions(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)],
        #         #                       gts, predictions, add_coordinate_frame=False)
        #         # test_new_vis(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)])
        #     else:
        #         save_filename = os.path.join(inference_visualization_dir, "{}.png".format(idx))
        #         buffer = show_pcs_with_predictions(datum["xyzs"][:len(gts)], datum["rgbs"][:len(gts)],
        #                                   gts, predictions, add_coordinate_frame=False, return_buffer=True)
        #         img = np.uint8(np.asarray(buffer) * 255)
        #         save_img(img, save_filename, text=tokenizer.convert_to_natural_sentence(datum["sentence"]))

        all_predictions.append(predictions)
        all_gts.append(gts)
        all_goal_specifications.append(goal_specification)
        all_sentences.append(object_selection_sentence)

    # create a more detailed report
    evaluate_target_object_predictions(all_gts, all_predictions, all_sentences, initial_scene_idxs,
                                       object_selection_inference.tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple model")
    parser.add_argument("--dataset_base_dir", help='location of the dataset', type=str)
    parser.add_argument("--model_dir", help='location for the saved model', type=str)
    parser.add_argument("--dirs_config", help='config yaml file for directories', default="", type=str)
    parser.add_argument("--inference_visualization_dir", help='location for saving visualizations of inference results',
                        type=str, default=None)
    parser.add_argument("--visualize", default=1, type=int, help='whether to visualize inference results while running')
    args = parser.parse_args()

    os.environ["DATETIME"] = time.strftime("%Y%m%d-%H%M%S")

    # # debug only
    # args.dataset_base_dir = "/home/weiyu/data_drive/data_new_objects_test_split"
    # args.model_dir = "/home/weiyu/Research/intern/StructFormer/models/object_selection_network/best_model"
    # args.dirs_config = "/home/weiyu/Research/intern/StructFormer/structformer/configs/data/line_dirs.yaml"
    # args.visualize = True

    if args.dirs_config:
        assert os.path.exists(args.dirs_config), "Cannot find config yaml file at {}".format(args.dirs_config)
        dirs_cfg = OmegaConf.load(args.dirs_config)
        dirs_cfg.dataset_base_dir = args.dataset_base_dir
        OmegaConf.resolve(dirs_cfg)
    else:
        dirs_cfg = None

    inference(args.model_dir, dirs_cfg, args.visualize, args.inference_visualization_dir)