import copy
import cv2
import h5py
import numpy as np
import os
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import random

# from brain2.utils.info import logwarn
# import brain2.utils.image as img
# import brain2.utils.transformations as tra
# import brain2.utils.camera as cam

import semantic_rearrangement.utils.brain2.camera as cam
import semantic_rearrangement.utils.brain2.image as img

from semantic_rearrangement.data.tokenizer import Tokenizer
from semantic_rearrangement.utils.rotation_continuity import compute_geodesic_distance_from_two_matrices
from semantic_rearrangement.utils.rearrangement import show_pcs, get_pts, show_pcs_with_labels
import semantic_rearrangement.utils.transformations as tra


class ObjectSetReferDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 max_num_all_objects,
                 max_num_shape_parameters, max_num_rearrange_features, max_num_anchor_features,
                 num_pts,
                 data_augmentation=True, debug=False):
        """
        :param data_roots:
        :param index_roots:
        :param split: train or test or valid
        :param tokenizer: tokenizer object
        :param max_num_all_objects: the max number of all objects
        :param max_num_shape_parameters: the max number of word tokens for describing the goal structure
        :param max_num_rearrange_features: the max number of word tokens for describing the query objects
        :param max_num_anchor_features: the max number of word tokens for describing the anchor objects
        :param num_pts: the number of points for each object point cloud
        :param data_augmentation: if set to true, add noises to point clouds
        :param debug:
        """

        self.debug = debug

        self.data_roots = data_roots
        print("data dirs:", self.data_roots)

        self.max_num_objects = max_num_all_objects
        self.max_num_rearrange_features = max_num_rearrange_features
        self.max_num_anchor_features = max_num_anchor_features
        self.max_num_shape_parameters = max_num_shape_parameters
        self.num_pts = num_pts

        self.tokenizer = tokenizer

        self.arrangement_data = []
        for data_root, index_root in zip(data_roots, index_roots):
            arrangement_indices_file = os.path.join(data_root, index_root, "{}_arrangement_indices_file_all.txt".format(split))
            if os.path.exists(arrangement_indices_file):
                with open(arrangement_indices_file, "r") as fh:
                    self.arrangement_data.extend([(os.path.join(data_root, f), t) for f, t in eval(fh.readline().strip())])
            else:
                print("{} does not exist".format(arrangement_indices_file))

        # Noise
        self.data_augmentation = data_augmentation
        # additive noise
        self.gp_rescale_factor_range = [12, 20]
        self.gaussian_scale_range = [0., 0.003]
        # multiplicative noise
        self.gamma_shape = 1000.
        self.gamma_scale = 0.001

    def add_noise_to_depth(self, depth_img):
        """ add depth noise """
        multiplicative_noise = np.random.gamma(self.gamma_shape, self.gamma_scale)
        depth_img = multiplicative_noise * depth_img
        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img):
        """ TODO: remove this code or at least celean it up"""
        xyz_img = xyz_img.copy()
        H, W, C = xyz_img.shape
        gp_rescale_factor = np.random.randint(self.gp_rescale_factor_range[0],
                                              self.gp_rescale_factor_range[1])
        gp_scale = np.random.uniform(self.gaussian_scale_range[0],
                                     self.gaussian_scale_range[1])
        small_H, small_W = (np.array([H, W]) / gp_rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=gp_scale, size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]
        return xyz_img

    def _get_rgb(self, h5, idx, ee=True):
        RGB = "ee_rgb" if ee else "rgb"
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        return rgb1

    def _get_depth(self, h5, idx, ee=True):
        DEPTH = "ee_depth" if ee else "depth"

    def _get_images(self, h5, idx, ee=True):
        if ee:
            RGB, DEPTH, SEG = "ee_rgb", "ee_depth", "ee_seg"
            DMIN, DMAX = "ee_depth_min", "ee_depth_max"
        else:
            RGB, DEPTH, SEG = "rgb", "depth", "seg"
            DMIN, DMAX = "depth_min", "depth_max"
        dmin = h5[DMIN][idx]
        dmax = h5[DMAX][idx]
        rgb1 = img.PNGToNumpy(h5[RGB][idx])[:, :, :3] / 255.  # remove alpha
        depth1 = h5[DEPTH][idx] / 20000. * (dmax - dmin) + dmin
        seg1 = img.PNGToNumpy(h5[SEG][idx])

        valid1 = np.logical_and(depth1 > 0.1, depth1 < 2.)

        # proj_matrix = h5['proj_matrix'][()]
        camera = cam.get_camera_from_h5(h5)
        if self.data_augmentation:
            depth1 = self.add_noise_to_depth(depth1)

        xyz1 = cam.compute_xyz(depth1, camera)
        if self.data_augmentation:
            xyz1 = self.add_noise_to_xyz(xyz1, depth1)

        # Transform the point cloud
        # Here it is...
        # CAM_POSE = "ee_cam_pose" if ee else "cam_pose"
        CAM_POSE = "ee_camera_view" if ee else "camera_view"
        cam_pose = h5[CAM_POSE][idx]
        if ee:
            # ee_camera_view has 0s for x, y, z
            cam_pos = h5["ee_cam_pose"][:][:3, 3]
            cam_pose[:3, 3] = cam_pos

        # Get transformed point cloud
        h, w, d = xyz1.shape
        xyz1 = xyz1.reshape(h * w, -1)
        xyz1 = trimesh.transform_points(xyz1, cam_pose)
        xyz1 = xyz1.reshape(h, w, -1)

        scene1 = rgb1, depth1, seg1, valid1, xyz1

        return scene1

    def __len__(self):
        return len(self.arrangement_data)

    def _get_ids(self, h5):
        """
        get object ids

        @param h5:
        @return:
        """
        ids = {}
        for k in h5.keys():
            if k.startswith("id_"):
                ids[k[3:]] = h5[k][()]
        return ids

    def get_data_index(self, idx):
        return self.arrangement_data[idx]

    def prepare_test_data(self, obj_xyzs, obj_rgbs, goal_specification, structure_parameters, gt_num_rearrange_objects):

        # getting object point clouds
        object_pad_mask = []
        rearrange_obj_labels = []
        for i, _ in enumerate(obj_xyzs):
            object_pad_mask.append(0)
            if i < gt_num_rearrange_objects:
                rearrange_obj_labels.append(1.0)
            else:
                rearrange_obj_labels.append(0.0)

        # pad data
        for i in range(self.max_num_objects - len(obj_xyzs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            rearrange_obj_labels.append(-100.0)
            object_pad_mask.append(1)

        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []

        # structure parameters
        # 5 parameters
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(5):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            for _ in range(4):
                sentence_pad_mask.append(0)
            sentence.append(tuple(["PAD"]))
            sentence_pad_mask.append(1)

        # object selection
        is_anchor = len(goal_specification["anchor"]["features"]) > 0
        # rearrange
        for tf in goal_specification["rearrange"]["features"]:
            comparator = tf["comparator"]
            type = tf["type"]
            value = tf["value"]
            if comparator is None:
                # discrete features
                if is_anchor:
                    # leave the desired value to be inferred from anchor
                    sentence.append(("MASK", type))
                else:
                    sentence.append((value, type))
            else:
                # continous features
                sentence.append((comparator, type))
            sentence_pad_mask.append(0)
        # pad, because we always have the fixed length, we don't need to pad this part of the sentence
        assert len(goal_specification["rearrange"]["features"]) == self.max_num_rearrange_features

        # anchor
        for tf in goal_specification["anchor"]["features"]:
            assert tf["comparator"] is None
            type = tf["type"]
            value = tf["value"]
            # discrete features
            sentence.append((value, type))
            sentence_pad_mask.append(0)
        # pad
        for i in range(self.max_num_anchor_features - len(goal_specification["anchor"]["features"])):
            sentence.append(tuple(["PAD"]))
            sentence_pad_mask.append(1)

        ###################################
        if self.debug:
            print(goal_specification)
            print("sentence:", sentence)
            # plt.figure()
            # plt.imshow(rgb)
            # plt.show()
            show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)
        ###################################

        # used to indicate whether the token is an object point cloud or a part of the instruction
        assert self.max_num_rearrange_features + self.max_num_anchor_features + self.max_num_shape_parameters == len(sentence)
        assert self.max_num_objects == len(rearrange_obj_labels)
        token_type_index = [0] * len(sentence) + [1] * self.max_num_objects
        position_index = list(range(len(sentence))) + [i for i in range(self.max_num_objects)]

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "object_pad_mask": object_pad_mask,
            "rearrange_obj_labels": rearrange_obj_labels,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "position_index": position_index,
            "t": 0,
            "filename": "",
            "goal_specification": goal_specification,
        }

        return datum

    def get_raw_data(self, idx):

        filename, t = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        # moved_objs = h5['moved_objs'][()].split(',')
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        # all_object_specs = goal_specification["rearrange"]["objects"] + goal_specification["anchor"]["objects"] + \
        #                    goal_specification["distract"]["objects"]

        ###################################
        # getting scene images and point clouds
        scene = self._get_images(h5, t, ee=True)
        rgb, depth, seg, valid, xyz = scene

        # getting object point clouds
        obj_xyzs = []
        obj_rgbs = []
        object_pad_mask = []
        rearrange_obj_labels = []
        for i, obj in enumerate(all_objs):
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts)
            obj_xyzs.append(obj_xyz)
            obj_rgbs.append(obj_rgb)
            object_pad_mask.append(0)
            if i < num_rearrange_objs:
                rearrange_obj_labels.append(1.0)
            else:
                rearrange_obj_labels.append(0.0)

        # pad data
        for i in range(self.max_num_objects - len(all_objs)):
            obj_xyzs.append(torch.zeros([1024, 3], dtype=torch.float32))
            obj_rgbs.append(torch.zeros([1024, 3], dtype=torch.float32))
            rearrange_obj_labels.append(-100.0)
            object_pad_mask.append(1)

        ###################################
        # preparing sentence
        sentence = []
        sentence_pad_mask = []

        # structure parameters
        # 5 parameters
        structure_parameters = goal_specification["shape"]
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            if structure_parameters["type"] == "circle":
                sentence.append((structure_parameters["radius"], "radius"))
            elif structure_parameters["type"] == "line":
                sentence.append((structure_parameters["length"] / 2.0, "radius"))
            for _ in range(5):
                sentence_pad_mask.append(0)
        else:
            sentence.append((structure_parameters["type"], "shape"))
            sentence.append((structure_parameters["rotation"][2], "rotation"))
            sentence.append((structure_parameters["position"][0], "position_x"))
            sentence.append((structure_parameters["position"][1], "position_y"))
            for _ in range(4):
                sentence_pad_mask.append(0)
            sentence.append(tuple(["PAD"]))
            sentence_pad_mask.append(1)

        # object selection
        is_anchor = len(goal_specification["anchor"]["features"]) > 0
        # rearrange
        for tf in goal_specification["rearrange"]["features"]:
            comparator = tf["comparator"]
            type = tf["type"]
            value = tf["value"]
            if comparator is None:
                # discrete features
                if is_anchor:
                    # leave the desired value to be inferred from anchor
                    sentence.append(("MASK", type))
                else:
                    sentence.append((value, type))
            else:
                # continous features
                sentence.append((comparator, type))
            sentence_pad_mask.append(0)
        # pad, because we always have the fixed length, we don't need to pad this part of the sentence
        assert len(goal_specification["rearrange"]["features"]) == self.max_num_rearrange_features

        # anchor
        for tf in goal_specification["anchor"]["features"]:
            assert tf["comparator"] is None
            type = tf["type"]
            value = tf["value"]
            # discrete features
            sentence.append((value, type))
            sentence_pad_mask.append(0)
        # pad
        for i in range(self.max_num_anchor_features - len(goal_specification["anchor"]["features"])):
            sentence.append(tuple(["PAD"]))
            sentence_pad_mask.append(1)

        ###################################
        if self.debug:
            print("all objects:", all_objs)
            print(goal_specification)
            print("sentence:", sentence)
            print(self.tokenizer.convert_to_natural_sentence(sentence[5:]))
            # plt.figure()
            # plt.imshow(rgb)
            # plt.show()
            show_pcs_with_labels(obj_xyzs, obj_rgbs, rearrange_obj_labels, add_coordinate_frame=True)
        ###################################

        # used to indicate whether the token is an object point cloud or a part of the instruction
        assert self.max_num_rearrange_features + self.max_num_anchor_features + self.max_num_shape_parameters == len(sentence)
        assert self.max_num_objects == len(rearrange_obj_labels)
        token_type_index = [0] * len(sentence) + [1] * self.max_num_objects
        position_index = list(range(len(sentence))) + [i for i in range(self.max_num_objects)]

        # shuffle the position of objects since now the order is rearrange, anchor, distract
        shuffle_object_indices = list(range(len(all_objs)))
        random.shuffle(shuffle_object_indices)
        shuffle_object_indices = shuffle_object_indices + list(range(len(all_objs), self.max_num_objects))
        obj_xyzs = [obj_xyzs[i] for i in shuffle_object_indices]
        obj_rgbs = [obj_rgbs[i] for i in shuffle_object_indices]
        object_pad_mask = [object_pad_mask[i] for i in shuffle_object_indices]
        rearrange_obj_labels = [rearrange_obj_labels[i] for i in shuffle_object_indices]

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "object_pad_mask": object_pad_mask,
            "rearrange_obj_labels": rearrange_obj_labels,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "position_index": position_index,
            "t": t,
            "filename": filename,
            "goal_specification": goal_specification,
        }

        return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer):

        object_pad_mask = torch.LongTensor(datum["object_pad_mask"])
        sentence = torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]])
        sentence_pad_mask = torch.LongTensor(datum["sentence_pad_mask"])
        token_type_index = torch.LongTensor(datum["token_type_index"])
        position_index = torch.LongTensor(datum["position_index"])
        rearrange_obj_labels = torch.FloatTensor(datum["rearrange_obj_labels"])

        tensors = {
            "xyzs": torch.stack(datum["xyzs"], dim=0),
            "rgbs": torch.stack(datum["rgbs"], dim=0),
            "object_pad_mask": object_pad_mask,
            "rearrange_obj_labels": rearrange_obj_labels,
            "sentence": sentence,
            "sentence_pad_mask": sentence_pad_mask,
            "token_type_index": token_type_index,
            "position_index": position_index,
            "t": datum["t"],
            "filename": datum["filename"],
            "goal_specification": datum["goal_specification"],
        }

        return tensors

    def __getitem__(self, idx):

        datum = self.convert_to_tensors(self.get_raw_data(idx), self.tokenizer)

        return datum

    @staticmethod
    def collate_fn(data):
        """
        :param data:
        :return:
        """

        batched_data_dict = {}
        for key in ["xyzs", "rgbs", "rearrange_obj_labels"]:
            batched_data_dict[key] = torch.cat([dict[key] for dict in data], dim=0)
        for key in ["object_pad_mask", "sentence", "sentence_pad_mask", "token_type_index", "position_index"]:
            batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)

        batched_data_dict["goal_specification"] = [dict["goal_specification"] for dict in data]

        return batched_data_dict


if __name__ == "__main__":
    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    dataset = ObjectSetReferDataset(data_roots=["/home/weiyu/data_drive/data_new_objects/examples_circle_new_objects/result"],
                                    index_roots=["index_34k"],
                                    split="train", tokenizer=tokenizer,
                                    max_num_all_objects=11,
                                    max_num_shape_parameters=5,
                                    max_num_rearrange_features=1,
                                    max_num_anchor_features=3,
                                    num_pts=1024,
                                    debug=True)

    for i, d in enumerate(dataset):
        print(i)
        for k in d:
            if torch.is_tensor(d[k]):
                print("--size", k, d[k].shape)
        for k in d:
            print(k, d[k])

        input("next?")

    # dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=1,
    #                         collate_fn=SemanticArrangementDataset.collate_fn)
    # for i, d in enumerate(dataloader):
    #     print(i)
    #     for k in d:
    #         print("--size", k, d[k].shape)
    #     for k in d:
    #         print(k, d[k])
    #
    #     input("next?")

