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
from semantic_rearrangement.utils.rearrangement import show_pcs, get_pts, combine_and_sample_xyzs
import semantic_rearrangement.utils.transformations as tra


class BinaryDataset(torch.utils.data.Dataset):

    def __init__(self, data_roots, index_roots, split, tokenizer,
                 max_num_objects, max_num_other_objects,
                 max_num_shape_parameters, max_num_rearrange_features, max_num_anchor_features,
                 num_pts,
                 data_augmentation=True, debug=False):
        """
        :param data_roots:
        :param index_roots:
        :param split: train or test or valid
        :param tokenizer: tokenizer object
        :param max_num_objects: the max number of "query" objects
        :param max_num_other_objects: the max number of "distractor" objects
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

        self.max_num_objects = max_num_objects
        self.max_num_other_objects = max_num_other_objects
        self.max_num_shape_parameters = max_num_shape_parameters
        self.max_num_rearrange_features = max_num_rearrange_features
        self.max_num_anchor_features = max_num_anchor_features
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

        # remove rearranged scenes
        arrangement_data = []
        for d in self.arrangement_data:
            if d[1] != 0:
                arrangement_data.append(d)
        self.arrangement_data = arrangement_data
        print("{} valid steps".format(len(self.arrangement_data)))

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

    def get_object_position_vocab_sizes(self):
        return self.tokenizer.get_object_position_vocab_sizes()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_data_index(self, idx):
        filename, step_t = self.arrangement_data[idx]
        return filename, step_t

    def get_raw_data(self, idx, inference_mode=False):

        filename, step_t = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        # moved_objs = h5['moved_objs'][()].split(',')
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        num_other_objs = len(goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"])

        assert len(all_objs) == num_rearrange_objs + num_other_objs, "{}, {}".format(len(all_objs), num_rearrange_objs + num_other_objs)
        assert num_rearrange_objs <= self.max_num_objects
        assert num_other_objs <= self.max_num_other_objects

        target_objs = all_objs[:num_rearrange_objs]
        other_objs = all_objs[num_rearrange_objs:]

        structure_parameters = goal_specification["shape"]

        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        all_objs = target_objs + other_objs

        ###################################
        forward_step_t = num_rearrange_objs - step_t

        # getting scene images and point clouds
        scene = self._get_images(h5, step_t, ee=True)
        rgb, depth, seg, valid, xyz = scene

        query_obj = all_objs[forward_step_t]
        anchor_obj = None
        if forward_step_t > 0:
            anchor_obj = all_objs[forward_step_t - 1]

        # getting object point clouds
        query_obj_xyz = None
        query_obj_rgb = None
        anchor_obj_xyz = None
        anchor_obj_rgb = None
        if self.debug:
            other_obj_xyzs = []
        for obj in all_objs:
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts)
            if not ok:
                raise Exception

            if obj == query_obj:
                query_obj_xyz = obj_xyz
                query_obj_rgb = obj_rgb
            elif obj == anchor_obj:
                anchor_obj_xyz = obj_xyz
                anchor_obj_rgb = obj_rgb
            else:
                if self.debug:
                    other_obj_xyzs.append(obj_xyz)

        mask_other = np.logical_and(seg != ids[query_obj], valid)
        if anchor_obj is None:
            anchor_obj_xyz = torch.zeros([self.num_pts, 3], dtype=torch.float32)
            anchor_obj_rgb = torch.zeros([self.num_pts, 3], dtype=torch.float32)
            ok, bg_xyz, bg_rgb, _ = get_pts(xyz, rgb, mask_other,
                                            num_pts=self.num_pts)
        else:
            pt_anchor = torch.mean(anchor_obj_xyz, dim=0)
            ok, bg_xyz, bg_rgb, _ = get_pts(xyz, rgb, mask_other,
                                            num_pts=self.num_pts,
                                            center=pt_anchor)
        if not ok:
            raise Exception

        if self.debug:
            vis_query_obj_rgb = np.tile(np.array([0, 1, 0], dtype=np.float), (self.num_pts, 1))
            vis_anchor_obj_rgb = np.tile(np.array([1, 0, 0], dtype=np.float), (self.num_pts, 1))
            vis_bg_rgb = np.tile(np.array([0, 0, 1], dtype=np.float), (self.num_pts, 1))
            other_obj_rgbs = [np.tile(np.array([0, 0, 1], dtype=np.float), (self.num_pts, 1)) for _ in other_obj_xyzs]
            print("Visualize query object (G), anchor object (R), and other objects")
            show_pcs([query_obj_xyz, anchor_obj_xyz] + other_obj_xyzs, [vis_query_obj_rgb, vis_anchor_obj_rgb] + other_obj_rgbs,
                     add_coordinate_frame=True)
            print("Visualize query object (G), anchor object (R), and background point cloud (B)")
            show_pcs([query_obj_xyz, anchor_obj_xyz, bg_xyz], [vis_query_obj_rgb, vis_anchor_obj_rgb, vis_bg_rgb],
                     add_coordinate_frame=True)

        ###################################
        # code below compute goal pc pose
        current_query_pc_center = torch.mean(query_obj_xyz, dim=0).numpy()[:3]
        current_anchor_pc_center = torch.mean(anchor_obj_xyz, dim=0).numpy()[:3]

        current_query_pc_pose = np.eye(4)
        current_query_pc_pose[:3, 3] = current_query_pc_center

        goal_query_pose = h5[query_obj][0]
        current_query_pose = h5[query_obj][step_t]

        goal_query_pc_pose = goal_query_pose @ np.linalg.inv(current_query_pose) @ current_query_pc_pose

        goal_query_pc_center = goal_query_pc_pose[:3, 3]
        goal_query_pc_translation_offset = goal_query_pc_center - current_anchor_pc_center
        goal_query_pc_rotation = np.eye(4)
        goal_query_pc_rotation[:3, :3] = goal_query_pc_pose[:3, :3]

        if self.debug:
            # verify the computed goal pc pose
            t = np.eye(4)
            t[:3, 3] = current_anchor_pc_center + goal_query_pc_translation_offset - current_query_pc_center
            new_query_obj_xyz = trimesh.transform_points(query_obj_xyz, t)

            # rotating in place
            query_obj_center = np.mean(new_query_obj_xyz, axis=0)
            centered_query_obj_xyz = new_query_obj_xyz - query_obj_center
            new_centered_query_obj_xyz = trimesh.transform_points(centered_query_obj_xyz, goal_query_pc_rotation, translate=True)
            new_query_obj_xyz = new_centered_query_obj_xyz + query_obj_center
            new_query_obj_xyz = torch.tensor(new_query_obj_xyz, dtype=query_obj_xyz.dtype)

            vis_new_query_obj_rgb = np.tile(np.array([1, 0.7, 0], dtype=np.float), (self.num_pts, 1))
            print("Visualize predicted object after move (Y), query object before moving (G), all other objects")
            show_pcs([new_query_obj_xyz, query_obj_xyz, anchor_obj_xyz] + other_obj_xyzs, [vis_new_query_obj_rgb, vis_query_obj_rgb, anchor_obj_rgb] + other_obj_rgbs,
                      add_coordinate_frame=True)

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
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)
        ###################################

        # add 3 positions for query, anchor, bg
        position_index = list(range(self.max_num_shape_parameters + 3))
        pad_mask = sentence_pad_mask + [0, 0, 0]

        obj_x_outputs = [goal_query_pc_translation_offset[0]]
        obj_y_outputs = [goal_query_pc_translation_offset[1]]
        obj_z_outputs = [goal_query_pc_translation_offset[2]]
        obj_theta_outputs = [goal_query_pc_rotation[:3, :3].flatten().tolist()]

        ###################################
        if self.debug:
            print("---")
            print("all objects:", all_objs)
            print("target objects:", target_objs)
            print("other objects:", other_objs)
            print(goal_specification)
            print("sentence:", sentence)
            print("obj_x_outputs", obj_x_outputs)
            print("obj_y_outputs", obj_y_outputs)
            print("obj_z_outputs", obj_z_outputs)
            print("obj_theta_outputs", obj_theta_outputs)
            print("obj_theta_outputs (euler)", tra.euler_from_matrix(np.array(obj_theta_outputs).reshape([3, 3])))
            # plt.figure()
            # plt.imshow(rgb)
            # plt.show()
            #
            # init_scene = self._get_images(h5, 0, ee=True)
            # plt.figure()
            # plt.imshow(init_scene[0])
            # plt.show()

        datum = {
            "query_xyz": query_obj_xyz,
            "query_rgb": query_obj_rgb,
            "anchor_xyz": anchor_obj_xyz,
            "anchor_rgb": anchor_obj_rgb,
            "bg_xyz": bg_xyz,
            "bg_rgb": bg_rgb,
            "sentence": sentence,
            "pad_mask": pad_mask,
            "position_index": position_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "t": step_t,
            "filename": filename}

        return datum

    def get_raw_sequence_data(self, idx):
        """
        This function is used together with self.convert_sequence_to_binary() for inference

        :param idx:
        :return:
        """

        filename, _ = self.arrangement_data[idx]

        h5 = h5py.File(filename, 'r')
        ids = self._get_ids(h5)
        # moved_objs = h5['moved_objs'][()].split(',')
        all_objs = sorted([o for o in ids.keys() if "object_" in o])
        goal_specification = json.loads(str(np.array(h5["goal_specification"])))
        num_rearrange_objs = len(goal_specification["rearrange"]["objects"])
        num_other_objs = len(goal_specification["anchor"]["objects"] + goal_specification["distract"]["objects"])

        assert len(all_objs) == num_rearrange_objs + num_other_objs, "{}, {}".format(len(all_objs),
                                                                                     num_rearrange_objs + num_other_objs)
        assert num_rearrange_objs <= self.max_num_objects
        assert num_other_objs <= self.max_num_other_objects

        target_objs = all_objs[:num_rearrange_objs]
        other_objs = all_objs[num_rearrange_objs:]

        structure_parameters = goal_specification["shape"]

        # Important: ensure the order is correct
        if structure_parameters["type"] == "circle" or structure_parameters["type"] == "line":
            target_objs = target_objs[::-1]
        elif structure_parameters["type"] == "tower" or structure_parameters["type"] == "dinner":
            target_objs = target_objs
        else:
            raise KeyError("{} structure is not recognized".format(structure_parameters["type"]))
        all_objs = target_objs + other_objs

        ###################################
        # getting initial scene images and point clouds
        scene = self._get_images(h5, num_rearrange_objs, ee=True)
        rgb, depth, seg, valid, xyz = scene

        # getting object point clouds
        obj_xyzs = []
        obj_rgbs = []
        other_bg_obj_xyzs = []
        other_bg_obj_rgbs = []
        for obj in target_objs:
            obj_mask = np.logical_and(seg == ids[obj], valid)
            if np.sum(obj_mask) <= 0:
                raise Exception
            ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=self.num_pts)
            if not ok:
                raise Exception
            obj_xyzs.append(obj_xyz)
            obj_rgbs.append(obj_rgb)

        for obj in ids:
            if obj not in target_objs:
                obj_mask = np.logical_and(seg == ids[obj], valid)
                if np.sum(obj_mask) <= 0:
                    continue
                ok, obj_xyz, obj_rgb, _ = get_pts(xyz, rgb, obj_mask, num_pts=None)
                if not ok:
                    raise Exception
                other_bg_obj_xyzs.append(obj_xyz)
                other_bg_obj_rgbs.append(obj_rgb)

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
            sentence.append(("PAD", None))
            sentence_pad_mask.append(1)
        ###################################

        # add 3 positions for query, anchor, bg
        position_index = list(range(self.max_num_shape_parameters + 3))
        pad_mask = sentence_pad_mask + [0, 0, 0]

        obj_x_outputs = [0]
        obj_y_outputs = [0]
        obj_z_outputs = [0]
        obj_theta_outputs = [[0] * 9]

        ###################################
        if self.debug:
            print("---")
            print("all objects:", all_objs)
            print("target objects:", target_objs)
            print("other objects:", other_objs)
            print(goal_specification)
            print("sentence:", sentence)
            print("obj_x_outputs", obj_x_outputs)
            print("obj_y_outputs", obj_y_outputs)
            print("obj_z_outputs", obj_z_outputs)
            print("obj_theta_outputs", obj_theta_outputs)
            print("obj_theta_outputs (euler)", tra.euler_from_matrix(np.array(obj_theta_outputs).reshape([3, 3])))

            # show_pcs(obj_xyzs, obj_rgbs, add_coordinate_frame=True)
            show_pcs(obj_xyzs + other_bg_obj_xyzs, obj_rgbs + other_bg_obj_rgbs, add_coordinate_frame=True)
            # plt.figure()
            # plt.imshow(rgb)
            # plt.show()
            #
            # init_scene = self._get_images(h5, 0, ee=True)
            # plt.figure()
            # plt.imshow(init_scene[0])
            # plt.show()

        datum = {
            "xyzs": obj_xyzs,
            "rgbs": obj_rgbs,
            "other_bg_xyzs": other_bg_obj_xyzs,
            "other_bg_rgbs": other_bg_obj_rgbs,
            "sentence": sentence,
            "pad_mask": pad_mask,
            "position_index": position_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "t": 0,
            "filename": filename}

        return datum

    def convert_sequence_to_binary(self, datum, t):

        obj_xyzs = datum["xyzs"]
        obj_rgbs = datum["rgbs"]
        other_obj_xyzs = datum["other_bg_xyzs"]
        other_obj_rgbs = datum["other_bg_rgbs"]

        query_obj_xyz = obj_xyzs[t]
        query_obj_rgb = obj_rgbs[t]
        bg_xyzs = other_obj_xyzs + [x for i, x in enumerate(obj_xyzs) if i != t]
        bg_rgbs = other_obj_rgbs + [x for i, x in enumerate(obj_rgbs) if i != t]

        if t == 0:
            anchor_obj_xyz = torch.zeros([self.num_pts, 3], dtype=torch.float32)
            anchor_obj_rgb = torch.zeros([self.num_pts, 3], dtype=torch.float32)
            bg_xyz, bg_rgb = combine_and_sample_xyzs(bg_xyzs, bg_rgbs, center=None, num_pts=self.num_pts)

        else:
            anchor_obj_xyz = obj_xyzs[t - 1]
            anchor_obj_rgb = obj_rgbs[t - 1]
            pt_anchor = torch.mean(anchor_obj_xyz, dim=0)
            bg_xyz, bg_rgb = combine_and_sample_xyzs(bg_xyzs, bg_rgbs, center=pt_anchor, radius=0.5,
                                                     num_pts=self.num_pts)

        if self.debug:
            vis_query_obj_rgb = np.tile(np.array([0, 1, 0], dtype=np.float), (self.num_pts, 1))
            vis_anchor_obj_rgb = np.tile(np.array([1, 0, 0], dtype=np.float), (self.num_pts, 1))
            vis_bg_rgb = np.tile(np.array([0, 0, 1], dtype=np.float), (self.num_pts, 1))
            vis_other_obj_rgbs = [np.tile(np.array([0, 0, 1], dtype=np.float), (self.num_pts, 1)) for _ in
                                  other_obj_xyzs]
            show_pcs([query_obj_xyz, anchor_obj_xyz] + other_obj_xyzs,
                     [vis_query_obj_rgb, vis_anchor_obj_rgb] + vis_other_obj_rgbs,
                     add_coordinate_frame=True)
            show_pcs([query_obj_xyz, anchor_obj_xyz, bg_xyz], [vis_query_obj_rgb, vis_anchor_obj_rgb, vis_bg_rgb],
                     add_coordinate_frame=True)

        datum = {
            "query_xyz": query_obj_xyz,
            "query_rgb": query_obj_rgb,
            "anchor_xyz": anchor_obj_xyz,
            "anchor_rgb": anchor_obj_rgb,
            "bg_xyz": bg_xyz,
            "bg_rgb": bg_rgb,
            "sentence": datum["sentence"],
            "pad_mask": datum["pad_mask"],
            "position_index": datum["position_index"],
            "obj_x_outputs": datum["obj_x_outputs"],
            "obj_y_outputs": datum["obj_y_outputs"],
            "obj_z_outputs": datum["obj_z_outputs"],
            "obj_theta_outputs": datum["obj_theta_outputs"],
            "t": t,
            "filename": datum["filename"]}

        return datum

    @staticmethod
    def convert_to_tensors(datum, tokenizer):

        sentence = torch.LongTensor([tokenizer.tokenize(*i) for i in datum["sentence"]])
        pad_mask = torch.LongTensor(datum["pad_mask"])
        position_index = torch.LongTensor(datum["position_index"])
        obj_x_outputs = torch.FloatTensor(datum["obj_x_outputs"])
        obj_y_outputs = torch.FloatTensor(datum["obj_y_outputs"])
        obj_z_outputs = torch.FloatTensor(datum["obj_z_outputs"])
        obj_theta_outputs = torch.FloatTensor(datum["obj_theta_outputs"])

        tensors = {
            "query_xyz": datum["query_xyz"],
            "query_rgb": datum["query_rgb"],
            "anchor_xyz": datum["anchor_xyz"],
            "anchor_rgb": datum["anchor_rgb"],
            "bg_xyz": datum["bg_xyz"],
            "bg_rgb": datum["bg_rgb"],
            "sentence": sentence,
            "pad_mask": pad_mask,
            "position_index": position_index,
            "obj_x_outputs": obj_x_outputs,
            "obj_y_outputs": obj_y_outputs,
            "obj_z_outputs": obj_z_outputs,
            "obj_theta_outputs": obj_theta_outputs,
            "t": datum["t"],
            "filename": datum["filename"]
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
        for key in ["query_xyz", "query_rgb", "anchor_xyz", "anchor_rgb", "bg_xyz", "bg_rgb",
                    "sentence", "pad_mask", "position_index",
                    "obj_x_outputs", "obj_y_outputs", "obj_z_outputs", "obj_theta_outputs"]:
            batched_data_dict[key] = torch.stack([dict[key] for dict in data], dim=0)

        return batched_data_dict


if __name__ == "__main__":
    tokenizer = Tokenizer("/home/weiyu/data_drive/data_new_objects/type_vocabs_coarse.json")
    dataset = BinaryDataset(data_roots=["/home/weiyu/data_drive/data_new_objects/examples_circle_new_objects/result"],
                            index_roots=["index_34k"],
                            split="train", tokenizer=tokenizer,
                            max_num_objects=7,
                            max_num_other_objects=5,
                            max_num_shape_parameters=5,
                            max_num_rearrange_features=0,
                            max_num_anchor_features=0,
                            num_pts=1024,
                            data_augmentation=False, debug=False)

    for i in range(0, 10):
        d = dataset.get_raw_data(i)
        d = dataset.convert_to_tensors(d, dataset.tokenizer)
        for k in d:
            if torch.is_tensor(d[k]):
                print("--size", k, d[k].shape)
        for k in d:
            print(k, d[k])
        input("next?")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8,
                            collate_fn=BinaryDataset.collate_fn)
    for i, d in enumerate(dataloader):
        print(i)
        for k in d:
            print("--size", k, d[k].shape)
        for k in d:
            print(k, d[k])
        input("next?")