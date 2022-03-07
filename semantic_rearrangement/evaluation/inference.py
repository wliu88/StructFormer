import os
import numpy as np
import torch
import copy

import semantic_rearrangement.utils.transformations as tra
from semantic_rearrangement.utils.rearrangement import show_pcs, save_pcs, move_one_object_pc, make_gifs, modify_language, sample_gaussians, fit_gaussians, show_pcs_color_order


class PointCloudRearrangement:

    """
    helps to keep track of point clouds and predicted object poses for inference

    ToDo: make the whole thing live on pytorch tensor
    ToDo: support binary format
    """

    def __init__(self, initial_datum, pose_format="xyz+3x3", use_structure_frame=True):

        assert pose_format == "xyz+3x3", "{} pose format not supported".format(pose_format)

        self.use_structure_frame = use_structure_frame

        self.num_target_objects = None
        self.num_other_objects = None

        # important: we do not store any padding pcs and poses
        self.initial_xyzs = {"xyzs": [], "rgbs": [], "other_xyzs": [], "other_rgbs": []}
        self.goal_xyzs = {"xyzs": [], "rgbs": []}
        self.goal_poses = {"obj_poses": []}
        if self.use_structure_frame:
            self.goal_poses["struct_pose"] = []

        self.set_initial_pc(initial_datum)

    def set_initial_pc(self, datum):

        self.num_target_objects = np.sum(np.array(datum["object_pad_mask"]) == 0)
        self.num_other_objects = np.sum(np.array(datum["other_object_pad_mask"]) == 0)

        self.initial_xyzs["xyzs"] = datum["xyzs"][:self.num_target_objects]
        self.initial_xyzs["rgbs"] = datum["rgbs"][:self.num_target_objects]
        self.initial_xyzs["other_xyzs"] = datum["other_xyzs"][:self.num_other_objects]
        self.initial_xyzs["other_rgbs"] = datum["other_rgbs"][:self.num_other_objects]

    def set_goal_poses(self, goal_struct_pose, goal_obj_poses, input_pose_format="xyz+3x3",
                       n_obj_idxs=None, skip_update_struct=False):
        """

        :param goal_struct_pose:
        :param goal_obj_poses:
        :param input_pose_format:
        :param n_obj_idxs: only set the goal poses for the indexed objects
        :param skip_update_struct: if set to true, do not update struct pose
        :return:
        """

        # in these cases, we need to ensure the goal poses have already been set so that we can only update some of it
        if n_obj_idxs is not None or skip_update_struct:
            assert len(self.goal_poses["obj_poses"]) != 0
            if self.use_structure_frame:
                assert len(self.goal_poses["struct_pose"]) != 0

        if input_pose_format == "xyz+3x3":
            # check input
            if not skip_update_struct and self.use_structure_frame:
                assert len(goal_struct_pose) == 12
            if n_obj_idxs is None:
                # in case the input contains padding poses
                if len(goal_obj_poses) != self.num_target_objects:
                    goal_obj_poses = goal_obj_poses[:self.num_target_objects]
            else:
                assert len(goal_obj_poses) == len(n_obj_idxs)
            assert all(len(gop) == 12 for gop in goal_obj_poses)

            # convert to standard form
            if not skip_update_struct and self.use_structure_frame:
                if type(goal_struct_pose) != list:
                    goal_struct_pose = goal_struct_pose.tolist()
            if type(goal_obj_poses) != list:
                goal_obj_poses = goal_obj_poses.tolist()
            for i in range(len(goal_obj_poses)):
                if type(goal_obj_poses[i]) != list:
                    goal_obj_poses[i] = goal_obj_poses.tolist()

        elif input_pose_format == "flat:xyz+3x3":
            # check input
            if not skip_update_struct and self.use_structure_frame:
                assert len(goal_struct_pose) == 12
            if n_obj_idxs is None:
                # flat means that object poses are in one list instead of a list of lists
                assert len(goal_obj_poses) == self.num_target_objects * 12
            else:
                assert len(goal_obj_poses) == len(n_obj_idxs) * 12

            # convert to standard form
            if not skip_update_struct and self.use_structure_frame:
                if type(goal_struct_pose) != list:
                    goal_struct_pose = goal_struct_pose.tolist()
            if type(goal_obj_poses) != list:
                goal_obj_poses = goal_obj_poses.tolist()

            goal_obj_poses = np.array(goal_obj_poses).reshape(-1, 12).tolist()

        elif input_pose_format == "flat:xyz+rpy":
            # check input
            if not skip_update_struct and self.use_structure_frame:
                assert len(goal_struct_pose) == 6
            if n_obj_idxs is None:
                assert len(goal_obj_poses) == self.num_target_objects * 6
            else:
                assert len(goal_obj_poses) == len(n_obj_idxs) * 6

            # convert to standard form
            if not skip_update_struct and self.use_structure_frame:
                if type(goal_struct_pose) != list:
                    goal_struct_pose = goal_struct_pose.tolist()
            if type(goal_obj_poses) != list:
                goal_obj_poses = np.array(goal_obj_poses).reshape(-1, 6).tolist()

            if not skip_update_struct and self.use_structure_frame:
                goal_struct_pose = goal_struct_pose[:3] + tra.euler_matrix(goal_struct_pose[3], goal_struct_pose[4], goal_struct_pose[5])[:3, :3].flatten().tolist()
            converted_goal_obj_poses = []
            for gop in goal_obj_poses:
                converted_goal_obj_poses.append(
                    gop[:3] + tra.euler_matrix(gop[3], gop[4], gop[5])[:3, :3].flatten().tolist())
            goal_obj_poses = converted_goal_obj_poses

        else:
            raise KeyError

        # update
        if not skip_update_struct and self.use_structure_frame:
            self.goal_poses["struct_pose"] = goal_struct_pose
        if n_obj_idxs is None:
            self.goal_poses["obj_poses"] = goal_obj_poses
        else:
            for count, oi in enumerate(n_obj_idxs):
                self.goal_poses["obj_poses"][oi] = goal_obj_poses[count]

    def get_goal_poses(self, output_pose_format="xyz+3x3",
                       n_obj_idxs=None, skip_update_struct=False, combine_struct_objs=False):
        """

        :param output_pose_format:
        :param n_obj_idxs: only retrieve the goal poses for the indexed objects
        :param skip_update_struct: if set to true, do not retrieve struct pose
        :param combine_struct_objs: one output, return a list of lists, where the first list if for the structure pose
                                    and remainings are for object poses
        :return:
        """
        if output_pose_format == "xyz+3x3":
            if self.use_structure_frame:
                goal_struct_pose = self.goal_poses["struct_pose"]
            goal_obj_poses = self.goal_poses["obj_poses"]

            if n_obj_idxs is not None:
                goal_obj_poses = [goal_obj_poses[i] for i in n_obj_idxs]

        elif output_pose_format == "flat:xyz+3x3":
            if self.use_structure_frame:
                goal_struct_pose = self.goal_poses["struct_pose"]

            if n_obj_idxs is None:
                goal_obj_poses = np.array(self.goal_poses["obj_poses"]).flatten().tolist()
            else:
                goal_obj_poses = np.array([self.goal_poses["obj_poses"][i] for i in n_obj_idxs]).flatten().tolist()

        elif output_pose_format == "flat:xyz+rpy":
            if self.use_structure_frame:
                ax, ay, az = tra.euler_from_matrix(np.asarray(self.goal_poses["struct_pose"][3:]).reshape(3, 3))
                goal_struct_pose = self.goal_poses["struct_pose"][:3] + [ax, ay, az]

            goal_obj_poses = []
            for gop in self.goal_poses["obj_poses"]:
                ax, ay, az = tra.euler_from_matrix(np.asarray(gop[3:]).reshape(3, 3))
                goal_obj_poses.append(gop[:3] + [ax, ay, az])

            if n_obj_idxs is None:
                goal_obj_poses = np.array(goal_obj_poses).flatten().tolist()
            else:
                goal_obj_poses = np.array([goal_obj_poses[i] for i in n_obj_idxs]).flatten().tolist()

        else:
            raise KeyError

        if not skip_update_struct and self.use_structure_frame:
            if not combine_struct_objs:
                return goal_struct_pose, goal_obj_poses
            else:
                return [goal_struct_pose] + goal_obj_poses
        else:
            return None, goal_obj_poses

    def rearrange(self, n_obj_idxs=None):
        """
        use stored object point clouds of the initial scene and goal poses to
        compute object point clouds of the goal scene.

        :param n_obj_idxs: only update the goal point clouds of indexed objects
        :return:
        """

        # initial scene and goal poses have to be set first
        assert all(len(self.initial_xyzs[k]) != 0 for k in ["xyzs", "rgbs"])
        assert all(len(self.goal_poses[k]) != 0 for k in self.goal_poses)

        # whether we are initializing or updating
        no_goal_xyzs_yet = True
        if len(self.goal_xyzs["xyzs"]):
            no_goal_xyzs_yet = False

        if n_obj_idxs is not None:
            assert no_goal_xyzs_yet is False

        if n_obj_idxs is not None:
            update_obj_idxs = n_obj_idxs
        else:
            update_obj_idxs = list(range(self.num_target_objects))

        if self.use_structure_frame:
            goal_struct_pose = self.goal_poses["struct_pose"]
        else:
            goal_struct_pose = None
        for obj_idx in update_obj_idxs:
            imagined_obj_xyz, imagined_obj_rgb = move_one_object_pc(self.initial_xyzs["xyzs"][obj_idx],
                                                                    self.initial_xyzs["rgbs"][obj_idx],
                                                                    self.goal_poses["obj_poses"][obj_idx],
                                                                    goal_struct_pose)

            if no_goal_xyzs_yet:
                self.goal_xyzs["xyzs"].append(imagined_obj_xyz)
                self.goal_xyzs["rgbs"].append(imagined_obj_rgb)
            else:
                self.goal_xyzs["xyzs"][obj_idx] = imagined_obj_xyz
                self.goal_xyzs["rgbs"][obj_idx] = imagined_obj_rgb

    def visualize(self, time_step, add_other_objects=False,
                  add_coordinate_frame=False, side_view=False, add_table=False,
                  show_vis=True, save_vis=False, save_filename=None, order_color=False):

        if time_step == "initial":
            xyzs = self.initial_xyzs["xyzs"]
            rgbs = self.initial_xyzs["rgbs"]
        elif time_step == "goal":
            xyzs = self.goal_xyzs["xyzs"]
            rgbs = self.goal_xyzs["rgbs"]
        else:
            raise KeyError()

        if add_other_objects:
            xyzs += self.initial_xyzs["other_xyzs"]
            rgbs += self.initial_xyzs["other_rgbs"]

        if show_vis:
            if not order_color:
                show_pcs(xyzs, rgbs, add_coordinate_frame=add_coordinate_frame, side_view=side_view, add_table=add_table)
            else:
                show_pcs_color_order(xyzs, rgbs, add_coordinate_frame=add_coordinate_frame, side_view=side_view, add_table=add_table)

        if save_vis and save_filename is not None:
            save_pcs(xyzs, rgbs, save_path=save_filename, add_coordinate_frame=add_coordinate_frame, side_view=side_view, add_table=add_table)