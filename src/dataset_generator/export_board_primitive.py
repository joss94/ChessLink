"""Contains the class for the primitive that hanges the HDRI background in the scene"""

import bpy
import os
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

import random
import json
import numpy as np
from mathutils import Matrix

import typing

from orb.primitives.basic_primitive import BasicPrimitive


class ExportBoardPrimitive(BasicPrimitive):
    def __init__(self):
        """Constructor"""
        super().__init__(name="Export board")

    def check(self) -> bool:
        """Checks if the primitive can run safely

        :return: `True`
        :rtype: bool
        """

        return True

    def get_intrinsics_parameters(self, camera):
        """Get the intrinsic parameters of a camera

        :param bpy.types.Object camera: Camera
        :return: Intrinsic matrix
        :rtype: 3x3 matrix
        """
        render = bpy.context.scene.render
        scale = render.resolution_percentage / 100
        resolution_x_in_px = scale * render.resolution_x
        resolution_y_in_px = scale * render.resolution_y
        aspect_ratio = resolution_y_in_px / resolution_x_in_px

        # the camera.data.sensor_height does not seem to be taken into account by
        # blender engine, it is determined by the aspect ratio of the resolution
        # parameters insteads
        focal = camera.data.lens
        s_u = resolution_x_in_px * focal / camera.data.sensor_width
        s_v = resolution_y_in_px * focal / (camera.data.sensor_width * aspect_ratio)

        if camera.data.sensor_fit == "VERTICAL":
            tmp = s_u
            s_u = s_v
            s_v = tmp

        c_u = resolution_x_in_px * scale / 2
        c_v = resolution_y_in_px * scale / 2

        return np.array(
            [
                [s_u, 0, c_u],
                [0, s_v, c_v],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def get_extrinsics_parameters(self, camera):
        """Get the extrinsic parameters of a camera

        :param bpy.types.Object camera: Camera
        :return: Extrinsic matrix
        :rtype: 3x3 matrix
        """

        # Decompose the matrix into R and T
        rotation = np.array(camera.matrix_world.to_euler().to_matrix())
        location = np.array(camera.location)

        # Invert rotation and translations to go from CAM -> WORLD to WORLD -> CAM
        rot_world2bcam = rotation.transpose()
        trans_world2bcam = -1.0 * rot_world2bcam @ location

        # Invert axis to be compatible with OpenCV convention (Y is bottom-down)
        rot_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_world2cv = rot_bcam2cv @ rot_world2bcam
        trans_world2cv = rot_bcam2cv @ trans_world2bcam

        # Merge back R and T into extrinsics matrix
        trans_world2cv = np.expand_dims(trans_world2cv, axis=1)
        return np.concatenate([rot_world2cv, trans_world2cv], axis=1)

    def get_object_bouding_box(self, obj):

        left = None
        bottom = None
        right = None
        top = None

        cam = bpy.data.objects["Camera"]
        extr = self.get_extrinsics_parameters(cam)
        intr = self.get_intrinsics_parameters(cam)

        w2c = intr @ extr

        depsgraph = bpy.context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)

        cos = np.array([v.co for v in object_eval.data.vertices])
        cos = np.append(cos, np.ones((cos.shape[0], 1)), axis=1)

        full_mat = np.array(w2c @ object_eval.matrix_world)
        image_cos = np.transpose(full_mat @ np.transpose(cos))
        image_cos[:, 0] /= image_cos[:, 2]
        image_cos[:, 1] /= image_cos[:, 2]

        render = bpy.context.scene.render
        scale = render.resolution_percentage / 100
        resolution_x_in_px = scale * render.resolution_x
        resolution_y_in_px = scale * render.resolution_y

        left = np.min(image_cos[:, 0]) / resolution_x_in_px
        right = np.max(image_cos[:, 0]) / resolution_x_in_px
        top = np.min(image_cos[:, 1]) / resolution_y_in_px
        bottom = np.max(image_cos[:, 1]) / resolution_y_in_px

        return [left, bottom, right, top]

    # pylint: disable = W0221, W0511
    def execute(
        self, boardCollection: bpy.types.Collection, output_folder: str, filename: str
    ):
        """Extract head of a human"""

        output_folder = bpy.path.abspath(output_folder)

        if not Path(output_folder).exists():
            os.makedirs(output_folder, exist_ok=True)

        annot = {}

        # Place all final pieces by duplicating temporary imports
        annot["pieces"] = []
        for piece_obj in boardCollection.objects:
            info = {}
            info["piece"] = piece_obj["CL_piece"]
            info["bbox"] = self.get_object_bouding_box(piece_obj)
            info["index"] = piece_obj.pass_index
            annot["pieces"].append(info)

        # Get board 2D projection
        annot["board"] = []
        grid_obj = bpy.data.objects["Grid"]
        for vert in grid_obj.data.vertices:
            world_co = grid_obj.matrix_world @ vert.co
            image_co = world_to_camera_view(
                bpy.context.scene, bpy.data.objects["Camera"], world_co
            )
            annot["board"].append(list(image_co.to_2d()))

        # Save annotations
        with open(str(Path(output_folder) / f"{filename}.json"), "w+") as o:
            o.write(str(json.dumps(annot, indent=4)))

    def get_output_names(self):
        return []
