"""Contains the class for the primitive that hanges the HDRI background in the scene"""

import bpy

import random
import numpy as np
from orb.primitives.delete_data_primitive import DeleteDataPrimitive

import typing

from orb.primitives.basic_primitive import BasicPrimitive

class CreateChessBoard(BasicPrimitive):
    def __init__(self):
        """Constructor"""
        super().__init__(name="Update board")

    def check(self) -> bool:
        """Checks if the primitive can run safely

        :return: `True`
        :rtype: bool
        """

        return True

    def create_piece(self, piece_tag, collection):
        object_name = None
        if piece_tag.lower() == 'p':
            object_name = "Pawn"
        elif piece_tag.lower() == 'n':
            object_name = "Knight"
        elif piece_tag.lower() == 'b':
            object_name = "Bishop"
        elif piece_tag.lower() == 'r':
            object_name = "Rook"
        elif piece_tag.lower() == 'q':
            object_name = "Queen"
        elif piece_tag.lower()== 'k':
            object_name = "King"

        piece_obj = None
        if object_name is not None:

            # Add piece
            piece_obj = bpy.data.objects[object_name].copy()
            piece_obj.data = bpy.data.objects[object_name].data.copy()
            piece_obj.hide_render=False
            collection.objects.link(piece_obj)

            # Set piece material
            piece_obj.color = [1, 1, 1, 1] if(piece_tag.isupper()) else [0, 0, 0, 1]
            piece_obj.scale = np.ones(3) * 1.0

        return piece_obj

    def randomize_piece(self, piece):
        for mod in piece.modifiers:
            if mod.type == 'NODES':
                for input in [s for s in mod.node_group.interface.items_tree if s.in_out == 'INPUT']:
                    if input.socket_type in ['NodeSocketFloat', 'NodeSocketInt']:
                        min = input.min_value
                        max = input.max_value
                        if input.socket_type == 'NodeSocketFloat':
                            value = np.random.random() * (max-min) + min
                        else:
                            value = np.random.randint(min, max)
                        mod[input.identifier] = value
        piece.modifiers.update()

        if piece.data and piece.data.shape_keys:
            for kb in piece.data.shape_keys.key_blocks:
                min = kb.slider_min
                max = kb.slider_max
                value = np.random.random() * (max-min) + min
                kb.value = value

        for obj in piece.children_recursive:
            self.randomize_piece(obj)

        piece.update_tag()

    # pylint: disable = W0221, W0511
    def execute(self, chessboardFile: str) -> typing.Tuple[bpy.types.Collection, str]:
        """Extract head of a human
        """
        print("Chessboard: ", chessboardFile)

        #Randomize pieces
        self.randomize_piece(bpy.data.objects["Chessboard"])

        #Rnadomize material
        bpy.data.materials["Pieces"].node_tree.nodes["black"].inputs[0].default_value = np.random.uniform(0.03, 0.1)
        bpy.data.materials["Pieces"].node_tree.nodes["black"].inputs[1].default_value = np.random.uniform(0.5, 1.0)
        bpy.data.materials["Pieces"].node_tree.nodes["black"].inputs[2].default_value = np.random.uniform(0.0, 0.2)

        bpy.data.materials["Pieces"].node_tree.nodes["white"].inputs[0].default_value = np.random.uniform(0.03, 0.1)
        bpy.data.materials["Pieces"].node_tree.nodes["white"].inputs[1].default_value = np.random.uniform(0.0, 1.0)
        bpy.data.materials["Pieces"].node_tree.nodes["white"].inputs[2].default_value = np.random.uniform(0.3, 1.0)

        bpy.data.materials["Pieces"].node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = np.random.rand()

        board_size = 0.45

        # Create a new collection to import objects
        if "ChessSet" in bpy.data.collections:
            DeleteDataPrimitive().execute(bpy.data.collections["ChessSet"])
        collection = bpy.data.collections.new("ChessSet")
        bpy.context.scene.collection.children.link(collection)

        # Import the chessboard
        with bpy.data.libraries.load(chessboardFile) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            collection.objects.link(obj)

        board_obj = data_to.objects[0]
        board_obj.scale *= board_size/8
        board_obj.location.z = 0
        board_obj["CL_piece"] = "board"

        occupied_squares=[]
        for piece in ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]:
            for i in range(random.randint(0, 4)):
                square = random.randint(0, 64)
                while square in occupied_squares:
                    square = random.randint(0, 64)

                occupied_squares.append(square)

                rank = 7 - int(square / 8)
                file = square % 8

                piece_obj = self.create_piece(piece, collection)
                if piece_obj:

                    piece_obj["CL_piece"] = piece
                    piece_obj.parent = None

                    # Move piece
                    piece_obj.rotation_euler.rotate_axis("Z", random.uniform(0.0, 6.32))
                    piece_obj.location = np.array([0.5 + file + random.uniform(-0.2, 0.2), rank + 0.5 + random.uniform(-0.2, 0.2), 0]) * board_size/8

                    # Add piece info
                    bpy.context.view_layer.update()

        bpy.context.view_layer.update()

        return collection, collection.name


    def get_output_names(self):
        return ["Collection", "Collection name"]
