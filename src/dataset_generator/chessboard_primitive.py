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
                    if input.socket_type == 'NodeSocketFloat':
                        mod[input.identifier] = np.random.uniform(input.min_value, input.max_value)
                    elif input.socket_type == 'NodeSocketInt':
                        mod[input.identifier] = np.random.randint(input.min_value, input.max_value)
                    elif input.socket_type == 'NodeSocketBool':
                         mod[input.identifier] = np.random.random() > 0.5
        piece.modifiers.update()

        if piece.data and piece.data.shape_keys:
            for kb in piece.data.shape_keys.key_blocks:
                kb.value = np.random.uniform(kb.slider_min, kb.slider_max)

        if piece.data:
            for mat in piece.data.materials:
                for n in mat.node_tree.nodes:
                    if n.type == 'GROUP':
                        for input in [s for s in n.node_tree.interface.items_tree if s.in_out == 'INPUT']:
                            if input.socket_type == 'NodeSocketFloat':
                                n.inputs[input.identifier].default_value = np.random.uniform(input.min_value, input.max_value)
                            elif input.socket_type == 'NodeSocketInt':
                                n.inputs[input.identifier].default_value = np.random.randint(input.min_value, input.max_value)
                            elif input.socket_type == 'NodeSocketBool':
                                n.inputs[input.identifier].default_value = np.random.random() > 0.5

        for obj in piece.children_recursive:
            self.randomize_piece(obj)

        piece.update_tag()

    # pylint: disable = W0221, W0511
    def execute(self) -> typing.Tuple[bpy.types.Collection, str]:
        """Extract head of a human
        """
        #Randomize pieces
        self.randomize_piece(bpy.data.objects["Chessboard"])

        board_size = 0.45

        # Create a new collection to import objects
        if "ChessSet" in bpy.data.collections:
            DeleteDataPrimitive().execute(bpy.data.collections["ChessSet"])
        collection = bpy.data.collections.new("ChessSet")
        bpy.context.scene.collection.children.link(collection)

        occupied_squares=[]
        index = 2
        for piece in ["p", "n", "b", "r", "q", "k", "P", "N", "B", "R", "Q", "K"]:

            min_qty = 1
            max_qty = 3

            if piece.lower() in ["q", "k"]:
                min_qty = 2

            for i in range(random.randint(min_qty, max_qty)):
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
                    piece_obj.pass_index = index
                    index +=1

                    # Move piece
                    piece_obj.rotation_euler.rotate_axis("Z", random.uniform(0.0, 6.32))
                    piece_obj.location = np.array([0.5 + file + random.uniform(-0.2, 0.2), rank + 0.5 + random.uniform(-0.2, 0.2), 0]) * board_size/8

                    # Add piece info
                    bpy.context.view_layer.update()

        bpy.context.view_layer.update()

        return collection, collection.name


    def get_output_names(self):
        return ["Collection", "Collection name"]
