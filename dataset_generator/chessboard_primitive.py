"""Contains the class for the primitive that hanges the HDRI background in the scene"""

import bpy
import os
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

import chess.pgn
import random
import json
import numpy as np
from mathutils import Matrix

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

    def get_object_bouding_box(self, obj):

        left = None
        bottom = None
        right = None
        top = None

        for vec_co in [v.co for v in obj.data.vertices]:
            world_co = obj.matrix_world @ vec_co
            image_co = world_to_camera_view(bpy.context.scene, bpy.data.objects['Camera'], world_co)

            if left is None or image_co.x < left:
                left = image_co.x
            if bottom is None or image_co.y < bottom:
                bottom = image_co.y
            if right is None or image_co.x > right:
                right = image_co.x
            if top is None or image_co.y > top:
                top = image_co.y

        return [left, bottom, right, top]

    def create_piece(self, piece_tag, collection, white_mat, black_mat):
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
            collection.objects.link(piece_obj)

            # Set piece material
            mat = white_mat if(piece_tag.isupper()) else black_mat
            if piece_obj.data.materials:
                piece_obj.data.materials.clear()
            piece_obj.data.materials.append(mat)

        return piece_obj

    # pylint: disable = W0221, W0511
    def execute(self, chessboardFile: str, pgnFile: str, piecesFile: str, outputPath: str) -> typing.Tuple[str, bpy.types.Collection]:
        """Extract head of a human
        """

        print("Chessboard: ", chessboardFile)
        print("Pieces: ", piecesFile)

        dir = Path(outputPath).parent
        if not dir.exists():
            os.makedirs(dir, exist_ok=True)

        annot = {}

        # Read chess game and go to a random position
        with open(pgnFile) as f:
            game = chess.pgn.read_game(f)

        with open(pgnFile) as f:
            n_moves = f.read().count('.')

        for i in range (0, random.randint(0, n_moves)):
            game = game.next()

        board = game.board()
        print(board)
        annot['position'] = str(board).replace('\n', '').replace(" ", "")
        print(annot['position'])

        # Create a new collection to import objects
        collection = bpy.data.collections.new("ChessSet")
        bpy.context.scene.collection.children.link(collection)

        # Import the chessboard
        with bpy.data.libraries.load(chessboardFile) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            collection.objects.link(obj)

        #Import the necessary pieces only once:
        tmp_collection = bpy.data.collections.new("Pieces_tmp")
        bpy.context.scene.collection.children.link(tmp_collection)
        with bpy.data.libraries.load(piecesFile) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            tmp_collection.objects.link(obj)

        # Select black and white materials
        white_mat = random.choice([mat for mat in bpy.data.materials if 'White' in mat.name])
        black_mat = random.choice([mat for mat in bpy.data.materials if 'Black' in mat.name])

        # Place all final pieces by duplicating temporary imports
        annot['pieces'] = []
        for square, piece in enumerate(annot['position']):

            rank = 7 - int(square / 8)
            file = square % 8

            piece_obj = self.create_piece(piece, collection, white_mat, black_mat)
            if piece_obj:

                # Move piece
                piece_obj.rotation_euler.rotate_axis("Z", random.uniform(0.0, 6.32))
                piece_obj.location = (0.5 + file + random.uniform(-0.2, 0.2), rank + 0.5 + random.uniform(-0.2, 0.2), 0)

                # Add piece info
                bpy.context.view_layer.update()
                info = {}
                info['piece'] = piece
                info['bbox'] = self.get_object_bouding_box(piece_obj)
                annot['pieces'].append(info)


        # Place random pieces next to the board
        use_side_pieces = random.random() > 0.5
        if use_side_pieces or True:
            n_pieces = int(random.random() * 15)
            print(f"Adding {n_pieces} side pieces")
            for i in range(n_pieces):
                piece = random.choice(["p", "n", "b", "r", "q"]) # The king is always on the board
                if random.random() > 0.5:
                    piece = piece.upper()

                piece_obj = self.create_piece(piece, collection, white_mat, black_mat)
                if piece_obj:

                    # Move piece outside the board
                    piece_obj.rotation_euler.rotate_axis("Z", random.uniform(0.0, 6.32))
                    piece_obj.location = (9.0 + random.uniform(-1.0, 1.0), random.uniform(-2.0, 10.0), 0)

                    # Put it on a random side
                    if random.random() > 0.5:
                        piece_obj.location[0] = -piece_obj.location[0] + 8.0
                    if random.random() > 0.5:
                        tmp = piece_obj.location[0]
                        piece_obj.location[0] = piece_obj.location[1]
                        piece_obj.location[1] = tmp

                    # Add piece info
                    bpy.context.view_layer.update()
                    info = {}
                    info['piece'] = piece
                    info['bbox'] = self.get_object_bouding_box(piece_obj)
                    annot['pieces'].append(info)


        # Delete tmp collection
        for obj in tmp_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(tmp_collection)
        bpy.data.orphans_purge(do_recursive=True)

        bpy.context.view_layer.update()

        # Get board 2D projection
        annot['board'] = []
        grid_obj = bpy.data.objects['Grid']
        for vert in grid_obj.data.vertices:
            world_co = grid_obj.matrix_world @ vert.co
            image_co = world_to_camera_view(bpy.context.scene, bpy.data.objects['Camera'], world_co)
            annot['board'].append(list(image_co.to_2d()))

        # Save annotations

        with open(outputPath + '.json', 'w+') as o:
            o.write(str(json.dumps(annot, indent = 4)))

        return (str(board), collection)


    def get_output_names(self):
        return ["Board position", "Collection"]