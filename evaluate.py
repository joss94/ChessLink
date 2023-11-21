import cv2
import numpy as np
import glob
import json
import chess


from utils import draw_results_on_image
from chessboard_parser import ChessboardParser

USE_YOLO=True
# USE_YOLO=False

labels=["P", "R", "N", "B", "Q", "K", "p", "r", "n", "b", "q", "k"]

def rotate_board(board):
    new_board = chess.Board.empty()
    for r in range(8):
        for c in range(8):
            src_square = chess.square(c, r)
            piece = board.piece_at(src_square)
            if piece:
                dst_square = chess.square(r, 7-c)
                new_board.set_piece_at(dst_square, piece)
    return new_board

if __name__ == "__main__":

    images_paths = glob.glob("/workspace/ChessLink/data/chessred_test/*/*.jpg")
    # images_paths = ["/workspace/ChessLink/data/chessred_test/0/G000_IMG040.jpg"]
    parser = ChessboardParser(device=0, yolo_detect=USE_YOLO)

    with open("/workspace/ChessLink/data/chessred_test/annotations.json") as f:
        annots = json.loads(f.read())

    perfect = 0
    total = 0
    errors={}
    for image_path in images_paths:

        annot = [a for a in annots["images"] if image_path.endswith(a["file_name"])][0]
        gt_pieces = [a for a in annots["annotations"]["pieces"] if a["image_id"] == annot["id"]]

        gt_board = chess.Board.empty()
        for piece in gt_pieces:
            s = chess.SQUARE_NAMES.index(piece["chessboard_position"])
            p = chess.Piece.from_symbol(labels[piece["category_id"]])
            gt_board.set_piece_at(s, p)

        results = parser.process_images([cv2.imread(image_path)])[0]
        board = chess.Board.empty()

        min_diffs = ""
        if results["info"] == "ok":
            board.set_board_fen(results["board"])

            min_count = 1e3
            for i in range(4):
                board = rotate_board(board)

                str1 = str(gt_board)#.replace("k", "q").replace("K", "Q")
                str2 = str(board)#.replace("k", "q").replace("K", "Q")

                diffs = [f"{a}-{b}" for a, b in zip(str1, str2) if a != b]# and a.lower() != "k" and b.lower() != "k"]

                count = len(diffs)
                if count < min_count:
                    min_count = count
                    min_diffs = diffs
                if count==0:
                    perfect+=1
                    break

            for diff in min_diffs:
                key = diff[0].lower()
                errors[key] = errors.get(key, 0) + 1
        else:
            print(results["info"])




        # if results["info"] == "ok" and min_count > 5:
        #     image = cv2.imread(image_path)
        #     draw_results_on_image(image, results)
        #     cv2.imwrite("/workspace/ChessLink/evaluate.jpg", image)
        #     a = input()


        total += 1
        print(f'({int(perfect/total*100)}%) - {errors} - {min_diffs}')
        # if not match:

            # print(board)
            # print("----")
            # print(gt_board)

        # print(board)
