import cv2
import numpy as np
import glob
import json
import chess


from utils import draw_results_on_image
from chessboard_parser import ChessboardParser

DEVICE=3
USE_YOLO=True
# USE_YOLO=False
BATCH_SIZE=64

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
    parser = ChessboardParser(device=DEVICE, yolo_detect=USE_YOLO)

    with open("/workspace/ChessLink/data/chessred_test/annotations.json") as f:
        annots = json.loads(f.read())

    perfect = 0
    total = 0
    errors={}

    index = 0
    print(" ")
    while index < len(images_paths):

        batch_gt_boards = []
        images = []
        for k in range(index, index + BATCH_SIZE):
            if k >= len(images_paths):
                break

            gt_board = chess.Board.empty()
            image_path = images_paths[k]
            annot = [a for a in annots["images"] if image_path.endswith(a["file_name"])][0]
            gt_pieces = [a for a in annots["annotations"]["pieces"] if a["image_id"] == annot["id"]]
            for piece in gt_pieces:
                s = chess.SQUARE_NAMES.index(piece["chessboard_position"])
                p = chess.Piece.from_symbol(labels[piece["category_id"]])
                gt_board.set_piece_at(s, p)
            batch_gt_boards.append(gt_board)

            images.append(cv2.imread(image_path))

        batch_results = parser.process_images(images)

        for k, (results, gt_board) in enumerate(zip(batch_results, batch_gt_boards)):

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

                if "k-q" in min_diffs:
                    cv2.imwrite(f"./output/images/{index+k}.jpg", images[k])
                for diff in min_diffs:
                    key = f'{diff[0].lower()}{diff[2].lower()}'
                    errors[key] = errors.get(key, 0) + 1

            total += 1
            filtered_errors = list(filter(lambda x:x[1]>10, sorted(errors.items(), key=lambda x:-x[1])))

        print(f'\033[F{index + BATCH_SIZE}/{len(images_paths)} {int(perfect/total*100)}% {filtered_errors}')

        # if results["info"] == "ok" and min_count > 0:
        #     image = cv2.imread(image_path)
        #     draw_results_on_image(image, results)
        #     cv2.imwrite("/workspace/ChessLink/evaluate.jpg", image)
        #     a = input()

        # if not match:

            # print(board)
            # print("----")
            # print(gt_board)

        # print(board)

        index += BATCH_SIZE
