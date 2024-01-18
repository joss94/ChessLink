import operator
import chess
import chess.pgn

def get_italian_board():
    board = chess.Board(chess.STARTING_FEN)
    board.push_uci('e2e4')
    board.push_uci('e7e5')
    board.push_uci('g1f3')
    board.push_uci('b8c6')
    board.push_uci('f1c4')
    board.push_uci('f8c5')
    return board

def get_reachable_positions(board, n_moves):
        positions = []
        for move in board.legal_moves:
            board.push(move)
            if n_moves == 1:
                positions.append(board.fen())
            else:
                positions.extend(get_reachable_positions(board, n_moves-1))
            board.pop()
        return positions

def board_fullstring(board):
    out = ""
    for i in range(64):
        p = board.piece_at(i)
        out += p.symbol() if p else "."
    return out

def moves_between_positions(board1, board2, n_moves, try_both_colors=False, still_squares=[]):
    possible_moves = []
    target = board_fullstring(board2)
    lowest_moves_count = 1e6

    if board_fullstring(board1).lower() == target.lower():
        return []

    def boards_loss(b1, b2):

        loss = 0
        for a, b in zip(board_fullstring(b1), board_fullstring(b2)):
            if a != b:
                if a == '.' or b == '.':
                    loss += 10
                    continue
                if a.lower() != b.lower():
                    loss += 3
                if a.isupper() != b.isupper():
                    loss += 3
        return loss

    def try_position(board, move_nb=1):
        nonlocal lowest_moves_count

        if move_nb > n_moves:
            return

        loss = boards_loss(board, board2)
        possible_moves.append((board.move_stack[-move_nb:] if move_nb > 0 else [], loss, move_nb))
        if loss == 0 and (move_nb < lowest_moves_count or lowest_moves_count < 0):
            lowest_moves_count = move_nb

        # If we already found a sequence matching the target exactly, do not go further
        if lowest_moves_count <= move_nb:
            return

        for move in board.legal_moves:

            if move.from_square in still_squares or move.to_square in still_squares:
                continue

            board.push(move)
            try_position(board, move_nb+1)
            board.pop()

    try_position(board1, move_nb=0)
    if try_both_colors:
        board1_bis = board1.copy()
        board1_bis.turn = chess.BLACK if board1.turn == chess.WHITE else chess.WHITE
        try_position(board1_bis, move_nb=0)

    possible_moves.sort(key=operator.itemgetter(1, 2))

    return possible_moves

def board_pgn_string(board):
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter)

def board_to_string(board, add_moves=True):

    string = "\n"
    if add_moves:
        string += board_pgn_string(board) + "\n"

    string += '   --------------- \n'
    for row in range(7, -1, -1):
        string += f'{row+1} |'
        for col in range(0, 8):
            p = board.piece_at(chess.square(col, row))
            string += p.symbol() if p is not None else '.'
            string += ' ' if col <7 else '|\n'
    string += '   --------------- \n'
    string += "   a b c d e f g h\n"
    return string

def boards_to_string(boards):
    out = ""
    if len(boards) == 0:
        return out

    boards_str = [board_to_string(board, False).split("\n") for board in boards]

    for line_nb in range(len(boards_str[0])):
        for board in boards_str:
            out += board[line_nb] + "   "
        out += "\n"
    return out



# board = chess.Board(chess.STARTING_FEN)
# board.push_uci('e2e4')
# board.push_uci('e7e5')
# board.push_uci('g1f3')

# board2 = get_italian_board()
# board2.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.BLACK))

# # positions = get_reachable_positions(board, 4)
# # for p in positions:
# #     print(chess.Board(p))
# # print(f"{len(positions)} POSITIONS FOUND")

# moves = moves_between_positions(board, board2, n_moves=3)
# moves.sort(key=lambda x: x[1])
# print("POSSIBLE MOVES:")

# for seq, ndiff in moves:
#     board_copy = board.copy()
#     for m in seq:
#         board_copy.push(m)
#     print(board_copy)
#     print(ndiff)
