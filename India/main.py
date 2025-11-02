# CODING GAME CODE
import sys
import time


class UltimateTicTacToe:
    def __init__(self):
        self.mini_boards = [
            [[0 for _ in range(3)] for _ in range(3)] for _ in range(9)]
        self.meta_board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.active_board = -1
        self.last_move = None
        self.move_count = 0

    @staticmethod
    def global_to_local(r, c):
        b = (r // 3) * 3 + (c // 3)
        return b, r % 3, c % 3

    @staticmethod
    def local_to_global(b, r, c):
        return (b // 3) * 3 + r, (b % 3) * 3 + c

    def get_valid_moves(self):
        moves = []
        boards = [self.active_board] if self.active_board != - \
            1 else [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for b in boards:
            if self.meta_board[b] != 0:
                continue
            for row in range(3):
                for col in range(3):
                    if self.mini_boards[b][row][col] == 0:
                        moves.append((b, row, col))
        return moves

    def apply_move(self, move, player):
        b, row, col = move
        self.mini_boards[b][row][col] = player
        self.last_move = (b, row, col)
        self.meta_board[b] = self.check_winner(self.mini_boards[b])
        next_board = 3 * row + col
        self.active_board = next_board if self.meta_board[next_board] == 0 else -1
        self.move_count += 1

    def check_winner(self, board):
        for i in range(3):
            if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
            if board[0][i] != 0 and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]
        if all(cell != 0 for row in board for cell in row):
            return 2
        return 0

    def is_terminal(self):
        meta_board_3x3 = [[0] * 3 for _ in range(3)]
        for i in range(9):
            meta_board_3x3[i // 3][i %
                                   3] = self.meta_board[i] if self.meta_board[i] in [1, -1] else 0
        return self.check_winner(meta_board_3x3) != 0

    def clone(self):
        new_board = UltimateTicTacToe()
        new_board.mini_boards = [[row[:] for row in board]
                                 for board in self.mini_boards]
        new_board.meta_board = self.meta_board[:]
        new_board.active_board = self.active_board
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        return new_board

    def count_threats(self, board, player):
        threats = 0
        for i in range(3):
            if board[i].count(player) == 1 and board[i].count(0) == 2:
                threats += 1
        for i in range(3):
            col = [board[j][i] for j in range(3)]
            if col.count(player) == 1 and col.count(0) == 2:
                threats += 1
        diag1 = [board[0][0], board[1][1], board[2][2]]
        if diag1.count(player) == 1 and diag1.count(0) == 2:
            threats += 1
        diag2 = [board[0][2], board[1][1], board[2][0]]
        if diag2.count(player) == 1 and diag2.count(0) == 2:
            threats += 1
        return threats

    def count_wins(self, board, player):
        wins = 0
        for i in range(3):
            if board[i].count(player) == 2 and board[i].count(0) == 1:
                wins += 1
        for i in range(3):
            col = [board[j][i] for j in range(3)]
            if col.count(player) == 2 and col.count(0) == 1:
                wins += 1
        diag1 = [board[0][0], board[1][1], board[2][2]]
        if diag1.count(player) == 2 and diag1.count(0) == 1:
            wins += 1
        diag2 = [board[0][2], board[1][1], board[2][0]]
        if diag2.count(player) == 2 and diag2.count(0) == 1:
            wins += 1
        return wins

    def evaluate_board_control(self, board, player):
        score = 0
        if board[1][1] == player:
            score += 4
        elif board[1][1] == -player:
            score -= 4
        corners = [board[0][0], board[0][2], board[2][0], board[2][2]]
        score += corners.count(player) * 3
        score -= corners.count(-player) * 3
        edges = [board[0][1], board[1][0], board[1][2], board[2][1]]
        score += edges.count(player) * 2
        score -= edges.count(-player) * 2

        return score

    def evaluate(self, player):
        score = 0

        position_value = [3, 2, 3, 2, 4, 2, 3, 2, 3]

        meta_3x3 = [[0] * 3 for _ in range(3)]
        for i in range(9):
            if self.meta_board[i] == player:
                meta_3x3[i // 3][i % 3] = player

                score += 800 * position_value[i]
            elif self.meta_board[i] == -player:
                meta_3x3[i // 3][i % 3] = -player
                score -= 800 * position_value[i]

        # Immediate meta-board win/loss (CRITICAL)
        meta_wins = self.count_wins(meta_3x3, player)
        meta_opp_wins = self.count_wins(meta_3x3, -player)
        if meta_wins > 0:
            score += 100000
        if meta_opp_wins > 0:
            score -= 150000

        # Meta-board threats
        meta_threats = self.count_threats(meta_3x3, player)
        meta_opp_threats = self.count_threats(meta_3x3, -player)
        score += meta_threats * 2000
        score -= meta_opp_threats * 2500

        for b in range(9):
            if self.meta_board[b] != 0:
                continue

            mini = self.mini_boards[b]
            multiplier = position_value[b]

            my_wins = self.count_wins(mini, player)
            opp_wins = self.count_wins(mini, -player)
            score += my_wins * 100 * multiplier
            score -= opp_wins * 120 * multiplier

            my_threats = self.count_threats(mini, player)
            opp_threats = self.count_threats(mini, -player)
            score += my_threats * 40 * multiplier
            score -= opp_threats * 45 * multiplier

            # Positional control
            score += self.evaluate_board_control(mini, player) * multiplier

        # === STRATEGIC CONSIDERATIONS ===
        # Penalize sending opponent to advantageous boards
        if self.last_move:
            _, lr, lc = self.last_move
            next_b = 3 * lr + lc
            if self.meta_board[next_b] == 0:
                # If we sent opponent to center board, that's bad
                if next_b == 4:
                    score -= 50
                # If opponent can win that board easily, penalty
                opp_wins_next = self.count_wins(
                    self.mini_boards[next_b], -player)
                if opp_wins_next > 0:
                    score -= 80

        return score

    def order_moves(self, moves, player):
        """Order moves for better alpha-beta pruning (speed optimization)"""
        def move_priority(move):
            b, r, c = move
            priority = 0

            # Prioritize center positions
            if r == 1 and c == 1:
                priority += 1000
            # Prioritize corner positions
            elif (r, c) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                priority += 500

            # Prioritize center board
            if b == 4:
                priority += 200

            # Quick evaluation: does this create a threat?
            test_board = [row[:] for row in self.mini_boards[b]]
            test_board[r][c] = player
            if self.count_threats(test_board, player) > self.count_threats(self.mini_boards[b], player):
                priority += 2000

            return priority

        return sorted(moves, key=move_priority, reverse=True)

    def minimax(self, depth, player, maximizingPlayer, alpha=-float('inf'), beta=float('inf'), start_time=None, time_limit=0.08):
        # Timeout check (leave buffer for output)
        if start_time and time.time() - start_time > time_limit:
            return self.evaluate(player), None

        if depth == 0 or self.is_terminal():
            return self.evaluate(player), None

        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return self.evaluate(player), None

        # Move ordering for better pruning
        if depth >= 2:
            valid_moves = self.order_moves(valid_moves, player)

        best_move = valid_moves[0]

        if maximizingPlayer:
            max_eval = -float('inf')
            for move in valid_moves:
                new_board = self.clone()
                new_board.apply_move(move, player)
                eval_score, _ = new_board.minimax(
                    depth - 1, player, False, alpha, beta, start_time, time_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                new_board = self.clone()
                new_board.apply_move(move, -player)
                eval_score, _ = new_board.minimax(
                    depth - 1, player, True, alpha, beta, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_move


# Initialize game state
board = UltimateTicTacToe()
player_bot = 1
player_opponent = -1
turn_num = 0

# Game loop
while True:
    opponent_row, opponent_col = [int(i) for i in input().split()]
    valid_action_count = int(input())
    actions = []
    for i in range(valid_action_count):
        row, col = [int(j) for j in input().split()]
        actions.append((row, col))

    # Apply opponent's move
    if opponent_row != -1 and opponent_col != -1:
        b, r, c = UltimateTicTacToe.global_to_local(opponent_row, opponent_col)
        board.apply_move((b, r, c), player_opponent)

    # Convert valid actions to internal format
    valid_moves_internal = []
    for row, col in actions:
        b, r, c = UltimateTicTacToe.global_to_local(row, col)
        valid_moves_internal.append((b, r, c))

    # Update active board
    if valid_moves_internal:
        active_boards = set(m[0] for m in valid_moves_internal)
        board.active_board = list(active_boards)[0] if len(
            active_boards) == 1 else -1

    # Adaptive depth based on game complexity
    if board.move_count < 10:
        depth = 2  # Early game: many branches
    elif board.move_count < 25:
        depth = 3  # Mid game
    elif board.move_count < 50:
        depth = 4  # Late-mid game
    else:
        depth = 5  # Endgame: few branches, can search deep

    # Adjust depth based on number of valid moves
    if len(valid_moves_internal) > 50:
        depth = min(depth, 2)
    elif len(valid_moves_internal) > 30:
        depth = min(depth, 3)

    # Time limit: 80ms (leave 20ms buffer from 100ms limit)
    time_limit = 0.9 if turn_num == 1 else 0.08  # First turn gets 900ms

    start = time.time()
    _, best_move = board.minimax(depth=depth, player=player_bot, maximizingPlayer=True,
                                 start_time=start, time_limit=time_limit)

    # Fallback if minimax failed
    if best_move is None or best_move not in valid_moves_internal:
        # Prefer center moves
        center_moves = [
            m for m in valid_moves_internal if m[1] == 1 and m[2] == 1]
        best_move = center_moves[0] if center_moves else valid_moves_internal[0]

    # Apply and output move
    board.apply_move(best_move, player_bot)
    global_row, global_col = UltimateTicTacToe.local_to_global(
        best_move[0], best_move[1], best_move[2])

    print(f"{global_row} {global_col}")
