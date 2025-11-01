This bot plays **Ultimate Tic-Tac-Toe** using a **Minimax algorithm with Alpha-Beta pruning** and heuristics to evaluate board states.  
It chooses moves strategically to **maximize its chances of winning** while **minimizing the opponent's opportunities**.
The main.py offers code which runs on codinggame
https://www.codingame.com/profile/36485cdf74d79814b1678b84c98cb5dc7642284 - codinggame profile

---

## Main Components

### 1. Board Representation
- `mini_boards`: 9 mini 3x3 boards
- `meta_board`: 3x3 meta board representing wins in mini-boards
- `active_board`: The mini-board where the next move must be played (`-1` means any board)

### 2. Move Conversion
- `global_to_local(r, c)`: Converts overall board coordinates to mini-board coordinates
- `local_to_global(b, r, c)`: Converts mini-board coordinates back to global coordinates
- This step was necessary because codinggame requires rows and columns as arguments, while the bot works with an additional argument - board

### 3. Valid Moves
- `get_valid_moves()`: Returns all possible moves on active boards or all boards if `active_board = -1`

### 4. Applying Moves
- `apply_move(move, player)`: Updates the board, checks mini-board winners, sets the next active_board

### 5. Evaluation Function
- `evaluate(player)`: Scores the current board state
  - Counts wins (two in a row with a chance to finish)
  - Counts threats (potential opportunities or opponent threats)
  - Considers positional control (center, corners, edges)
  - Penalizes sending the opponent to advantageous boards
  - The more score, the better is the move.

### 6. Move Ordering
- `order_moves(moves, player)`: Prioritizes moves for better pruning
  - Center > corners > edges
  - Moves creating threats are prioritized
  - This function optimizes speed of decision, not considering irrelevant branches.

### 7. Minimax with Alpha-Beta Pruning
- Recursively explores possible moves to a given depth
- Uses `alpha` and `beta` to prune branches
- Checks time limit to stay under CodinGame’s 100ms per turn
- Returns `(best_score, best_move)`

### 8. Game Loop
- Reads the opponent’s move
- Converts moves to internal format and updates active_board
- Dynamically adjusts search depth
- Calls `minimax()` to select the best move
- Fallback ensures a valid move if Minimax fails

