import random

def generate_candidates(rows, cols, max_region_size=None):
    """
    生成所有候选“加号区域”。
    每个候选区域由一个提示格 (r,c) 及扩展参数 a, b, u, d 决定，
    区域覆盖：
      - 水平方向：所在行从 (r, c-a) 到 (r, c+b)
      - 垂直方向：所在列从 (r-u, c) 到 (r+d, c)
    区域大小 = a+b+u+d+1，对应提示数字 = 区域大小 - 1
    为避免提示数字为0，要求区域大小至少为2。
    参数 max_region_size 可限制区域最大面积（如需要）。
    """
    candidates = []
    for r in range(rows):
        for c in range(cols):
            max_left = c
            max_right = cols - 1 - c
            max_up = r
            max_down = rows - 1 - r
            for a in range(max_left + 1):      # 向左扩展 a 格
                for b in range(max_right + 1): # 向右扩展 b 格
                    for u in range(max_up + 1):    # 向上扩展 u 格
                        for d in range(max_down + 1):  # 向下扩展 d 格
                            # 如果所有扩展均为 0，则区域大小为1，跳过
                            if a == 0 and b == 0 and u == 0 and d == 0:
                                continue
                            # 计算区域覆盖的所有单元格（集合避免重复计入提示格）
                            cells = set()
                            # 水平方向（所在行）
                            for j in range(c - a, c + b + 1):
                                cells.add((r, j))
                            # 垂直方向（所在列）
                            for i in range(r - u, r + d + 1):
                                cells.add((i, c))
                            region_size = len(cells)  # 应为 a+b+u+d+1
                            # 若设置了区域大小上限则跳过超出者
                            if max_region_size is not None and region_size > max_region_size:
                                continue
                            candidate = {
                                'clue': (r, c),
                                'left': a,
                                'right': b,
                                'up': u,
                                'down': d,
                                'cells': cells,
                                'digit': region_size - 1  # 提示数字（必然>=1）
                            }
                            candidates.append(candidate)
    return candidates

def solve_tiling(rows, cols, candidates):
    """
    在棋盘 (rows x cols) 上，从候选区域中选择一组互不重叠的区域，
    使得这些区域的并集正好覆盖整个棋盘。
    
    优化策略：
      1. 预先构建每个单元格到候选区域的映射（cell_to_candidates）。
      2. 每次选择未覆盖单元格时，选取候选区域数目最少的单元格（MRV 启发式）。
    """
    board_cells = {(i, j) for i in range(rows) for j in range(cols)}
    
    # 构建每个单元格到候选区域的映射
    cell_to_candidates = { (i, j): [] for i in range(rows) for j in range(cols) }
    for cand in candidates:
        for cell in cand['cells']:
            if cell in cell_to_candidates:
                cell_to_candidates[cell].append(cand)
    
    def backtrack(covered, solution):
        if covered == board_cells:
            return solution
        # 在未覆盖单元格中选择候选区域数最少的那个（MRV 启发式）
        uncovered = board_cells - covered
        best_cell = None
        best_cands = None
        best_count = float('inf')
        for cell in uncovered:
            # 计算当前 cell 的可用候选区域（与已覆盖区域不冲突）
            valid_cands = [cand for cand in cell_to_candidates[cell] if cand['cells'].isdisjoint(covered)]
            if len(valid_cands) < best_count:
                best_count = len(valid_cands)
                best_cell = cell
                best_cands = valid_cands
            if best_count == 0:
                # 该单元格无候选区域，则当前解不可行
                return None
        
        # 随机打乱候选顺序
        random.shuffle(best_cands)
        for cand in best_cands:
            new_covered = covered.union(cand['cells'])
            solution.append(cand)
            res = backtrack(new_covered, solution)
            if res is not None:
                return res
            solution.pop()
        return None

    return backtrack(set(), [])

def solve_tiling_with_retries(seed, rows, cols, max_total_attempts=50, max_attempts_per_candidate=1):
    """
    尝试求解铺板问题。如果连续多次使用同一候选列表未能求解，
    则重新生成候选区域，最多尝试 max_total_attempts 次，避免长时间卡住。
    """
    random.seed(seed)
    total_attempts = 0
    while total_attempts < max_total_attempts:
        candidates = generate_candidates(rows, cols)
        for _ in range(max_attempts_per_candidate):
            solution = solve_tiling(rows, cols, candidates)
            total_attempts += 1
            if solution is not None:
                return solution
    return None

def create_boards(solution, rows, cols):
    """
    根据区域铺板解答生成：
      - 完整解答棋盘：每个单元格填入所在区域的字母；
      - 谜题棋盘：仅在每个区域的提示格位置显示字母，其余为 '0'；
      - 提示映射：字母 -> digit（digit = 区域大小 - 1）
    """
    sol_board = [['' for _ in range(cols)] for _ in range(rows)]
    puzzle_board = [['0' for _ in range(cols)] for _ in range(rows)]
    clues = {}
    # 为每个区域分配一个字母（按顺序 a, b, c, …）
    for index, region in enumerate(solution):
        letter = chr(97 + index)
        r, c = region['clue']
        clues[letter] = region['digit']
        # 在完整解答中，将区域所有单元格填为该字母
        for (i, j) in region['cells']:
            sol_board[i][j] = letter
        # 在谜题中，仅提示格位置显示字母，其余保持 '0'
        puzzle_board[r][c] = letter
    return sol_board, puzzle_board, clues

def print_board(board):
    for row in board:
        print(''.join(row))

def verify(puzzle_board, clues, generated_answer):
    """
    验证 generated_answer 是否为合法答案。要求：
      1. puzzle_board 与 generated_answer 尺寸一致；
      2. 对于每个提示字母 letter（及其提示数字 clues[letter]）：
         - puzzle_board 中该 letter 出现恰好一次，其位置 (r,c) 为提示格，
           且 generated_answer[r][c] 必须为 letter；
         - generated_answer 中所有 letter 格子必须都位于行 r 或列 c 上；
         - 在 generated_answer 中，以 (r,c) 为中心沿所在行向左右扫描，得到横向连续段；
           同理沿所在列向上下扫描得到纵向连续段；两段的并集（expected_region）
           应与 generated_answer 中所有 letter 格子（region_cells）相同；
         - 令 expected_clue = (横向段长度 – 1) + (纵向段长度 – 1)，允许其与 clues[letter] 差值不超过 1。
      3. generated_answer 中所有格子均应属于某个 clues 中的字母。
      
    返回 True 表示答案合法，否则返回 False。
    """
    rows = len(puzzle_board)
    if rows == 0:
        return False
    cols = len(puzzle_board[0])
    # 检查尺寸一致
    if len(generated_answer) != rows or any(len(row) != cols for row in generated_answer):
        return False

    valid_letters = set(clues.keys())
    # 检查 generated_answer 每个格子均为有效字母
    for r in range(rows):
        for c in range(cols):
            if generated_answer[r][c] not in valid_letters:
                return False

    for letter, clue_val in clues.items():
        # 找到 puzzle_board 中 letter 的提示格，必须恰好1个
        clue_positions = [(r, c) for r in range(rows) for c in range(cols) if puzzle_board[r][c] == letter]
        if len(clue_positions) != 1:
            return False
        clue_r, clue_c = clue_positions[0]
        # 检查 generated_answer 中该位置为 letter
        if generated_answer[clue_r][clue_c] != letter:
            return False

        # region_cells：generated_answer 中所有 letter 格子
        region_cells = {(r, c) for r in range(rows) for c in range(cols) if generated_answer[r][c] == letter}
        # 每个 cell 必须与 (clue_r, clue_c) 在同一行或同一列
        for (r, c) in region_cells:
            if r != clue_r and c != clue_c:
                return False

        # 在 clue 所在行，扫描出以 (clue_r, clue_c) 为中心的连续 letter 段
        row_segment = set()
        # 向左扫描
        c_left = clue_c
        while c_left >= 0 and generated_answer[clue_r][c_left] == letter:
            row_segment.add((clue_r, c_left))
            c_left -= 1
        # 向右扫描
        c_right = clue_c + 1
        while c_right < cols and generated_answer[clue_r][c_right] == letter:
            row_segment.add((clue_r, c_right))
            c_right += 1

        # 在 clue 所在列，扫描出连续 letter 段
        col_segment = set()
        r_up = clue_r - 1
        while r_up >= 0 and generated_answer[r_up][clue_c] == letter:
            col_segment.add((r_up, clue_c))
            r_up -= 1
        r_down = clue_r + 1
        while r_down < rows and generated_answer[r_down][clue_c] == letter:
            col_segment.add((r_down, clue_c))
            r_down += 1

        expected_region = row_segment | col_segment
        if expected_region != region_cells:
            return False

        horizontal_extension = len(row_segment) - 1
        vertical_extension = len(col_segment) - 1
        expected_clue = horizontal_extension + vertical_extension
        if abs(expected_clue - clue_val) > 1:
            return False

    return True

if __name__ == "__main__":
    # 设定棋盘大小（例如 5x5）
    rows, cols = 5, 6
    for i in range(100):
        solution = solve_tiling_with_retries(i, rows, cols)
        if solution is None:
            print("未能找到合法的铺板方案，请重试或调整参数。")
        else:
            sol_board, puzzle_board, clues = create_boards(solution, rows, cols)
            print("生成的谜题棋盘（仅提示格显示字母，其余为 0）：")
            print_board(puzzle_board)
            print("\n提示映射（字母: 数字，其中数字 = 扩展格数，均 >= 1）：")
            for letter in sorted(clues):
                print(f"{letter}:{clues[letter]}")
            print("\n完整解答棋盘：")
            print_board(sol_board)
            print(f"示例 {i} 验证结果：", verify(puzzle_board, clues, sol_board))
