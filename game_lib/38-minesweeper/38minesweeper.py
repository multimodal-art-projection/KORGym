import random

# 全局变量保存各轮游戏的状态
games = {}
current_epoch = 0

def generate(seed):
    global current_epoch
    random.seed(seed)
    
    # 默认使用9x9棋盘，10个地雷
    rows, cols = 9, 9
    mines = 10
    
    # 生成地雷位置
    positions = [(i, j) for i in range(rows) for j in range(cols)]
    mine_pos = random.sample(positions, mines)
    
    # 初始化棋盘
    board = [[0]*cols for _ in range(rows)]
    for i, j in mine_pos:
        board[i][j] = -1
    
    # 计算数字提示
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),         (0,1),
                  (1,-1),  (1,0),  (1,1)]
    
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == -1:
                continue
            count = 0
            for dx, dy in directions:
                x, y = i+dx, j+dy
                if 0 <= x < rows and 0 <= y < cols:
                    if board[x][y] == -1:
                        count += 1
            board[i][j] = count
    
    # 保存游戏状态
    current_epoch += 1
    games[current_epoch] = {
        'actual': [row[:] for row in board],
        'mask': [['?' for _ in range(cols)] for _ in range(rows)],
        'score': 0.0,
        'mines': mines,
        'game_over': False
    }
    
    return board

def reveal_empty(actual, mask, row, col):
    """递归揭开空白区域"""
    rows, cols = len(actual), len(actual[0])
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if mask[r][c] != '?':
            continue
        
        val = actual[r][c]
        if val > 0:
            mask[r][c] = str(val)
            continue
        
        mask[r][c] = '0'
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nr, nc = r+dx, c+dy
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append((nr, nc))

def verify(board, score, epoch, action):
    if epoch not in games:
        return board, score, epoch
    
    game = games[epoch]
    if game['game_over']:
        return game['mask'], game['score'], epoch
    
    try:
        cmd, pos = action.strip().split(' ', 1)
        cmd = cmd.lower()
        row, col = map(int, pos.strip('()').split(','))
    except:
        return game['mask'], game['score'], epoch
    
    rows = len(game['actual'])
    cols = len(game['actual'][0])
    
    if not (0 <= row < rows and 0 <= col < cols):
        return game['mask'], game['score'], epoch
    
    mask = game['mask']
    actual = game['actual']
    
    if cmd == 'uncover':
        if mask[row][col] != '?':
            return game['mask'], game['score'], epoch
        
        if actual[row][col] == -1:  # 踩雷
            game['game_over'] = True
            correct = sum(1 for i in range(rows) for j in range(cols)
                          if mask[i][j] == 'F' and actual[i][j] == -1)
            game['score'] = correct / game['mines']
            # 显示所有地雷
            for i in range(rows):
                for j in range(cols):
                    if actual[i][j] == -1:
                        mask[i][j] = 'X'
            
        else:
            reveal_empty(actual, mask, row, col)
            # 检查胜利条件
            revealed = sum(cell != '?' and cell != 'F' for row in mask for cell in row)
            total_safe = rows*cols - game['mines']
            if revealed == total_safe:
               game['game_over'] = True
                # 直接得满分
               game['score'] = 1.0  
    
    elif cmd == 'flag':
        if mask[row][col] == '?':
            mask[row][col] = 'F'
            if actual[row][col] == -1:
                game['score'] += 1/game['mines']
        elif mask[row][col] == 'F':
            mask[row][col] = '?'
            if actual[row][col] == -1:
                game['score'] -= 1/game['mines']

         # 检查是否满足胜利条件：标记数等于地雷数且全部正确
        if check_victory(game):
            game['game_over'] = True
            # 直接计算得分为满分（正确标记数 / 总雷数 = 1.0）
            game['score'] = 1.0    
    # 更新游戏状态
    game['mask'] = mask
    return game['mask'], game['score'], epoch

def check_victory(game):
    """检查胜利条件：标记数等于地雷数且所有标记正确"""
    total_flags = sum(row.count('F') for row in game['mask'])
    correct_flags = 0
    for i in range(len(game['actual'])):
        for j in range(len(game['actual'][0])):
            if game['mask'][i][j] == 'F' and game['actual'][i][j] == -1:
                correct_flags += 1
    return total_flags == game['mines'] and correct_flags == game['mines']

def test_perfect_flags():
    global games, current_epoch
    games = {}
    current_epoch = 0

    seed = 456
    generate(seed)
    epoch = current_epoch
    game = games[epoch]

    # 标记所有地雷
    mines = [(i, j) for i in range(9) for j in range(9) if game['actual'][i][j] == -1]
    for r, c in mines:
        verify(None, 0.0, epoch, f"flag {r},{c}")

    assert game['score'] == 1.0, "正确标记应得1.0分"
    print("测试用例3通过：正确标记所有地雷得满分")

def test_score_when_hit_mine():
    # 重置全局状态
    global games, current_epoch
    games = {}
    current_epoch = 0

    # 生成固定棋盘
    seed = 123
    generate(seed)
    epoch = current_epoch
    game = games[epoch]

    # 正确标记3个地雷
    mines = [(i, j) for i in range(9) for j in range(9) if game['actual'][i][j] == -1]
    for r, c in mines[:3]:
        verify(None, 0.0, epoch, f"flag {r},{c}")

    # 踩第四个地雷（未被标记）
    unmarked_mine = mines[3]
    verify(None, game['score'], epoch, f"uncover {unmarked_mine[0]},{unmarked_mine[1]}")

    # 验证得分：3/10 = 0.3
    assert abs(game['score'] - 0.3) < 1e-9, f"得分应为0.3，实际是{game['score']}"
    print("测试通过：踩雷时得分基于正确标记数计算")
def test_uncover_all_safe():
    global games, current_epoch
    games = {}
    current_epoch = 0

    seed = 123
    generate(seed)
    epoch = current_epoch
    game = games[epoch]

    # 揭开所有非雷格
    for i in range(9):
        for j in range(9):
            if game['actual'][i][j] != -1:
                verify(None, game['score'], epoch, f"uncover {i},{j}")

    assert game['score'] == 1.0, "应得满分1.0"
    print("测试用例2通过：揭开所有安全格得满分")
def test_hit_mine_score():
    global games, current_epoch
    games = {}
    current_epoch = 0

    seed = 123
    generate(seed)
    epoch = current_epoch
    game = games[epoch]

    # 找到第一个地雷并踩雷
    mine = next((i, j) for i in range(9) for j in range(9) if game['actual'][i][j] == -1)
    verify(None, 0.0, epoch, f"uncover {mine[0]},{mine[1]}")

    assert game['score'] == 0.0, "踩雷后得分应为0"
    print("测试用例1通过：踩雷得0分")
# 示例用法
if __name__ == "__main__":
    test_hit_mine_score()
    test_uncover_all_safe()
    test_perfect_flags()
    test_score_when_hit_mine()
 
    

 