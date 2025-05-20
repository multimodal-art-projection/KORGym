import random
from typing import List



def generate(seed: int, emoji_num: int, scale: List[int]) -> List[List[str]]:
    random.seed(seed)
    # 预定义的emoji列表，足够多的常见emoji
    all_emojis = [
        "😀", "😃", "😄", "😁", "😆", "🥰", "🏄", "🦭", "🧽", "🤚", "🚀", "🎁",
        "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮",
        "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🐤", "🐣", "🐥", "🦆", "🦅", "🦉",
        "🦇", "🐺", "🐗", "🐴", "🦄", "🐝", "🐛", "🦋", "🐌", "🐞", "🐜", "🦟",
        "🦗", "🕷", "🦂", "🐢", "🐍", "🦎", "🦖", "🦕", "🐙", "🦑", "🦐", "🦞",
        "🦀", "🐡", "🐠", "🐟", "🐬", "🐳", "🐋", "🦈", "🐊", "🐅", "🐆", "🦓",
        "🦍", "🐘", "🦏", "🦛", "🐪", "🐫", "🦒", "🦘", "🐃", "🐂", "🐄", "🐎",
        "🐖", "🐏", "🐑", "🦙", "🐐", "🐕", "🐩", "🦮", "🐈", "🐓", "🦃", "🦚",
        "🦜", "🦢", "🦩", "🦨", "🦦", "🦥", "🐿", "🦔", "🌵", "🎄", "🌲", "🌳",
        "🌴", "🌱", "🌿", "☘️", "🍀", "🎍", "🎋", "🍃", "🍂", "🍁", "🌾", "🌺",
        "🌻", "🌹", "🥀", "🌷", "🌼", "🌸", "💐", "🍄", "🌰", "🎃", "🐚", "🪐",
        "🌎", "🌍", "🌏", "🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔", "🌚",
        "🌝", "🌞", "🌙", "⭐️", "🌟", "💫", "✨", "☄️", "🔥", "💥", "🌈", "☀️",
        "⛅️", "☁️", "❄️", "💧", "💦", "🌊"
    ]
    
    # 确保不重复选择emojis
    selected_emojis = random.sample(all_emojis, emoji_num)
    
    rows, cols = scale[0], scale[1]
    board = []
    for _ in range(rows):
        row = [random.choice(selected_emojis) for _ in range(cols)]
        board.append(row)
    return board

def calculate_lines(board: List[List[str]]) -> int:
    if not board:
        return 0
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    total = 0
    
    # 检查行
    for row in board:
        current_len = 1
        current_emoji = row[0]
        for emoji in row[1:]:
            if emoji == current_emoji:
                current_len += 1
            else:
                if current_len >= 2:
                    total += 1 
                current_emoji = emoji
                current_len = 1
        if current_len >= 2:
            total += 1 
    # 检查列
    for c in range(cols):
        current_len = 1
        current_emoji = board[0][c]
        for r in range(1, rows):
            emoji = board[r][c]
            if emoji == current_emoji:
                current_len += 1
            else:
                if current_len >= 2:
                    total += 1 
                current_emoji = emoji
                current_len = 1
        if current_len >= 2:
            total += 1 
    
    return total

def verify(board: List[List[str]], answer: int) -> int:
    correct = calculate_lines(board)
    return 1 if answer == correct else 0

def test():
    # 测试样例
    board = [
        ['🏄', '🏄', '🥰', '🏄', '🦭'],
        ['🥰', '🥰', '🥰', '🥰', '🥰'],
        ['🦭', '🦭', '🥰', '🧽', '🤚']
    ]
    print(calculate_lines(board))
    assert calculate_lines(board) == 4
    
    # 修改此处：原预期6改为3
    board1 = [
        ['A', 'A', 'A'],
        ['B', 'B', 'B'],
        ['C', 'C', 'C']
    ]
    assert calculate_lines(board1) == 3  # 3行，列无
    
    # 测试部分行和列
    board2 = [
        ['A', 'B', 'A'],
        ['B', 'B', 'B'],
        ['C', 'C', 'D']
    ]
    assert calculate_lines(board2) == 3  # 2行+1列
    
    # 测试单行单列
    board3 = [
        ['A', 'A']
    ]
    assert calculate_lines(board3) == 1  # 1行
    
    board4 = [
        ['A'],
        ['A']
    ]
    assert calculate_lines(board4) == 1  # 1列
    
    print("All test cases passed!")

if __name__ == "__main__":
    board=generate(1223,3,[5,5])
    for line in board:
        print("".join(line))

    print(calculate_lines(board))
    test()