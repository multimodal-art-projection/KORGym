import math
import random
import os
import time
from PIL import Image, ImageDraw, ImageFont

# 定义元素颜色映射
COLOR_MAP = {
    "P": "#FF0000",  # 红色 - 玩家
    "W": "#808080",  # 灰色 - 墙
    "E": "#FFFFFF",  # 白色 - 空地
    "J": "#00FF00",  # 绿色 - 跳板
    "A": "#FFA500",  # 橙色 - 反向器
    "T": "#800080",  # 紫色 - 陷阱
    "R": "#FFFF00",  # 黄色 - 重复器
    "default": "#0000FF"  # 蓝色 - 其他元素（如传送门）
}

def draw_map(game_map, current_pos, save_filename=None, step=0):
    """
    将地图和玩家位置保存为图片
    :param game_map: 二维列表，地图数据
    :param current_pos: 元组 (x, y)，玩家当前位置
    :param step: 当前步骤编号（用于文件名）
    """
    # 参数设置
    cell_size = 50  # 每个单元格的像素大小
    border = 2      # 单元格边框宽度
    
    rows = len(game_map)
    cols = len(game_map[0]) if rows > 0 else 0
    
    # 创建画布
    img_width = cols * cell_size + (cols + 1) * border
    img_height = rows * cell_size + (rows + 1) * border
    img = Image.new("RGB", (img_width, img_height), color="#000000")
    draw = ImageDraw.Draw(img)
    
    # 加载字体
    font_size = 24
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default(font_size)

    # 绘制每个单元格
    for i in range(rows):
        for j in range(cols):
            # 计算单元格位置
            x1 = j * (cell_size + border) + border
            y1 = i * (cell_size + border) + border
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # 获取元素类型
            element = game_map[i][j]
            if (i, j) == current_pos and current_pos != (-1, -1):
                color = COLOR_MAP["P"]  # 玩家位置高亮
                element = "P"
            else:
                color = COLOR_MAP.get(element, COLOR_MAP["default"])

            # 绘制单元格背景
            draw.rectangle([x1, y1, x2, y2], fill=color)

            # 绘制元素符号
            if element != "E":
                text = element
                left, top, right, bottom = draw.textbbox((0, 0), text, font)
                text_width, text_height = right, bottom
                draw.text(
                    (x1 + (cell_size - text_width) // 2, y1 + (cell_size - text_height) // 2),
                    text,
                    fill="#000000",
                    font=font
                )

    # 保存图片
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    if save_filename is None:
        img.save(f"{output_dir}/step_{step:03d}.png")
    else:
        img.save(f"{output_dir}/{save_filename}")

def generate(seed, scale, num_step):
    while True:
        game_map, game_map_png, task = generate_core(seed, scale, num_step)
        if game_map is None:
            seed += 1
            continue
        simulate_result = simulate(game_map, task)
        if simulate_result is None:
            seed += 1
            continue
        verify_result = verify(game_map, task, simulate_result)
        if verify_result == 1:
            return game_map, game_map_png, task
        else:
            seed += 1

def generate_core(seed, scale, num_step):
    random.seed(seed)
    rows, cols = scale
    area = (rows-2) * (cols-2)
    portal_num_max = math.ceil(area * 0.05)
    jatr_num_max = math.ceil(area * 0.4) // 4
    print("area = {}, portal_num_max = {}, jatr_num_max = {}".format(area, portal_num_max, jatr_num_max))
    assert area > 1 + portal_num_max * 2 + jatr_num_max, "Too small area, too many elements"

    # Initialize map with 'E' and set borders to 'W'
    game_map = [['E' for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                game_map[i][j] = 'W'

    # Place player 'P'
    possible_positions = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            possible_positions.append((i, j))
    if not possible_positions:
        raise ValueError('No available positions to place player')
    p_pos = random.choice(possible_positions)
    possible_positions.remove(p_pos)
    game_map[p_pos[0]][p_pos[1]] = 'P'

    # Place portals
    portal_num = random.randint(1, portal_num_max)
    portal_id = 1
    for _ in range(portal_num):
        if len(possible_positions) >= 2:
            pos1 = random.choice(possible_positions)
            possible_positions.remove(pos1)
            pos2 = random.choice(possible_positions)
            possible_positions.remove(pos2)
            game_map[pos1[0]][pos1[1]] = str(portal_id)
            game_map[pos2[0]][pos2[1]] = str(portal_id)
            portal_id += 1

    # Place other elements
    elements = ['J', 'A', 'T', 'R']
    for elem in elements:
        count = random.randint(0, jatr_num_max)
        for _ in range(count):
            if possible_positions:
                pos = random.choice(possible_positions)
                possible_positions.remove(pos)
                game_map[pos[0]][pos[1]] = elem

    # Generate task
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    task = [random.choice(directions) for _ in range(num_step)]

    game_map_png = "step_init.png"
    draw_map(game_map, (-1, -1), save_filename=game_map_png, step=0)
    return game_map, game_map_png, task

def simulate(map, task):
    # Find initial player position
    start_pos = None
    rows, cols = len(map), len(map[0]) if len(map) > 0 else 0
    for i in range(rows):
        for j in range(cols):
            if map[i][j] == 'P':
                start_pos = (i, j)
                break
        if start_pos:
            break
    current_pos = start_pos
    action_idx = 0
    trapped = 0
    repeated_action = None

    outer_loop_count = 0
    while action_idx < len(task):
        outer_loop_count += 1
        if outer_loop_count > 200:
            print("Infinite outer loop detected")
            return None
        draw_map(map, current_pos, save_filename=None, step=action_idx)
        if trapped > 0:
            trapped -= 1
            action_idx += 1
            continue
        # Determine current direction
        if repeated_action is not None:
            current_action = repeated_action
            repeated_action = None
        else:
            current_action = task[action_idx]
            action_idx += 1
        dx, dy = 0, 0
        if current_action == 'UP':
            dx = -1
        elif current_action == 'DOWN':
            dx = 1
        elif current_action == 'LEFT':
            dy = -1
        elif current_action == 'RIGHT':
            dy = 1
        # Calculate new position
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        element = map[new_x][new_y]
        # Process elements
        inner_loop_count = 0
        while True:
            inner_loop_count += 1
            if inner_loop_count > 200:
                print("Infinite inner loop detected")
                return None
            if element == 'W':
                new_x, new_y = current_pos
                break
            if element.isdigit():
                portals = None
                for i in range(rows):
                    for j in range(cols):
                        if map[i][j] == element and (i, j) != (new_x, new_y):
                            portals = i, j
                if portals:
                    new_x, new_y = portals
                break
            elif element == 'J':
                nx = new_x + dx*2
                ny = new_y + dy*2
                if 0 <= nx < rows and 0 <= ny < cols and map[nx][ny] != 'W':
                    new_x, new_y = nx, ny
                    element = map[new_x][new_y]
                else:
                    element = 'E' # set J to E in this turn
                    break
            elif element == 'A':
                dx, dy = -dx, -dy
                nx = current_pos[0] + dx
                ny = current_pos[1] + dy
                if 0 <= nx < rows and 0 <= ny < cols and map[nx][ny] != 'W':
                    new_x, new_y = nx, ny
                    element = map[new_x][new_y]
                else:
                    new_x, new_y = current_pos
                    element = 'E' # set ele to E in this turn
                    break
            elif element == 'T':
                trapped = 1
                break
            elif element == 'R':
                repeated_action = current_action
                break
            else:
                break
        current_pos = (new_x, new_y)

    draw_map(map, current_pos, save_filename=None, step=action_idx)
    return current_pos

def verify(map, task, pred_pos):
    correct_pos = simulate(map, task)
    return 1 if correct_pos == pred_pos else 0

# 测试示例
if __name__ == "__main__":
    # 示例地图和玩家位置
    test_map = [
        ["W", "W", "W", "W"],
        ["W", "W", "T", "W"],
        ["W", "P", "A", "W"],
        ["W", "R", "1", "W"],
        ["W", "J", "1", "W"],
        ["W", "W", "W", "W"]
    ]
    current_pos = (-1, -1)

    # 生成图片
    draw_map(test_map, current_pos, 'step_init.png', step=0)

    # 生成地图和任务
    seed = time.time()
    scale = (10, 10)
    num_step = 10
    game_map, game_map_png, task = generate(seed, scale, num_step)
    print([f"{idx}: {item}" for idx, item in enumerate(task)])
    print(f"222{simulate(game_map, task)}")
    verify_result = verify(game_map, task, simulate(game_map, task))
    print(verify_result)
    verify_result = verify(game_map, task, (-1, -1))
    print(verify_result)
