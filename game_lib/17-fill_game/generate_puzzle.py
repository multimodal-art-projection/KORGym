from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import random
import math

# 设置字体路径（如 Arial Bold）
FONT_PATH = "arialbd.ttf"  

# 目标统一尺寸
TARGET_SIZE = (200, 200)  # 标准尺寸
OPTION_SIZE = (70, 70)    # 选项块大小

def generate(seed: int,  nums: int = 6):
    """
    生成“填图游戏”题目示例，确保所有图片 resize 到 200x200，
    并提供题目文字 + 两行排列的选项（Pillow 10.0+ 兼容）。
    """
    random.seed(seed)
    base_directory = "pictures"
    sub_folder = os.listdir(base_directory)
    category=random.sample(sub_folder,1)
    img_dir = base_directory+'/'+category[0]
    # 1. 读取并统一 resize 图片
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not all_imgs:
        raise FileNotFoundError("图片目录为空，请放置至少一张图片。")

    # 选取主图，并 resize
    main_img_name = random.choice(all_imgs)
    main_img_path = os.path.join(img_dir, main_img_name)
    original_img = Image.open(main_img_path).convert("RGBA")
    original_img = original_img.resize(TARGET_SIZE, Image.LANCZOS)
    width, height = TARGET_SIZE  # 200x200

    # 2. 挖去拼图块（右侧中部，改为方形）
    shape_mask = Image.new("L", TARGET_SIZE, 0)
    draw_mask = ImageDraw.Draw(shape_mask)

    shape_w, shape_h = OPTION_SIZE
    offset_x = width - shape_w - 20
    offset_y = (height - shape_h) // 2

    # 使用矩形代替原来的斜角多边形
    draw_mask.rectangle([offset_x, offset_y, offset_x + shape_w, offset_y + shape_h], fill=255)

    # 3. 裁剪正确拼图块
    correct_piece = Image.new("RGBA", TARGET_SIZE, (0, 0, 0, 0))
    correct_piece.paste(original_img, mask=shape_mask)
    correct_piece_cropped = correct_piece.crop((offset_x, offset_y, offset_x + shape_w, offset_y + shape_h))
    correct_piece_cropped = correct_piece_cropped.resize(OPTION_SIZE, Image.LANCZOS)

    # 4. 生成带“缺口”的拼图
    puzzle_img = original_img.copy()
    inv_mask = Image.new("L", TARGET_SIZE, 255)
    inv_draw = ImageDraw.Draw(inv_mask)
    inv_draw.rectangle([offset_x, offset_y, offset_x + shape_w, offset_y + shape_h], fill=0)

    puzzle_array = puzzle_img.load()
    inv_array = inv_mask.load()
    for y in range(height):
        for x in range(width):
            if inv_array[x, y] == 0:
                puzzle_array[x, y] = (255, 255, 255, 0)  # 挖空

    # 5. 生成干扰块
    distractor_pieces = []
    while len(distractor_pieces) < (nums - 1):
        distract_img_name = random.choice(all_imgs)
        distract_img_path = os.path.join(img_dir, distract_img_name)
        d_img = Image.open(distract_img_path).convert("RGBA").resize(TARGET_SIZE, Image.LANCZOS)

        rand_x = random.randint(0, width - shape_w)
        rand_y = random.randint(0, height - shape_h)
        piece = d_img.crop((rand_x, rand_y, rand_x + shape_w, rand_y + shape_h))
        piece = piece.resize(OPTION_SIZE, Image.LANCZOS)

        if random.random() < 0.3:
            piece = piece.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            piece = piece.transpose(Image.ROTATE_90)

        distractor_pieces.append(piece)

    while len(distractor_pieces) < (nums - 1):
        distractor_pieces.append(correct_piece_cropped.copy())

    # 6. 生成最终拼图（随机排列选项）
    all_options = distractor_pieces + [correct_piece_cropped]
    random.shuffle(all_options)
    correct_idx = all_options.index(correct_piece_cropped)
    answer_options = [chr(ord('A') + i) for i in range(nums)]
    correct_answer = answer_options[correct_idx]

    # ---------------------
    #   排版到最终图
    # ---------------------
    title_text = "1. Please choose the shape that best fits the missing piece"

    try:
        font_title = ImageFont.truetype(FONT_PATH, 40)
    except IOError:
        font_title = ImageFont.load_default()

    left, top, right, bottom = font_title.getbbox(title_text)
    title_w = right - left
    title_h = bottom - top

    margin = 20
    columns = 4
    rows = math.ceil(nums / columns)

    options_total_width = (OPTION_SIZE[0] + margin) * columns - margin
    options_total_height = (OPTION_SIZE[1] + margin) * rows - margin

    final_width = max(width, options_total_width, title_w) + margin * 2
    final_height = margin + title_h + margin + height + margin + options_total_height + margin

    final_image = Image.new("RGBA", (final_width, final_height), (255, 255, 255, 255))
    draw_final = ImageDraw.Draw(final_image)

    # 1) 绘制标题(居中)
    title_x = (final_width - title_w) // 2
    title_y = margin
    draw_final.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font_title)

    # 2) 贴拼图
    puzzle_x = (final_width - width) // 2  
    puzzle_y = title_y + title_h + margin
    final_image.paste(puzzle_img, (puzzle_x, puzzle_y), puzzle_img)

    # 3) 两行排列选项
    try:
        font_option = ImageFont.truetype(FONT_PATH, 24)
    except IOError:
        font_option = ImageFont.load_default()

    options_start_x = (final_width - options_total_width) // 2
    options_start_y = puzzle_y + height + margin

    current_x = options_start_x
    current_y = options_start_y

    for i, piece_img in enumerate(all_options):
        final_image.paste(piece_img, (current_x, current_y), piece_img)
        draw_final.text((current_x + 5, current_y + 5),
                        answer_options[i], fill=(255, 0, 0), font=font_option)
        current_x += OPTION_SIZE[0] + margin
        if (i + 1) % columns == 0:
            current_x = options_start_x
            current_y += OPTION_SIZE[1] + margin

    return final_image, correct_answer


if __name__ == "__main__":
    seed_value = 422112
    
    nums_options = 16

    puzzle_img, correct_ans = generate(seed_value, nums_options)
    puzzle_img.save("puzzle_demo.png")

    print("Correct answer:", correct_ans)
    print("Puzzle image saved as puzzle_demo.png")
