import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from game_logic import generate, move_ball, is_solved, is_stuck, CAPACITY

def print_state_text(state):
    tube_labels = [chr(65 + i) for i in range(len(state))]
    print("Note: tubes are [bottom, ..., top], rightmost = top ball.")
    for i, tube in enumerate(state):
        print(f"{tube_labels[i]}: {tube}")

def parse_move(cmd):
    clean = cmd.replace(" ", "").upper()
    if len(clean) != 2:
        return None, None
    return clean[0], clean[1]

def render_state_to_image(state, move_count, single_image_mode=False):
    """
    Render the board to a PNG image using Pillow.
    If single_image_mode=True, overwrite cache/board.png each update.
    If False, store a history in cache/history/board_{move_count}.png.
    """
    margin = 50
    spacing = 100
    tube_width = 30
    tube_height = 120
    circle_r = 12

    total_tubes = len(state)
    width = margin * 2 + (total_tubes - 1) * spacing + tube_width
    height = 300

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, tube in enumerate(state):
        x_left = margin + i * spacing
        y_top = (height - tube_height) // 2
        x_right = x_left + tube_width
        y_bottom = y_top + tube_height
        draw.rectangle([x_left, y_top, x_right, y_bottom], outline=(0,0,0), width=2)
        label = chr(65 + i)
        draw.text((x_left, y_top - 15), label, fill=(0,0,0), font=font)

        slot_h = tube_height / CAPACITY
        for slot_i in range(CAPACITY):
            ball_color_id = tube[slot_i]
            if ball_color_id != 0:
                color_map = {
                    1: (255,0,0),
                    2: (0,255,0),
                    3: (0,0,255),
                    4: (255,255,0),
                    5: (255,0,255),
                    6: (0,255,255),
                }
                fill_color = color_map.get(ball_color_id, (128,128,128))
                cx = x_left + tube_width / 2
                cy = y_bottom - slot_h * (slot_i + 0.5)
                draw.ellipse([(cx - circle_r, cy - circle_r), (cx + circle_r, cy + circle_r)], fill=fill_color, outline=(0,0,0))

    os.makedirs("cache", exist_ok=True)
    if single_image_mode:
        file_path = "cache/board.png"
    else:
        history_dir = "cache/history"
        if move_count == 0:
            if os.path.exists(history_dir):
                shutil.rmtree(history_dir)
            os.makedirs(history_dir)
        file_path = f"{history_dir}/board_{move_count}.png"

    img.save(file_path)
    print(f"Saved board image to {file_path}")

def main_image_ui(config):
    seed = config.get("seed", 42)
    level = config.get("level", 1)
    history_mode = config.get("history_mode", False)

    state = generate(level, seed)
    print("=== Ball Sort Puzzle (Text+Image UI) ===")
    print_state_text(state)

    move_count = 0
    render_state_to_image(state, move_count, single_image_mode=not history_mode)

    while True:
        if is_solved(state):
            print("Congratulations, you solved the puzzle!")
            break
        if is_stuck(state):
            print("No more legal moves available. You lost!")
            break

        cmd = input("Enter a move (e.g. A D or AD) or 'q' to quit: ")
        if cmd.lower() == 'q':
            print("Exiting.")
            break

        s, d = parse_move(cmd)
        if s is None or d is None:
            print("Invalid command. Try again.")
            continue

        if not move_ball(state, s, d):
            print("Invalid move!")
        else:
            move_count += 1
            print_state_text(state)
            render_state_to_image(state, move_count, single_image_mode=not history_mode)
