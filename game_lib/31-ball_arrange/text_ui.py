# text_ui.py
from game_logic import generate, move_ball, is_solved, is_stuck

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

def main_text(config):
    """
    We no longer read num_colors from config. We only read:
      - config["seed"]
      - config["level"]
    """
    seed = config.get("seed", 42)
    level = config.get("level", 1)
    
    state = generate(level, seed)
    print("=== Ball Sort Puzzle (Pure Text UI) ===")
    print_state_text(state)
    
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
            print("Invalid command.")
            continue
        
        if not move_ball(state, s, d):
            print("Illegal move!")
        else:
            print_state_text(state)
