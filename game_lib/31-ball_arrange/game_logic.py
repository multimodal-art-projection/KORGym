# game_logic.py
import random
import copy

CAPACITY = 4  # Each tube has 4 slots

def num_colors_for_level(level):
    """
    Example mapping:
      level=1 -> 3 colors
      level=2 -> 4 colors
      level=3 -> 5 colors
      etc.
    """
    return level + 2

def generate(level, seed=None):
    """
    Generate the puzzle based on level:
      - num_colors = level + 2
      - total tubes = num_colors + 2
      - The first 'num_colors' tubes are filled with 4 balls each
      - The last 2 tubes are empty
    """
    if seed is not None:
        random.seed(seed)
    
    n = num_colors_for_level(level)   # e.g. level=1 => n=3 colors
    total_tubes = n + 2              # e.g. level=1 => 5 tubes
    
    # Prepare 4*n balls
    balls = []
    for color in range(1, n + 1):
        balls.extend([color] * CAPACITY)
    random.shuffle(balls)
    
    # Fill the first n tubes
    state = []
    idx = 0
    for _ in range(n):
        tube = balls[idx : idx + CAPACITY]
        idx += CAPACITY
        state.append(tube)
    
    # Add 2 empty tubes
    for _ in range(2):
        state.append([0] * CAPACITY)
    
    return state

def move_ball(state, src, dst):
    """
    Move the top ball from tube 'src' to tube 'dst' if the move is legal.
    Rightmost element is the top (index 3).
    """
    label_map = {chr(65 + i): i for i in range(len(state))}
    if isinstance(src, str):
        src = label_map.get(src.upper(), -1)
    if isinstance(dst, str):
        dst = label_map.get(dst.upper(), -1)
    if not (0 <= src < len(state) and 0 <= dst < len(state)):
        return False
    
    src_tube = state[src]
    dst_tube = state[dst]
    
    # Find top ball in src
    src_top = -1
    for i in range(CAPACITY - 1, -1, -1):
        if src_tube[i] != 0:
            src_top = i
            break
    if src_top == -1:
        return False  # src empty
    
    ball = src_tube[src_top]
    
    # Count how many balls in dst
    dst_count = sum(1 for x in dst_tube if x != 0)
    if dst_count >= CAPACITY:
        return False  # dst full
    
    # If dst not empty, top color must match
    if dst_count > 0:
        dst_top = -1
        for i in range(CAPACITY - 1, -1, -1):
            if dst_tube[i] != 0:
                dst_top = i
                break
        if dst_top == -1:
            return False
        if dst_tube[dst_top] != ball:
            return False
        place_index = dst_top + 1
    else:
        place_index = 0
    
    # Execute move
    src_tube[src_top] = 0
    dst_tube[place_index] = ball
    return True

def is_solved(state):
    """
    Puzzle is solved if every NON-empty tube is fully occupied by
    4 identical balls. (Ignore tubes that are completely empty.)
    """
    for tube in state:
        if all(x == 0 for x in tube):
            # completely empty => ignore
            continue
        # must be full and identical
        if any(x == 0 for x in tube):
            return False
        if len(set(tube)) != 1:
            return False
    return True

def is_stuck(state):
    """
    Return True if no legal moves remain.
    """
    for i, tube in enumerate(state):
        # find top ball in tube i
        top_idx = -1
        for j in range(CAPACITY - 1, -1, -1):
            if tube[j] != 0:
                top_idx = j
                break
        if top_idx == -1:
            continue  # empty
        ball = tube[top_idx]
        
        # check if we can move to another tube
        for k, dst_tube in enumerate(state):
            if k == i:
                continue
            dst_count = sum(1 for x in dst_tube if x != 0)
            if dst_count < CAPACITY:
                if dst_count == 0:
                    return False
                # must match top color
                dst_top_idx = -1
                for z in range(CAPACITY - 1, -1, -1):
                    if dst_tube[z] != 0:
                        dst_top_idx = z
                        break
                if dst_top_idx != -1 and dst_tube[dst_top_idx] == ball:
                    return False
    return True

def verify(state, actions):
    """
    Apply actions to a copy of state and check if puzzle is solved.
    """
    import copy
    temp = copy.deepcopy(state)
    for (s, d) in actions:
        if not move_ball(temp, s, d):
            return 0
    return 1 if is_solved(temp) else 0
