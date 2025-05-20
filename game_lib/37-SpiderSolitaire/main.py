from spider_solitaire import SpiderSolitaire
import copy
import random

def generate(seed):
    """
    Generate initial game state using provided seed
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        board: List representing the initial game state
    """
    # Create a new game with the given seed
    game = SpiderSolitaire()
    board = game.setup_game(seed)
    
    # Create a deep copy to avoid modifying the internal state
    return copy.deepcopy(board)

def verify(board, score, epoch, action):
    """
    Verify and apply an action to the current game state
    
    Args:
        board: List representing the current game state
        score: Current score
        epoch: Current game epoch/round
        action: String representing the action to take
        
    Returns:
        board: Updated game state after action
        score: Updated score
        epoch: Updated epoch
    """
    # We need to recreate the game state
    # In a real implementation, we would maintain the full game state across calls
    seed = getattr(verify, 'seed', None)
    if seed is None:
        seed = random.randint(1, 100000)
        verify.seed = seed
        verify.game = SpiderSolitaire()
        verify.game.setup_game(seed)
    
    game = verify.game
    
    # Process the action
    if action.strip().lower() == "hit":
        # Deal cards
        success = game.deal_cards()
        if not success:
            print("Cannot deal cards. Make sure each column has at least one card and there are enough cards in the deck.")
    else:
        # Parse move action in format (A,4,B) - Move cards from column A starting at index 4 to column B
        try:
            # Remove parentheses and split by comma
            parts = action.strip("()").split(",")
            if len(parts) != 3:
                print("Invalid move format. Expected format: (A,4,B)")
                return board, score, epoch
                
            # Convert column names to indices (A=0, B=1, etc.)
            from_col = ord(parts[0].strip().upper()) - ord('A')
            start_idx = int(parts[1].strip())
            to_col = ord(parts[2].strip().upper()) - ord('A')
            
            # Apply the move
            success = game.move_cards(from_col, start_idx, to_col)
            if not success:
                print("Invalid move. Please check your move and try again.")
        except Exception as e:
            print(f"Error processing move: {e}")
            return board, score, epoch
    
    # Update score and epoch
    new_score = game.completed_sets
    new_epoch = epoch + 1
    
    # Return updated game state with visibility applied
    return game.get_visible_board(), new_score, new_epoch

def display_board(board):
    """
    Display the current game board in a readable format
    
    Args:
        board: The current game board
    """
    column_labels = "ABCDEFGHIJ"
    
    # Find maximum column length for formatting
    max_length = max(len(col) for col in board)
    
    # Print column headers
    print("  " + " ".join(column_labels[:len(board)]))
    print("  " + "-" * (2 * len(board) - 1))
    
    # Print rows from top to bottom
    for i in range(max_length):
        row = []
        for j, column in enumerate(board):
            if i < len(column):
                card = column[i]
                if card[0] == 'unknown':
                    row.append("XX")
                else:
                    row.append(f"{card[1]}{card[0][0]}")  # Format as rank + first letter of suit
            else:
                row.append("  ")
        print(f"{i} {' '.join(row)}")

def main():
    """Main function to run the game interactively"""
    # Initialize game
    seed = int(input("Enter a seed for the game (or press Enter for random seed): ") or str(random.randint(1, 100000)))
    
    # Create a full game instance
    game = SpiderSolitaire()
    game.setup_game(seed)
    
    # For verify function
    verify.seed = seed
    verify.game = game
    
    # Start the game
    board = game.get_visible_board()
    score = 0
    epoch = 0
    
    # Game loop
    while True:
        print("\n" + "="*50)
        print(f"Epoch: {epoch}, Score: {game.score}, Steps: {game.steps}, Completed Sets: {game.completed_sets}")
        print(f"Remaining cards in deck: {len(game.deck)}")
        display_board(board)
        
        action = input("\nEnter your move ((A,4,B) - Move cards from column A starting at index 4 to column B\n'Hit' to deal cards, 'undo' to undo last move, 'q' to quit): ")
        action = action.lower()
        
        if action == 'q':
            break
        elif action == 'undo':
            if game.undo():
                print("Move undone.")
            else:
                print("No moves to undo.")
            board = game.get_visible_board()
            continue
        
        # Process action via verify function to ensure compatibility
        new_board, new_score, new_epoch = verify(board, score, epoch, action)
        board = new_board
        score = new_score
        epoch = new_epoch
        
        # Check for win condition
        if game.completed_sets == 8:  # 8 completed sets = all cards
            print("\nCongratulations! You've won the game!")
            break

if __name__ == "__main__":
    main()