from spider_solitaire import SpiderSolitaire
import copy
import random
import tkinter as tk
from tkinter import messagebox, simpledialog
import os
from PIL import Image, ImageTk

class SpiderSolitaireUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spider Solitaire")
        self.root.geometry("1000x700")
        self.root.configure(bg="#076324")  # Green background like a card table
        
        # Game state
        self.game = SpiderSolitaire()
        self.seed = None
        self.board = None
        self.selected_cards = None
        self.selected_column = None
        self.selected_index = None
        
        # Card dimensions
        self.card_width = 80
        self.card_height = 120
        self.card_overlap = 30  # Vertical overlap of cards in a column
        
        # Load card images
        self.load_card_images()
        
        # Create UI elements
        self.create_ui()
        
        # Start a new game
        self.new_game()
    
    def load_card_images(self):
        """Load card images or create colored rectangles if images not available"""
        self.card_images = {}
        self.card_back = None
        
        # Try to load card images if available
        try:
            # This assumes you have card images in a 'cards' directory
            # If not, we'll fall back to colored rectangles
            card_dir = "cards"
            if os.path.exists(card_dir):
                # Load card back
                self.card_back = ImageTk.PhotoImage(
                    Image.open(os.path.join(card_dir, "back.png")).resize((self.card_width, self.card_height))
                )
                
                # Load suit images
                suits = ["hearts", "diamonds", "clubs", "spades"]
                ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
                
                for suit in suits:
                    for rank in ranks:
                        img = Image.open(os.path.join(card_dir, f"{rank}_{suit}.png"))
                        img = img.resize((self.card_width, self.card_height))
                        self.card_images[(suit, rank)] = ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Could not load card images: {e}")
            print("Using colored rectangles instead")
            # We'll create colored rectangles in the draw_card method
    
    def create_ui(self):
        """Create the UI elements"""
        # Top frame for game info and controls
        self.top_frame = tk.Frame(self.root, bg="#076324")
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Game info labels
        self.info_frame = tk.Frame(self.top_frame, bg="#076324")
        self.info_frame.pack(side=tk.LEFT)
        
        self.score_label = tk.Label(self.info_frame, text="Score: 0", font=("Arial", 12), bg="#076324", fg="white")
        self.score_label.pack(side=tk.LEFT, padx=5)
        
        self.steps_label = tk.Label(self.info_frame, text="Steps: 0", font=("Arial", 12), bg="#076324", fg="white")
        self.steps_label.pack(side=tk.LEFT, padx=5)
        
        self.completed_sets_label = tk.Label(self.info_frame, text="Completed Sets: 0", font=("Arial", 12), bg="#076324", fg="white")
        self.completed_sets_label.pack(side=tk.LEFT, padx=5)
        
        self.deck_label = tk.Label(self.info_frame, text="Deck: 0", font=("Arial", 12), bg="#076324", fg="white")
        self.deck_label.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        self.control_frame = tk.Frame(self.top_frame, bg="#076324")
        self.control_frame.pack(side=tk.RIGHT)
        
        self.new_game_btn = tk.Button(self.control_frame, text="New Game", command=self.new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.hit_btn = tk.Button(self.control_frame, text="Hit", command=self.hit)
        self.hit_btn.pack(side=tk.LEFT, padx=5)
        
        self.undo_btn = tk.Button(self.control_frame, text="Undo", command=self.undo)
        self.undo_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for the game board
        self.canvas = tk.Canvas(self.root, bg="#076324", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
    
    def new_game(self):
        """Start a new game"""
        # Ask for a seed
        seed_input = simpledialog.askstring("New Game", "Enter a seed for the game (leave empty for random, -1 for cheat mode):")
        if seed_input:
            try:
                self.seed = int(seed_input)
            except ValueError:
                self.seed = random.randint(1, 100000)
        else:
            self.seed = random.randint(1, 100000)
        
        # Initialize game
        self.game = SpiderSolitaire()
        
        # Check if cheat mode is activated
        
        self.game.setup_game(self.seed)
        if self.seed == -1:
            self.game.setup_cheat_mode()
        self.board = self.game.get_visible_board()
        
        # Reset selection
        self.selected_cards = None
        self.selected_column = None
        self.selected_index = None
        
        # Update UI
        self.update_ui()
    
    def hit(self):
        """Deal cards from the deck"""
        success = self.game.deal_cards()
        if not success:
            messagebox.showinfo("Cannot Deal", "Cannot deal cards. Make sure each column has at least one card and there are enough cards in the deck.")
        else:
            self.board = self.game.get_visible_board()
            self.update_ui()
            
            # Check for win after dealing
            if self.game.completed_sets == 8:
                messagebox.showinfo("Congratulations", "You've won the game!")
    
    def undo(self):
        """Undo the last move"""
        if self.game.undo():
            self.board = self.game.get_visible_board()
            self.selected_cards = None
            self.selected_column = None
            self.selected_index = None
            self.update_ui()
        else:
            messagebox.showinfo("Cannot Undo", "No moves to undo.")
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        # Calculate column width
        column_width = self.canvas.winfo_width() / len(self.board)
        
        # Determine which column was clicked
        col_idx = int(event.x / column_width)
        if col_idx >= len(self.board):
            return
        
        # Determine the card index in the column
        column = self.board[col_idx]
        y_offset = 20  # Starting y position
        card_idx = -1
        
        # If the column is empty, set card_idx to 0 (indicating we want to move to an empty column)
        if len(column) == 0:
            card_idx = 0
        else:
            for i in range(len(column)):
                card_top = y_offset + i * self.card_overlap
                card_bottom = card_top + self.card_height
                
                # If this is the last card, it extends fully
                if i == len(column) - 1:
                    card_bottom = card_top + self.card_height
                
                if card_top <= event.y <= card_bottom:
                    # Check if we're in the overlapping region of a card that's not the last one
                    if i < len(column) - 1 and event.y <= card_top + self.card_overlap:
                        card_idx = i
                        break
                    else:
                        card_idx = i
        
        # If no card was clicked and the column is not empty, return
        if card_idx == -1 and len(column) > 0:
            return
        
        # If a card is already selected, try to move it to the clicked column
        if self.selected_cards is not None:
            success = self.game.move_cards(self.selected_column, self.selected_index, col_idx)
            
            if success:
                self.board = self.game.get_visible_board()
                
                # Check for win after move
                if self.game.completed_sets == 8:
                    self.update_ui()
                    messagebox.showinfo("Congratulations", "You've won the game!")
            
            # Reset selection
            self.selected_cards = None
            self.selected_column = None
            self.selected_index = None
            self.update_ui()
            return
        
        # If no card is currently selected and the column is empty, return
        if len(column) == 0:
            return
        
        # Check if the card is face down
        if column[card_idx][0] == 'unknown':
            return  # Can't select face-down cards
        
        # If no card is currently selected, select this one
        # Check if we can select this card (and cards below it)
        # Cards must be in descending order and same suit to be moved together
        can_select = True
        for i in range(card_idx, len(column) - 1):
            current_card = column[i]
            next_card = column[i + 1]
            
            # Skip check for unknown cards
            if current_card[0] == 'unknown' or next_card[0] == 'unknown':
                can_select = False
                break
            
            # Check if cards form a valid sequence
            current_rank = self.get_card_value(current_card[1])
            next_rank = self.get_card_value(next_card[1])
            
            if current_rank != next_rank + 1 or current_card[0] != next_card[0]:
                can_select = False
                break
        
        # Always allow selecting at least the clicked card
        self.selected_column = col_idx
        self.selected_index = card_idx
        self.selected_cards = column[card_idx:]
        self.update_ui()
    
    def get_card_value(self, rank):
        """Convert card rank to numeric value"""
        if rank == 'A':
            return 1
        elif rank in ['J', 'Q', 'K']:
            return {'J': 11, 'Q': 12, 'K': 13}[rank]
        else:
            return int(rank)
    
    def update_ui(self):
        """Update the UI to reflect the current game state"""
        # Update info labels
        self.score_label.config(text=f"Score: {self.game.score}")
        self.steps_label.config(text=f"Steps: {self.game.steps}")
        self.completed_sets_label.config(text=f"Completed Sets: {self.game.completed_sets}")
        self.deck_label.config(text=f"Deck: {len(self.game.deck)}")
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw column headers
        column_width = self.canvas.winfo_width() / len(self.board)
        for i in range(len(self.board)):
            x = i * column_width + column_width / 2
            self.canvas.create_text(x, 10, text=chr(65 + i), fill="white", font=("Arial", 12))
        
        # Draw cards
        for col_idx, column in enumerate(self.board):
            x = col_idx * column_width + (column_width - self.card_width) / 2
            y_offset = 20  # Starting y position
            
            for card_idx, card in enumerate(column):
                y = y_offset + card_idx * self.card_overlap
                
                # Highlight selected cards
                highlight = (self.selected_column == col_idx and card_idx >= self.selected_index)
                
                self.draw_card(x, y, card, highlight)
        
        # Draw deck if there are cards left
        if len(self.game.deck) > 0:
            deck_x = self.canvas.winfo_width() - self.card_width - 20
            deck_y = 20
            self.draw_deck(deck_x, deck_y)
    
    def draw_card(self, x, y, card, highlight=False):
        """Draw a card at the specified position"""
        # Determine card color and text
        if card[0] == 'unknown':
            # Draw card back
            if self.card_back:
                self.canvas.create_image(x, y, anchor=tk.NW, image=self.card_back)
            else:
                self.canvas.create_rectangle(x, y, x + self.card_width, y + self.card_height, 
                                            fill="#000080", outline="white")
                self.canvas.create_text(x + self.card_width/2, y + self.card_height/2, 
                                        text="", fill="white")
        else:
            # Draw card face
            suit, rank = card
            
            # Determine card color based on suit - using four distinct colors
            suit_colors = {
                "hearts": "#FF0000",    # Bright Red
                "diamonds": "#FF6600",  # Orange-Red
                "clubs": "#006600",     # Dark Green
                "spades": "#000080"     # Navy Blue
            }
            suit_symbols = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
            suit_symbols = {v: k for k, v in suit_symbols.items()}
            card_color = suit_colors.get(suit_symbols[suit], "black")  # Default to black if suit not found
            
            # Check if we have an image for this card
            card_key = (suit, rank)
            if card_key in self.card_images:
                self.canvas.create_image(x, y, anchor=tk.NW, image=self.card_images[card_key])
            else:
                # Draw a rectangle with text
                fill_color = "#FFFFFF"
                if highlight:
                    outline_color = "#FFFF00"  # Yellow highlight
                    outline_width = 3
                else:
                    outline_color = "black"
                    outline_width = 1
                
                self.canvas.create_rectangle(x, y, x + self.card_width, y + self.card_height, 
                                            fill=fill_color, outline=outline_color, width=outline_width)
                
                # Draw suit symbol
                suit_symbols = {"hearts": "♥", "diamonds": "♦", "clubs": "♣", "spades": "♠"}
                # Make sure we're using the original suit name, not the symbol
                if suit in suit_symbols:
                    suit_symbol = suit_symbols[suit]
                else:
                    # If it's already a symbol, use it directly
                    suit_symbol = suit
                
                # Draw rank and suit side by side
                # Top left corner
                self.canvas.create_text(x + 8, y + 15, text=rank, fill=card_color, anchor=tk.NW, font=("Arial", 12, "bold"))
                self.canvas.create_text(x + 22, y + 15, text=suit_symbol, fill=card_color, anchor=tk.NW, font=("Arial", 12))
                
                # Bottom right corner
                self.canvas.create_text(x + self.card_width - 22, y + self.card_height - 15, 
                                        text=suit_symbol, fill=card_color, anchor=tk.SE, font=("Arial", 12))
                self.canvas.create_text(x + self.card_width - 8, y + self.card_height - 15, 
                                        text=rank, fill=card_color, anchor=tk.SE, font=("Arial", 12, "bold"))
                
                # Center symbol for better visibility
                self.canvas.create_text(x + self.card_width/2, y + self.card_height/2, 
                                        text=suit_symbol, fill=card_color, font=("Arial", 24))
            
    def draw_deck(self, x, y):
        """Draw the deck of remaining cards"""
        if self.card_back:
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.card_back)
        else:
            self.canvas.create_rectangle(x, y, x + self.card_width, y + self.card_height, 
                                        fill="#000080", outline="white")
            self.canvas.create_text(x + self.card_width/2, y + self.card_height/2, 
                                    text=f"{len(self.game.deck)}", fill="white", font=("Arial", 14, "bold"))
            
            # Add a label
            self.canvas.create_text(x + self.card_width/2, y + self.card_height + 15, 
                                    text="Deck (Click 'Hit')", fill="white", font=("Arial", 10))

# Main function to run the game
def main():
    root = tk.Tk()
    app = SpiderSolitaireUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()