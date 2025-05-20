import random
import string
import os
import collections

class CryptoWord:
    def __init__(self, sentences_file="sentences.txt"):
        """
        Initialize the CryptoWord game.
        
        Args:
            sentences_file (str): Path to the file containing sentences.
        """
        self.sentences_file = sentences_file
        self.sentences = self._load_sentences()
        self.emojis = [
            "😀", "😂", "😍", "🤔", "😎", "🥳", "😴", "🤩", "🥺", "😱",
            "🙄", "😇", "🤗", "🤫", "🤭", "🤥", "🤮", "🤧", "🥶", "🥵",
            "🤠", "🥴", "🤑", "🤓", "🧐", "😈", "👻", "👽", "🤖", "💩",
            "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯",
            "🦁", "🐮", "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🦆", "🦉"
        ]
        
    def _load_sentences(self):
        """Load sentences from file."""
        if not os.path.exists(self.sentences_file):
            # Fallback sentences if file doesn't exist
            return [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "Sphinx of black quartz, judge my vow."
            ]
        # dataset from https://github.com/facebookresearch/asset/blob/main/dataset/asset.valid.orig
        with open(self.sentences_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def generate(self, seed=None, encoding_table=None, replacement_ratio=0.5):
        """
        Generate an encoded sentence.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
            encoding_table (dict, optional): Custom encoding table.
            replacement_ratio (float): Ratio of letters to replace with emojis (0.0-1.0).
            
        Returns:
            dict: Contains the encoded_sentence, answer (most frequent emoji mapping),
                 and complete_mapping (full emoji-to-letter mapping for verification)
        """
        if seed is not None:
            random.seed(seed)
        
        # Select a random sentence
        original_sentence = random.choice(self.sentences)
        original_sentence_lower = original_sentence.lower()
        
        # Count letter frequencies in the sentence
        letter_counts = collections.Counter([c for c in original_sentence_lower if c in string.ascii_lowercase])
        unique_letters = list(letter_counts.keys())
        
        # Determine how many letters to replace
        num_unique_letters = len(unique_letters)
        num_to_replace = max(1, min(int(num_unique_letters * replacement_ratio), num_unique_letters))
        
        # Randomly select letters to replace
        random.shuffle(unique_letters)
        letters_to_replace = unique_letters[:num_to_replace]
        
        # Create a shuffled copy of emojis to use
        available_emojis = self.emojis.copy()
        random.shuffle(available_emojis)
        
        # Create encoding table
        if encoding_table is None:
            encoding_table = {}
            for i, letter in enumerate(letters_to_replace):
                if i < len(available_emojis):
                    encoding_table[letter] = available_emojis[i]
        
        # Create the reverse mapping for answer generation
        reverse_mapping = {v: k for k, v in encoding_table.items()}
        
        # Encode the sentence
        encoded_sentence = ""
        for char in original_sentence_lower:
            if char in encoding_table:
                encoded_sentence += encoding_table[char]
            else:
                encoded_sentence += char
        
        # Find the most frequent emoji
        emoji_counts = {}
        for char in encoded_sentence:
            if char in reverse_mapping:
                emoji_counts[char] = emoji_counts.get(char, 0) + 1
        
        most_frequent_emoji = None
        if emoji_counts:
            most_frequent_emoji = max(emoji_counts.items(), key=lambda x: x[1])[0]
        
        # Format the hint answer
        hint_answer = ""
        if most_frequent_emoji:
            hint_answer = f"{most_frequent_emoji}={reverse_mapping[most_frequent_emoji]}"
        
        return {
            "encoded_sentence": encoded_sentence,
            "hint": hint_answer,
            "answer": reverse_mapping  # Return the full mapping for verification
        }
    
    def verify(self, answer, generated_answer):
        """
        Verify the player's answer.
        
        Args:
            answer (dict): Complete emoji-to-letter mapping from generate method.
            generated_answer (str): Player's answer in the format "emoji=letter,emoji=letter,...".
            
        Returns:
            dict: Feedback on which emojis were correctly identified.
        """
        feedback = {}
        
        # Get the correct emoji-to-letter mapping from the parameter
        correct_mapping = answer
        
        # Parse the generated answer
        # Expected format: emoji=letter, emoji=letter, ...
        for pair in generated_answer.split(','):
            pair = pair.strip()
            if '=' in pair:
                emoji_guess, letter_guess = pair.split('=')
                emoji_guess = emoji_guess.strip()
                letter_guess = letter_guess.strip().lower()
                
                # Check if the emoji exists in our mapping and if the guess is correct
                if emoji_guess in correct_mapping:
                    feedback[emoji_guess] = letter_guess == correct_mapping[emoji_guess]
                else:
                    feedback[emoji_guess] = False
        
        return {"feedback": feedback}