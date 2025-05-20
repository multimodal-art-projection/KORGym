date_calculate_prompt='''
You are a good game problem-solver, I'll give you a question.\nYour task is:\n- First, answer the question.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: 1923/05/10'
    Question: The date {num} days {direction} is {date}, what is the date today?
'''

word_problem_prompt='''
You are a good game problem-solver, I'll give you a question.\nYour task is:\n- First, answer the question.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: apple'
    Question: {question}
'''

problem_2048_prompt='''
You are a good game problem-solver, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: LEFT'

Rules:The game is played on a 4x4 grid, with each tile containing a number that is a power of 2 (e.g., 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048). Your goal is to combine the tiles to have more scores. The game ends when there are no more valid moves, or when you achieve the 2048 tile.In the game board, 0 means empty tile and | means the delimiter between tiles. At the beginning of the game, two tiles with the number 2 or 4 will appear randomly on the grid. You can swipe left, right, up, or down to move all tiles in that direction. All tiles will shift to the edge of the grid, and any empty spaces will be filled by a new tile (2 or 4).When two tiles of the same number touch, they will merge into one tile with the sum of those numbers and you will get the score of the new tiles. For example, two tiles with the number 2 will merge to form a 4. After merging, the new tile will not combine again in the same move. You lose the game if the grid is full, and no valid moves are left. A valid move is when two adjacent tiles are the same or there is an empty space to move a tile into. Keep in mind that combining tiles strategically is key. Try to keep the larger tiles in a corner and work towards merging smaller tiles to get higher scores.

For example,if the Game board is

0|0|4|0

0|2|0|8

0|0|4|0

0|0|0|2

and the answer is DOWN

the next state of Game board will be

0|0|0|0

0|0|0|0

0|0|0|8

0|2|8|2

and since the two '4' merge into '8',so you will get 8 score

Game board:

{board}

The answer you give should be one of 'LEFT', 'RIGHT', 'UP' and 'DOWN'
'''

snake_game_prompt='''
You are a good game player, I'll give you a game board and rules.\nYour task is:\n- First, give your answer according to the game board and rules.\n- Second, output the answer in the required format. The last line of your response should be in the following format: 'Answer: $YOUR_ANSWER' (without quotes), where YOUR_ANSWER is your final answer to the question,e.g.'Answer: LEFT'

You are controlling a snake in a Snake game.The board size is 8x8.In the game board. The goal of the game is to control the snake, eat as many apples as possible, and grow the snake in length. Each time the snake eats an apple, the score increases by one. The game ends when the snake collides with itself or the walls of the board.The game board is a grid with walls by '#' around the edges.The snake starts with a length of 1 (just the head). The head is represented by 'H' and the body by 'S'. The game starts with 3 apples placed randomly on the board. Apples are represented by 'A'. The snake starts moving in the 'UP' direction. The snake moves one square at a time in the direction it is facing: 'UP', 'DOWN', 'LEFT', or 'RIGHT'. The player controls the snake’s movement by providing direction inputs. The snake cannot reverse its direction (i.e., it cannot turn 'UP' to 'DOWN' or 'LEFT' to 'RIGHT' directly).The snake loses the game if it collides with itself  or the walls. Each time the snake's head moves to a square occupied by an apple ('A'), the snake eats the apple and grows in length and meanwhile, the score will increase 1 point. The Current direction indicates the direction in which the snake is currently moving forward and the Game board indicates the current map of game.

For example,if the board is 
########
#     A#
#S     #
#H     #
#AA    #
#      #
#      #
########

and the direction you give is DOWN,then you will eat an apple ,increase 1 score and the next state of board will be 
########
#  A  A#
#S     #
#S     #
#HA    #
#      #
#      #
########

The Direction you give should be one of 'LEFT', 'RIGHT', 'UP' and 'DOWN'

Game board:

{board}

Current direction: {direction}

'''