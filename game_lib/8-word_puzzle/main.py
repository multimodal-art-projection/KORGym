import math
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict
import os
import base64

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
    
class CrosswordGenerator:
    def __init__(self, word_clues_path: str = "high_quality_word_clues.csv"):
        """初始化填字游戏生成器"""
        self.word_bank = pd.read_csv(word_clues_path)
        self.word_bank['word'] = self.word_bank['word'].str.strip().str.lower()
        self.word_bank = self.word_bank.drop_duplicates('word').set_index('word')['clue'].to_dict()
        
        self.valid_words: Dict[str, str] = {
            w: c for w, c in self.word_bank.items() 
            if 3 <= len(w) <= 12 and w.isalpha()
        }


    def _select_words(self, num: int, seed: int):
        random.seed(seed)
        words = list(self.valid_words.keys())
        weights = [3 if 5 <= len(w) <=8 else 1 for w in words]
        
        selected = []
        while len(selected) < num and words:
            chosen = random.choices(words, weights=weights, k=1)[0]
            if chosen not in selected:
                selected.append(chosen)
                idx = words.index(chosen)
                words.pop(idx)
                weights.pop(idx)
        
        descriptions = [self.valid_words[w] for w in selected]
        return selected, descriptions

    def _place_words(self, words: List[str], grid_size: int = 20) -> Tuple[List[List[str]], List[Dict]]:
        grid = [[None]*grid_size for _ in range(grid_size)]
        placed_info = []
        
        sorted_words = sorted(enumerate(words), key=lambda x: -len(x[1]))
        
        for original_idx, word in sorted_words:
            word = word.upper()
            placed = False
            
        
            max_attempts = 200 if original_idx == 0 else 500
            for _ in range(max_attempts):
                direction = random.choice(['across', 'down'])
                word_len = len(word)
                
                # 根据方向计算最大坐标
                if direction == 'across':
                    max_col = grid_size - word_len
                    max_row = grid_size - 1
                else:
                    max_row = grid_size - word_len
                    max_col = grid_size - 1
                
                if max_row < 0 or max_col < 0:
                    continue
                
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
                
                valid = True
                overlaps = 0
                temp_grid = [row[:] for row in grid]
                
                for i in range(word_len):
                    r = start_row + (i if direction == 'down' else 0)
                    c = start_col + (i if direction == 'across' else 0)
                    
                    if temp_grid[r][c]:
                        if temp_grid[r][c] != word[i]:
                            valid = False
                            break
                        overlaps += 1
                    else:
                        temp_grid[r][c] = word[i]
                
                # 交叉检查逻辑（首单词不需要交叉）
                if valid and (overlaps > 0 or original_idx == 0):
                    # 写入网格
                    for i in range(word_len):
                        r = start_row + (i if direction == 'down' else 0)
                        c = start_col + (i if direction == 'across' else 0)
                        grid[r][c] = temp_grid[r][c]
                    
                    placed_info.append({
                        'number': original_idx + 1,
                        'row': start_row,
                        'col': start_col,
                        'direction': direction,
                        'word': word
                    })
                    placed = True
                    break
                
            if not placed:
                return grid, placed_info
                    
        return grid, placed_info

    # def _render_image(self, char_grid: List[List[str]], placed_info: List[dict], difficulty: float) -> str:
    #     """
    #     生成填字游戏图片（带难度控制的字母遮盖）
        
    #     参数：
    #         difficulty: 0-1之间的难度值，0最易，1最难
    #     """
    #     grid_size = len(char_grid)
    #     cell_size = 35
    #     padding = 25
        
    #     img = Image.new('RGB', 
    #         (grid_size*cell_size + 2*padding, 
    #         grid_size*cell_size + 2*padding),
    #         color=(255, 255, 255))
        
    #     draw = ImageDraw.Draw(img)
    #     masked_positions = set()

    #     # 计算需要遮盖的位置
    #     for info in placed_info:
    #         word = info['word']
    #         word_len = len(word)
            
    #         # 根据难度计算遮盖数量
    #         k = max(1, math.ceil(difficulty * word_len))
    #         k = min(k, word_len)  # 确保不超过单词长度
            
    #         # 随机选择遮盖位置
    #         indices = random.sample(range(word_len), k)
            
    #         # 将索引转换为网格坐标
    #         start_row, start_col = info['row'], info['col']
    #         direction = info['direction']
            
    #         for i in indices:
    #             if direction == 'across':
    #                 r = start_row
    #                 c = start_col + i
    #             else:
    #                 r = start_row + i
    #                 c = start_col
    #             masked_positions.add((r, c))

    #     # 加载字体
    #     try:
    #         font = ImageFont.truetype("arial.ttf", 14)
    #         num_font = ImageFont.truetype("arial.ttf", 12)
    #     except:
    #         font = ImageFont.load_default()
    #         num_font = ImageFont.load_default()

    #     # 绘制网格和内容
    #     for r in range(grid_size):
    #         for c in range(grid_size):
    #             x0 = padding + c * cell_size
    #             y0 = padding + r * cell_size
    #             x1 = x0 + cell_size
    #             y1 = y0 + cell_size

    #             is_active = char_grid[r][c] is not None
                
    #             if is_active:
    #                 # 绘制白底格子
    #                 draw.rectangle([x0, y0, x1, y1], fill='white', outline='#CCCCCC')
                    
    #                 # 绘制编号
    #                 for info in placed_info:
    #                     if info['row'] == r and info['col'] == c:
    #                         draw.text(
    #                             (x0 + 2, y0 + 2), 
    #                             str(info['number']), 
    #                             fill='#FF4444',
    #                             font=num_font
    #                         )
                    
    #                 # 根据遮盖状态显示字母或下划线
    #                 if (r, c) not in masked_positions:
    #                     char = char_grid[r][c]
    #                     # 计算居中位置
    #                     bbox = draw.textbbox((0, 0), char, font=font)
    #                     text_width = bbox[2] - bbox[0]
    #                     text_height = bbox[3] - bbox[1]
    #                     tx = x0 + (cell_size - text_width) / 2
    #                     ty = y0 + (cell_size - text_height) / 2
    #                     draw.text((tx, ty), char, fill='black', font=font)
    #                 else:
    #                     # 绘制下划线表示遮盖
    #                     underline_y = y0 + cell_size - 4
    #                     draw.line([x0+3, underline_y, x1-3, underline_y], 
    #                             fill='#666666', width=2)
    #             else:
    #                 # 绘制灰底格子
    #                 draw.rectangle([x0, y0, x1, y1], fill='#DDDDDD', outline='#CCCCCC')

    #     img_path = f'crossword_{random.randint(1000,9999)}.png'
    #     img.save(img_path)
    #     return img_path

    def _render_image(self, char_grid: List[List[str]], placed_info: List[dict], difficulty: float):
        """
        生成填字游戏图片（带难度控制的字母遮盖）

        参数：
            difficulty: 0-1之间的难度值，0最易，1最难
        """
        grid_size = len(char_grid)
        cell_size = 35

        # 计算实际填字的起始和结束位置
        min_row = min(info['row'] for info in placed_info)
        max_row = max(info['row'] + (len(info['word']) if info['direction'] == 'down' else 0) for info in placed_info)
        min_col = min(info['col'] for info in placed_info)
        max_col = max(info['col'] + (len(info['word']) if info['direction'] == 'across' else 0) for info in placed_info)

        # 计算裁剪后的网格大小
        cropped_rows = max_row - min_row
        cropped_cols = max_col - min_col

        padding = 25
        img = Image.new('RGB', 
            ((cropped_cols + 1) * cell_size + 2 * padding, 
            (cropped_rows + 1) * cell_size + 2 * padding),
            color=(255, 255, 255))

        draw = ImageDraw.Draw(img)
        masked_positions = set()

        # 计算需要遮盖的位置
        for info in placed_info:
            word = info['word']
            word_len = len(word)

            # 根据难度计算遮盖数量
            k = max(1, math.ceil(difficulty * word_len))
            k = min(k, word_len)

            # 随机选择遮盖位置
            indices = random.sample(range(word_len), k)

            # 将索引转换为网格坐标
            start_row, start_col = info['row'], info['col']
            direction = info['direction']

            for i in indices:
                if direction == 'across':
                    r = start_row
                    c = start_col + i
                else:
                    r = start_row + i
                    c = start_col
                masked_positions.add((r, c))

        # 加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            num_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            num_font = ImageFont.load_default()

        # 绘制网格和内容
        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                x0 = padding + (c - min_col) * cell_size
                y0 = padding + (r - min_row) * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                is_active = char_grid[r][c] is not None
                
                if is_active:
                    # 绘制白底格子
                    draw.rectangle([x0, y0, x1, y1], fill='white', outline='#CCCCCC')
                    
                    # 绘制编号
                    for info in placed_info:
                        if info['row'] == r and info['col'] == c:
                            draw.text(
                                (x0 + 2, y0 + 2), 
                                str(info['number']), 
                                fill='#FF4444',
                                font=num_font
                            )
                    
                    # 根据遮盖状态显示字母或下划线
                    if (r, c) not in masked_positions:
                        char = char_grid[r][c]
                        # 计算居中位置
                        bbox = draw.textbbox((0, 0), char, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        tx = x0 + (cell_size - text_width) / 2
                        ty = y0 + (cell_size - text_height) / 2
                        draw.text((tx, ty), char, fill='black', font=font)
                    else:
                        # 绘制下划线表示遮盖
                        underline_y = y0 + cell_size - 4
                        draw.line([x0+3, underline_y, x1-3, underline_y], 
                                fill='#666666', width=2)
                else:
                    # 绘制灰底格子
                    draw.rectangle([x0, y0, x1, y1], fill='#DDDDDD', outline='#CCCCCC')
        os.makedirs('cache', exist_ok=True)
        img_path = f'cache/crossword_{random.randint(1000,9999)}.png'
        img.save(img_path)
        return img_path


    def generate(self, seed: int):
        """生成填字游戏"""
        num = random.randint(5,15)
        difficulty = random.randint(5,9)/10
        if not 0 <= difficulty <= 1:
            raise ValueError("难度值必须在0到1之间")
        if not (1 <= num <= 15):
            raise ValueError("单词数量需在1-15之间")
        if num > len(self.valid_words):
            raise ValueError("可用单词不足")
        
        # 带重试机制的生成循环
        max_retries = 20
        used_words = set()
        
        for retry in range(max_retries):
            # 选择候选单词（排除已用过的）
            candidates = [w for w in self.valid_words if w not in used_words]
            selected, descs = self._select_words(num, seed + retry)
            
            # 尝试布局
            grid, placed = self._place_words(selected)
            placed_numbers = {p['number'] for p in placed}
            
            # 成功条件检查
            if len(placed) == num:
                final_words = [p['word'].lower() for p in sorted(placed, key=lambda x:x['number'])]
                final_descs = [descs[p['number']-1] for p in sorted(placed, key=lambda x:x['number'])]
                img_path = self._render_image(grid, placed, difficulty)
                return img_path, final_descs, final_words
            
            # 记录失败单词避免重复使用
            used_words.update(selected)
        
        raise RuntimeError(f"在{max_retries}次尝试后仍无法生成有效的填字游戏")

    @staticmethod
    def verify(correct: List[str], answers: List[str]):
        """验证答案"""
        correct_dict = {i+1:w.lower() for i,w in enumerate(correct)}
        answer_dict = {}
        
        for ans in answers:
            parts = ans.replace(' ', '').split('.', 1)
            if len(parts) != 2:
                continue
            try:
                num = int(parts[0])
                answer_dict[num] = parts[1].lower()
            except:
                continue
        
        return sum(1 for k in correct_dict if answer_dict.get(k, '') == correct_dict[k])

# 使用示例
if __name__ == "__main__":
    generator = CrosswordGenerator()
    
    try:
        image_path, clues, answers = generator.generate(seed=42)
        print(f"成功生成填字游戏：{image_path}")
        print(answers)
        print("提示：")
        for i, clue in enumerate(clues, 1):
            print(f"{i}. {clue}")
        
        # 模拟用户答案
        user_answers = [f"{i+1}. {w}" for i,w in enumerate(answers)]
        score = CrosswordGenerator.verify(answers, user_answers)
        print(f"验证结果：{score}/{len(answers)}")
    except Exception as e:
        print(f"生成失败：{str(e)}")
