class SpiderSolitaire:
    def __init__(self, board=None, deck=None, visibility=None):
        """
        初始化蜘蛛纸牌游戏

        参数:
            board: 可选预定义的牌面状态
            deck: 可选预定义的牌堆
            visibility: 可选预定义牌面可见状态
        """
        self.board = board if board else []
        self.deck = deck if deck else []  # 牌堆中剩余的卡牌
        self.visibility = visibility if visibility else []  # 记录卡牌是否可见
        self.completed_sets = 0  # 已经完成的牌组数（从K到A的同花色序列）
        self.score = 500  # 初始分数（普通模式在 setup_game 中被设为 0）
        self.steps = 0  # 玩家移动次数
        self.history = []  # 保存移动历史，用于撤销操作

    def setup_game(self, seed=None):
        """
        初始化新的一局蜘蛛纸牌游戏

        参数:
            seed: 随机种子（用于重现性）
        """
        import random
        if seed is not None:
            random.seed(seed)

        # 创建并洗牌 (两副牌共104张)
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        self.deck = []
        for _ in range(2):  # 两副牌
            for suit in suits:
                for rank in ranks:
                    self.deck.append((suit, rank))

        # 洗牌，并作进一步随机处理
        random.shuffle(self.deck)
        mid_point = len(self.deck) // 2
        cut_point = random.randint(mid_point - 10, mid_point + 10)
        temp = self.deck[cut_point:]
        random.shuffle(temp)
        self.deck = self.deck[:cut_point] + temp

        # 初始化10列的牌面和对应可见状态
        self.board = [[] for _ in range(10)]
        self.visibility = [[] for _ in range(10)]

        # 前4列每列6张牌
        for i in range(4):
            for j in range(6):
                card = self.deck.pop(0)
                self.board[i].append(card)
                # 仅最后一张牌初始时可见
                self.visibility[i].append(j == 5)

        # 其余6列每列5张牌
        for i in range(4, 10):
            for j in range(5):
                card = self.deck.pop(0)
                self.board[i].append(card)
                # 仅最后一张牌初始时可见
                self.visibility[i].append(j == 4)

        self.completed_sets = 0
        self.score = 0  # 普通模式下初始分数设为 0
        self.steps = 0
        self.history = []

        return self.get_visible_board()

    def setup_cheat_mode(self):
        """
        设置作弊模式，提供较易获胜的预设牌面，用于测试
        """
        self.board = [[] for _ in range(10)]
        self.visibility = [[] for _ in range(10)]
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

        # 第一列：近乎完整的红心序列（K到3）
        self.board[0] = [(suits[0], r) for r in ranks[2:]][::-1]
        self.visibility[0] = [True] * len(self.board[0])

        # 第二列：近乎完整的方块序列（K到3）
        self.board[1] = [(suits[1], r) for r in ranks[2:]][::-1]
        self.visibility[1] = [True] * len(self.board[1])

        # 第三列：红心的A和2
        self.board[2] = [(suits[0], '2'), (suits[0], 'A')]
        self.visibility[2] = [True] * len(self.board[2])

        # 第四列：方块的A和2
        self.board[3] = [(suits[1], '2'), (suits[1], 'A')]
        self.visibility[3] = [True] * len(self.board[3])

        # 第五列：近乎完整的梅花序列（K到9）
        self.board[4] = [(suits[2], r) for r in ranks[8:]][::-1]
        self.visibility[4] = [True] * len(self.board[4])

        # 第六列：近乎完整的梅花序列（9到4）
        self.board[5] = [(suits[2], r) for r in ranks[3:8]][::-1]
        self.visibility[5] = [True] * len(self.board[5])

        # 第七列：近乎完整的黑桃序列（3到K）
        self.board[6] = [(suits[2], r) for r in ranks[:3]][::-1]
        self.visibility[6] = [True] * len(self.board[6])

        # 剩余列填充一些随机牌
        for i in range(7, 10):
            self.board[i] = [(suits[i % 4], ranks[j]) for j in range(3)]
            self.visibility[i] = [True] * len(self.board[i])
        print(self.board)
        
        # 设置剩余牌堆
        self.deck = []
        for _ in range(5):  # 添加一些牌到牌堆中
            for suit in suits:
                self.deck.append((suit, 'K'))

        self.completed_sets = 0
        self.score = 500
        self.steps = 0
        self.history = []

    def get_card_value(self, card):
        """
        获取卡牌的数值（用于比较大小）

        参数:
            card: 一个元组 (花色, 点数)

        返回:
            int: 卡牌的数值
        """
        rank_values = {
            'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13
        }
        return rank_values[card[1]]

    def is_sequence(self, cards):
        """
        判断一组牌是否构成同花色连续递减的序列

        参数:
            cards: 待检测的一组卡牌列表

        返回:
            bool: 若牌组构成有效序列则返回True
        """
        if not cards:
            return False

        suit = cards[0][0]
        if any(card[0] != suit for card in cards):
            return False

        values = [self.get_card_value(card) for card in cards]
        for i in range(1, len(values)):
            if values[i] != values[i-1] - 1:
                return False

        return True

    def is_complete_sequence(self, cards):
        """
        检查卡牌列表是否构成完整的同花色K到A的序列

        参数:
            cards: 待检测的卡牌列表

        返回:
            bool: 若构成完整序列则返回True
        """
        if len(cards) != 13:
            return False

        if self.get_card_value(cards[0]) != 13 or self.get_card_value(cards[-1]) != 1:
            return False

        return self.is_sequence(cards)

    def can_move_cards(self, from_column, start_idx, to_column):
        """
        检查是否可以将一组牌从一列移动到另一列

        参数:
            from_column: 源列索引
            start_idx: 从哪张牌开始移动（索引）
            to_column: 目标列索引

        返回:
            bool: 如果该移动有效则返回True
        """
        if from_column < 0 or from_column >= len(self.board):
            return False
        if start_idx < 0 or start_idx >= len(self.board[from_column]):
            return False
        if to_column < 0 or to_column >= len(self.board):
            return False
        if not self.visibility[from_column][start_idx]:
            return False

        cards_to_move = self.board[from_column][start_idx:]
        if len(cards_to_move) > 1 and not self.is_sequence(cards_to_move):
            return False

        if self.board[to_column]:
            top_card = self.board[to_column][-1]
            if not self.visibility[to_column][-1]:
                return False
            if self.get_card_value(cards_to_move[0]) != self.get_card_value(top_card) - 1:
                return False

        return True

    def move_cards(self, from_column, start_idx, to_column):
        """
        将牌从一列移动到另一列

        参数:
            from_column: 源列索引
            start_idx: 待移动牌组在源列中的起始索引
            to_column: 目标列索引

        返回:
            bool: 如果移动有效且成功则返回True
        """
        if not self.can_move_cards(from_column, start_idx, to_column):
            return False

        move_record = {
            'type': 'move',
            'from_column': from_column,
            'start_idx': start_idx,
            'to_column': to_column,
            'cards': self.board[from_column][start_idx:],
            'visibility': self.visibility[from_column][start_idx:],
            'reveal': len(self.board[from_column]) > start_idx and start_idx > 0 and not self.visibility[from_column][start_idx-1]
        }

        # 移动卡牌及其可见状态
        cards_to_move = self.board[from_column][start_idx:]
        visibility_to_move = self.visibility[from_column][start_idx:]

        self.board[from_column] = self.board[from_column][:start_idx]
        self.visibility[from_column] = self.visibility[from_column][:start_idx]

        self.board[to_column].extend(cards_to_move)
        self.visibility[to_column].extend(visibility_to_move)

        # 如果源列还剩牌，确保最后一张牌可见
        if self.board[from_column] and not self.visibility[from_column][-1]:
            self.visibility[from_column][-1] = True
            move_record['revealed_card'] = self.board[from_column][-1]

        # 不再扣分，原来这里会执行 self.score -= 1
        self.steps += 1

        self.history.append(move_record)

        # 检查是否有完成的牌序
        completed = self.check_completed_sequences()
        if completed > 0:
            self.score += 1 * completed  # 每完成一组完整序列奖励 1 分

        return True

    def deal_cards(self):
        """
        从牌堆中向各列发牌

        返回:
            bool: 如果发牌成功则返回True
        """
        if len(self.deck) < 10:
            return False
        if any(not column for column in self.board):
            return False

        deal_record = {
            'type': 'deal',
            'cards': []
        }

        for i in range(10):
            card = self.deck.pop(0)
            deal_record['cards'].append((i, card))
            self.board[i].append(card)
            self.visibility[i].append(True)

        self.history.append(deal_record)

        completed = self.check_completed_sequences()
        if completed > 0:
            self.score += 1 * completed

        return True

    def check_completed_sequences(self):
        """
        检查并移除已完成的牌序（同花色由K到A）

        返回:
            int: 本次检查中完成并移除的序列数量
        """
        completed = 0

        for column_idx in range(len(self.board)):
            column = self.board[column_idx]
            i = len(column) - 1
            while i >= 12:
                all_visible = all(self.visibility[column_idx][j] for j in range(i-12, i+1))
                if all_visible and self.is_complete_sequence(column[i-12:i+1]):
                    if completed == 0:
                        fold_record = {
                            'type': 'fold',
                            'columns': []
                        }
                        self.history.append(fold_record)
                    fold_record = self.history[-1]
                    fold_record['columns'].append({
                        'column': column_idx,
                        'start_idx': i-12,
                        'cards': column[i-12:i+1],
                        'visibility': self.visibility[column_idx][i-12:i+1]
                    })
                    self.board[column_idx] = column[:i-12] + column[i+1:]
                    self.visibility[column_idx] = self.visibility[column_idx][:i-12] + self.visibility[column_idx][i+1:]
                    column = self.board[column_idx]
                    completed += 1
                    i = len(column) - 1
                else:
                    i -= 1

        for i in range(len(self.board)):
            if self.board[i] and not self.visibility[i][-1]:
                self.visibility[i][-1] = True

        self.completed_sets += completed
        return completed

    def undo(self):
        """
        撤销上一步的移动或发牌操作

        返回:
            bool: 如果撤销成功则返回True
        """
        if not self.history:
            return False

        last_move = self.history.pop()

        if last_move['type'] == 'move':
            from_column = last_move['from_column']
            to_column = last_move['to_column']

            self.board[to_column] = self.board[to_column][:-len(last_move['cards'])]
            self.visibility[to_column] = self.visibility[to_column][:-len(last_move['visibility'])]

            self.board[from_column].extend(last_move['cards'])
            self.visibility[from_column].extend(last_move['visibility'])

            if 'revealed_card' in last_move:
                self.visibility[from_column][-len(last_move['cards'])-1] = False

            # 不再恢复移动前的扣分，因为现在移动不扣分
            self.steps -= 1

        elif last_move['type'] == 'deal':
            for column_idx, card in reversed(last_move['cards']):
                self.board[column_idx].pop()
                self.visibility[column_idx].pop()
                self.deck.insert(0, card)

        elif last_move['type'] == 'fold':
            for fold in last_move['columns']:
                column_idx = fold['column']
                start_idx = fold['start_idx']
                self.board[column_idx] = (
                    self.board[column_idx][:start_idx] +
                    fold['cards'] +
                    self.board[column_idx][start_idx:]
                )
                self.visibility[column_idx] = (
                    self.visibility[column_idx][:start_idx] +
                    fold['visibility'] +
                    self.visibility[column_idx][start_idx:]
                )
                self.completed_sets -= 1
                self.score -= 1  # 撤销一组完成序列，扣回1分

        return True

    def get_visible_board(self):
        """
        根据卡牌的可见性返回当前牌面状态（隐藏的牌显示为 unknown）

        返回:
            list: 可见的牌面状态
        """
        visible_board = []
        for col_idx, column in enumerate(self.board):
            visible_column = []
            for card_idx, card in enumerate(column):
                if self.visibility[col_idx][card_idx]:
                    visible_column.append(card)
                else:
                    visible_column.append(('unknown', 'unknown'))
            visible_board.append(visible_column)
        return visible_board

    def get_state(self):
        """
        获取当前游戏状态，包含剩余hit次数信息
        """
        return {
            'board': self.board,
            'deck': self.deck,
            'visibility': self.visibility,
            'completed_sets': self.completed_sets,
            'score': self.score,
            'steps': self.steps,
            'history': self.history,
            # 计算剩余可发牌次数
            'remaining_hits': len(self.deck) // 10
        }

    def remaining_hits(self) -> int:
        """
        返回剩余可以发牌（hit）的次数
        """
        return len(self.deck) // 10
