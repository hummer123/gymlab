from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import str_key


class Gamer():
    def __init__(self, name = "", A = None, display = False):
        self.name = name
        self.A = A # 动作空间
        self.display = display
        self.cards = []
        self.policy = None
        self.learning_rate = None

    def __str__(self):
        return self.name
    
    def _value_of(self, card):
        ''' 根据牌的字符判断牌的数值大小，A 被视为1, J, Q, K 被视为10， 其他牌按数字大小计算 
        Args:
            card: str, 牌的字符表示
        Return:
            int, 牌的数值大小
        '''
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J', 'Q', 'K']:
                v = 10
            else:
                v = 0
        finally:
            return v
        
    def get_points(self):
        num_of_useable_ace = 0
        total_point = 0
        cards = self.cards

        if cards is None:
            return 0, False
        for card in cards:
            v = self._value_of(card)
            if v == 1:
                num_of_useable_ace += 1
                v = 11
            total_point += v

        while total_point > 21 and num_of_useable_ace > 0:
            total_point -= 10
            num_of_useable_ace -= 1

        return total_point, bool(num_of_useable_ace)
    
    def receive(self, cards = []):
        cards = list(cards)
        for card in cards:
            self.cards.append(card)

    def discharge_cards(self):
        ''' 清空手牌 '''
        self.cards.clear()

    def cards_info(self):
        ''' 显示手牌信息 '''
        return self._info(f"  - {self} -> {self.cards}\n")
    
    def _info(self, msg):
        if self.display:
            print(msg, end='')


class Dealer(Gamer):
    ''' 庄家类 '''
    def __init__(self, name = "", A = None, display = False):
        super(Dealer, self).__init__(name, A, display)
        self.policy = self.dealer_policy # 庄家策略

    def first_card_value(self):
        ''' 获取庄家的第一张牌的点数 '''
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self):
        ''' 庄家策略：点数小于17点时继续要牌，否则停止要牌 '''
        dealer_points, _ = self.get_points()
        action = ""
        if dealer_points < 17:
            action = self.A[0] # 要牌
        else:
            action = self.A[1] # 停牌
        return action
    

class Player(Gamer):
    ''' 玩家类 '''
    def __init__(self, name = "", A = None, display = False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy # 玩家策略，由外部设置

    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return (dealer_first_card_value, player_points, useable_ace)
    
    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))
    
    def naive_policy(self):
        ''' 玩家策略：点数小于12点时继续要牌，否则停止要牌 '''
        player_points, _ = self.get_points()
        action = ""
        if player_points < 12:
            action = self.A[0] # 要牌
        else:
            action = self.A[1] # 停牌
        return action


class Arena():
    ''' 21点游戏环境类 '''
    def __init__(self, A = None, display = False):
        self.cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'] * 4
        self.card_q = Queue(maxsize=52) # 牌堆
        self.cards_in_pool = [] # 已经用过的牌池
        self.A = A
        self.display = display
        self.load_cards(self.cards) # 初始化牌堆
        self.episodes = [] # 记录每局游戏的信息
    
    def load_cards(self, cards):
        ''' 初始化牌堆 '''
        shuffle(cards)
        for card in cards:
            self.card_q.put(card)
        cards.clear()
        return 
    
    def reward_of(self, dealer, player):
        ''' 计算奖励 '''
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        reward = 0

        if player_points > 21:
            reward =  -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, dealer_points, player_points, useable_ace

    def serve_card_to(self, gamer, n=1):
        ''' 给玩家发牌 
        Args:
            gamer: Gamer, 庄家/玩家
            n: int, 发牌数量
        Return:
            None
        '''
        cards = []
        for _ in range(n):
            if self.card_q.empty():
                self._info("Reshuffling the cards...\n")
                shuffle( self.cards_in_pool )
                self._info(f"Cards in pool before reshuffling: {len(self.cards_in_pool)}\n")
                assert(len(self.cards_in_pool) > 20) # 确保牌池中有足够的牌进行重新洗牌
                self.load_cards( self.cards_in_pool)
            cards.append(self.card_q.get())

        self._info(f"send({n}) card: {cards} to {gamer}\n")

        gamer.receive(cards)
        gamer.cards_info()
   
    def recycle_cards(self, *gamers):
        ''' 回收玩家手牌到牌池 '''
        for gamer in gamers:
            self.cards_in_pool.append(gamer.cards)
            gamer.discharge_cards()

    def play_game(self, dealer, player):
        ''' 进行一局游戏 
        Args:
            dealer: Dealer, 庄家
            player: Player, 玩家
        Return:
            reward: int, 奖励值
        '''
        self._info("=== New Game ===\n")
        self.serve_card_to(dealer, n=2)
        self.serve_card_to(player, n=2)
        episode = []

        if player.policy is None:
            self._info("Player policy is not defined!\n")
            return 
        if dealer.policy is None:
            self._info("Dealer policy is not defined!\n")
            return

        # 玩家行动
        self._info(f"{player} starts playing...\n")
        while True:
            action = player.policy()
            self._info(f"{player} action: {action}\n")
            episode.append( (player.get_state(dealer), action) ) # 记录一个(s, a)
            if action == self.A[0]: # type: ignore # 要牌
                self.serve_card_to(player)
            else: # 停牌
                break
        
        reward, dealer_points, player_points, useable_ace = self.reward_of(dealer, player)

        if player_points > 21:
            self._info(f"{player} busts with {player_points} points!!! Reward: {reward}\n")
            self.recycle_cards(player, dealer)
            self.episodes.append( (episode, reward) )
            self._info("======= game over =======\n\n")
            return episode, reward
        
        # 庄家行动
        self._info(f"{dealer} starts playing...\n")
        while True:
            action = dealer.policy()
            self._info(f"{dealer} action: {action}\n")
            if action == self.A[0]: # 要牌
                self.serve_card_to(dealer)
            else: # 停牌
                break
        
        self._info(f"Other stop to call card.\n")
        reward, dealer_points, player_points, useable_ace = self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()

        if reward == 1:
            self._info(f"{player} Wins with {player_points} points!!! Reward: {reward}\n")
        elif reward == -1:
            self._info(f"{player} Loses with {player_points} points!!! Reward: {reward}\n")
        else:
            self._info(f"{player} Draws with {player_points} points!!! Reward: {reward}\n")

        self._info(f"player final points: {player_points}，dealer final points: {dealer_points}\n")
        self._info("======= game over =======\n\n")
        self.recycle_cards(player, dealer)
        self.episodes.append( (episode, reward) )
        return episode, reward
    
    def play_games(self, dealer, player, n_episodes=2, show_statistics=True):
        ''' 进行多局游戏 
        Args:
            dealer: Dealer, 庄家
            player: Player, 玩家
            n_episodes: int, 游戏局数
            show_statistics: bool, 是否显示统计信息
        Return:
            None
        '''
        results = [0, 0, 0] # 输，平局，赢
        self.episodes.clear()

        for _ in tqdm(range(n_episodes), desc="Playing games"):
            episode, reward = self.play_game(dealer, player) # type: ignore
            results[1 + reward] += 1
            # if player.learning_method is not None:
            #     player.learning_method(episode, reward)

        if show_statistics:
            print(f"Out of {n_episodes} games: Wins: {results[2]}, Draws: {results[1]}, Losses: {results[0]}, Win Rate: {results[2] / n_episodes:.2%}")
        return 
    
    def _info(self, msg):
        if self.display:
            print(msg, end='')


# Generate Game
A = ['hit', 'stand']
display = False

player = Player(name="Player1", A=A, display=display)
dealer = Dealer(name="Dealer1", A=A, display=display)

arena = Arena(A=A, display=display)

arena.play_games(dealer, player, n_episodes=1000)
