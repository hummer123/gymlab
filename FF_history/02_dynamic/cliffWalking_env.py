import copy


class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        # 状态转移矩阵 P[state][action] = [(probability, next_state, reward, done)]
        self.P = self.createP()

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.ncol * self.nrow)]
        # 4种动作， change[0]:上， change[1]:下， change[2]:左， change[3]:右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if (i == self.nrow - 1) and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
        return P