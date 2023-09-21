class CliffWalkingEnv:
    def __init__(self, nrow=4, ncol=12):
        self.nrow = nrow
        self.ncol = ncol
        self.P = self.creatEnv()

    def creatEnv(self):
        """
        (0, 0) -- -- -- -- -- -- -- -- --  --  --  > col
        |   .  .  .  .  .  .  .  .  .  .  .  .   |
        |   .  .  .  .  .  .  .  .  .  .  .  .   |
        |   .  .  .  .  .  .  .  .  .  .  .  .   |
        begin  x  x  x  x  x  x  x  x  x  x  target
        v
        row
        """
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]   # up, down, left, right
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]

        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow-1 and j > 0:
                        P[i*self.ncol+j][a] = [(1, i*self.ncol+j, 0, True)]
                        continue
                    next_x = min(self.ncol-1, max(0, j+change[a][0]))
                    next_y = min(self.nrow-1, max(0, i+change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = 0
                    done = False
                    if next_y == self.nrow-1 and next_x > 0:
                        done = True
                        if next_x != self.ncol-1:
                            reward = -100
                        else:
                            reward = 100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P