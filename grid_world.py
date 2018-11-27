"""
Creating a 2D and 3D grid environment for agent to move around.
Created By Ishani Vyas
"""
import numpy as np
import matplotlib.pyplot as plt
from math import log2,ceil

np.set_printoptions(linewidth=362)
np.set_printoptions(sign=' ')

def mountains(s, d, x, y, r=np.random.uniform, nsc=3.0, nse=27.0):
    """Apply the diamond-square algorithm to produce random mountains."""
    #   See: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    h = d//2  # Calc the half-dimension using integer division.
    s[x+h,y+h] = (s[x,y] + s[x+d,y] + s[x,y+d] + s[x+d,y+d])/4 + r(d/nsc)  # Center-middle
    # Calculate the edge values.
    s[x+h,y+0] = (s[x+0,y+0] + s[x+d,y+0])/2 + r(d/nse)  # Top-center
    s[x+0,y+h] = (s[x+0,y+0] + s[x+0,y+d])/2 + r(d/nse)  # Left-middle
    s[x+h,y+d] = (s[x+d,y+d] + s[x+0,y+d])/2 + r(d/nse)  # Bottom-center
    s[x+d,y+h] = (s[x+d,y+d] + s[x+d,y+0])/2 + r(d/nse)  # Bottom-middle
    # Recusively calculate the values for the sub-squares
    if h >= 2:
        mountains(s, h, x+0, y+0, r, nsc, nse)
        mountains(s, h, x+0, y+h, r, nsc, nse)
        mountains(s, h, x+h, y+0, r, nsc, nse)
        mountains(s, h, x+h, y+h, r, nsc, nse)

def testMountains(d=16, r=np.random.uniform, nsc=3.0, nse=5.0):
    s = np.ceil(np.random.uniform(0, 40, size=(d+1,d+1)))
    mountains(s, d, 0, 0, r, nsc, nse)
    s = np.ceil(s)
    plt.imshow(s)
    plt.show()
    return s

#-m = testMountains(256)

def clouds(s, d, x, y, z, r=np.random.uniform, nsc=3.0, nsf=27.0):
    """Apply the diamond-square algorithm to produce random clouds."""
    #   See: https://en.wikipedia.org/wiki/Diamond-square_algorithm
    h = d//2  # Calc the half-dimension using integer division.
    s[x+h,y+h,z+h] = (s[x,y,z]   + s[x+d,y,z]   + s[x,y+d,z]   + s[x+d,y+d,z] + s[x,y,z+d] + s[x+d,y,z+d] + s[x,y+d,z+d] + s[x+d,y+d,z+d])/8 + r(d/nsc)  # Center-middle
    # Calculate the 6 face-centered values.
    s[x+h,y+h,z+0] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0] + s[x+d,y+d,z+0] + s[x+d,y+0,z+0])/4 + r(d/nsf)  # XY plane, Z=z+0
    s[x+h,y+0,z+h] = (s[x+0,y+0,z+0] + s[x+d,y+0,z+0] + s[x+d,y+0,z+d] + s[x+0,y+0,z+d])/4 + r(d/nsf)  # XZ plane, Y=y+0
    s[x+0,y+h,z+h] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0] + s[x+0,y+d,z+d] + s[x+0,y+0,z+d])/4 + r(d/nsf)  # YZ plane, X=x+0
    s[x+h,y+h,z+d] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d] + s[x+0,y+0,z+d] + s[x+0,y+d,z+d])/4 + r(d/nsf)  # XY plane, Z=z+d
    s[x+h,y+d,z+h] = (s[x+d,y+d,z+d] + s[x+0,y+d,z+d] + s[x+0,y+d,z+0] + s[x+d,y+d,z+0])/4 + r(d/nsf)  # XZ plane, Y=y+d
    s[x+d,y+h,z+h] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d] + s[x+d,y+0,z+0] + s[x+d,y+d,z+0])/4 + r(d/nsf)  # YZ plane, X=x+d
    # Calculate the 12 edge-centered values.
    s[x+0,y+0,z+h] = (s[x+0,y+0,z+0] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+0,y+h,z+0] = (s[x+0,y+0,z+0] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    s[x+h,y+0,z+0] = (s[x+0,y+0,z+0] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+d,y+d,z+h] = (s[x+d,y+d,z+d] + s[x+d,y+d,z+0])/2 + r(d/nsf)
    s[x+d,y+h,z+d] = (s[x+d,y+d,z+d] + s[x+d,y+0,z+d])/2 + r(d/nsf)
    s[x+h,y+d,z+d] = (s[x+d,y+d,z+d] + s[x+0,y+d,z+d])/2 + r(d/nsf)
    s[x+0,y+d,z+h] = (s[x+0,y+d,z+d] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    s[x+d,y+0,z+h] = (s[x+d,y+0,z+d] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+0,y+h,z+d] = (s[x+0,y+d,z+d] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+d,y+h,z+0] = (s[x+d,y+d,z+0] + s[x+d,y+0,z+0])/2 + r(d/nsf)
    s[x+h,y+0,z+d] = (s[x+d,y+0,z+d] + s[x+0,y+0,z+d])/2 + r(d/nsf)
    s[x+h,y+d,z+0] = (s[x+d,y+d,z+0] + s[x+0,y+d,z+0])/2 + r(d/nsf)
    # Recusively calculate the values for the 8 sub-cubes.
    if h >= 2:
        clouds(s, h, x+0, y+0, z+0, r, nsc, nsf)
        clouds(s, h, x+0, y+0, z+h, r, nsc, nsf)
        clouds(s, h, x+0, y+h, z+0, r, nsc, nsf)
        clouds(s, h, x+0, y+h, z+h, r, nsc, nsf)
        clouds(s, h, x+h, y+0, z+0, r, nsc, nsf)
        clouds(s, h, x+h, y+0, z+h, r, nsc, nsf)
        clouds(s, h, x+h, y+h, z+0, r, nsc, nsf)
        clouds(s, h, x+h, y+h, z+h, r, nsc, nsf)

def testClouds(d):
    s = np.ceil(np.random.uniform(0, 40, size=(d+1,d+1,d+1)))
    clouds(s, d, 0, 0, 0)
    s = np.ceil(s)
    for i in range(min(64,d+1)):
        print("i=%d" % i)
        plt.imshow(s[:,:,i])
        plt.show()
    return s

#-c = testClouds(64)

class World:
    def __init__(self):
        pass

    def distance(self, s0, s1):
        """Return the Manhattan distance between the two states."""
        return np.sum(np.abs(s0 - s1))

    def adjacent(self, s):
        """Return the coordinates of grid spaces next to `s`."""
        pass

class Grid2D(World):
    """
    2D grid world
    """
    def __init__(self, width, height, maxRange):
        self.terminalState  = 'TERMINAL_STATE'
        self.state_dim      = 2  # The number of coordinates needed to unambiguously define an agent's state in the world.
        self.max_action_dim = 8  # 3*3 - 1
        self.width          = width
        self.height         = height

        d = 2**ceil(log2(max(width, height)))
        space = np.random.uniform(maxRange, size=(d+1,d+1))
        mountains(space, d, 0, 0, np.random.uniform)
        self.data = np.round(space[0:width,0:height])

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def copy(self):
        g = Grid2D(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid2D(self.width, self.height)
        g.data = self.data
        return g

    def distance(self, s0, s1):
        """Return the Manhattan distance between the two states but allow diagonal moves."""
        d = np.abs(s0 - s1)
        return np.sum(d) - np.min(d)

    def adjacent(self, s):
        a = []
        d = [0,1]
        for dy in d:
            for dx in d:
                continue if dx == dy == 0
                s_prime = s + np.array([dx, dy])
                continue if s_prime[0] < 0 or s_prime[1] < 0
                continue if s_prime[0] >= self.width or s_prime[1] >= self.height
                a.append(s_prime)
        return a

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())

class Grid3D(World):
    """
        3D gridworld
    """
    def __init__(self, widthOrArray, heightOrDeep=None, depth=None, initialValue=' '):
        self.terminalState  = 'TERMINAL_STATE'
        self.state_dim      = 3   # The number of coordinates needed to unambiguously define an agent's state in the world.
        self.max_action_dim = 26  # 3*3*3 - 1
        if isinstance(widthOrArray, np.ndarray):
            if heightOrDeep:
                self.data = np.array(widthOrArray)
            else:
                self.data = widthOrArray
            self.width = self.data.shape[0]
            self.height = self.data.shape[1]
            self.depth =  self.data.shape[2]
        else:
            self.width = widthOrArray
            self.height = heightOrDeep
            self.depth = depth
            self.data = np.full((widthOrArray,heightOrDeep,depth), initialValue)
        # TODO initialize the world to something more than initialValue:
        #   See: https://en.wikipedia.org/wiki/Diamond-square_algorithm

    def copy(self):
        """Copy the data of a 3D grid into a new instance of Grid3D."""
        return Grid3D(self.data, True)

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        return Grid3D(self.data, False)

    def distance(self, s0, s1):
        """Return the Manhattan distance between the two states but allow diagonal moves."""
        d = np.abs(s0 - s1)
        return np.sum(d) - np.min(d)

    def adjacent(self, s):
        """Return the coordinates of grid spaces next to `s`."""
        a = []
        d = [-1,0,1]
        for dz in d:
            for dy in d:
                for dx in d:
                    continue if dx == dy == 0
                    s_prime = s + np.array([dx, dy])
                    continue if s_prime[0] < 0 or s_prime[1] < 0
                    continue if s_prime[0] >= self.width or s_prime[1] >= self.height
                    a.append(s_prime)
        return a

    def _getLegacyText(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

def makeGrid(gridString):
  width, height = len(gridString[0]), len(gridString)
  grid = Grid2D(width, height)
  for ybar, line in enumerate(gridString):
    y = height - ybar - 1
    for x, el in enumerate(line):
      grid[x][y] = el
  return grid

class Agent:
    def __init__(self, env):
        # Store state and action dimension
        self.state_dim      = env.state_dim
        self.max_action_dim = env.max_action_dim
        self.state          = np.array((self.state_dim))

    def get_action(self, env):
        # Randomly do stuff
        return np.random.choice(env.adjacent(self.state))

    def train(self, memory):
        pass

class EpsilonGreedyAgent(Agent):
    def __init__(self, env):
        super(EpsilonGreedyAgent, self).__init__(self, env)
        # Agent learning parameters
        self.epsilon       = 1.0   # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta          = 0.99  # learning rate
        self.gamma         = 0.99  # reward discount factor

        # Initialize Q[s,a] table
        self.Q = np.zeros(self.state_dim + self.action_dim, dtype=float)

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.adjacent(self.state))
        else:
            # exploit on allowed actions
            actions_allowed = env.adjacent(self.state)
            Q_s             = self.Q[self.state, actions_allowed]
            actions_greedy  = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def train(self, memory):
        # -----------------------------
        # Update:
        #
        # Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])
        #
        #  R[s,a] = reward for taking action a from state s
        #  beta = learning rate
        #  gamma = discount factor
        # -----------------------------
        (state, action, state_next, reward, done) = memory
        sa = state + (action,)
        self.Q[sa] += self.beta * (reward + self.gamma*np.max(self.Q[state_next]) - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        greedy_policy = np.zeros((self.state_dim[0], self.state_dim[1]), dtype=int)
        for x in range(self.state_dim[0]):
            for y in range(self.state_dim[1]):
                greedy_policy[y, x] = np.argmax(self.Q[y, x, :])
        print("\nGreedy policy(y, x):")
        print(greedy_policy)
        print()
