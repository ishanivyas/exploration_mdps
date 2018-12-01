"""
Creating a 2D and 3D grid environment for agent to move around.
Created By Ishani Vyas
"""
import numpy as np
import random_worlds as rw
from math import log2,ceil

class World:
    def __init__(self):
        pass

    def distance(self, s0, s1):
        """Return the (scalar) distance between the two states."""
        pass

    def contains(self, s):
        """Return True if state is within the bounds of the World."""
        pass

    def adjacent(self, s):
        """Return the actions avalaible from state s and the states they reach."""
        pass

    def actions_allowed(self, s):
        """Return the available actions from `s`."""
        return self.adjacent(s)[0]

    def transition(self, s, a):
        """
            Returns the next state after taking action `a` from state `s`.
            NOTE: for some Worlds this may not be deterministic.
        """
        return np.add(s, a)


class Grid(World):
    def __init__(self):
        pass

    def contains(self, s):
        """Return True if state is within the bounds of the World."""
        return np.amin(s) >= 0 and np.amin(self.bounds - s) > 0

    def distance(self, s0, s1):
        """Return the Manhattan distance between the two states but allow diagonal moves."""
        d = np.abs(s0 - s1)  # Manhattan distance...
        return np.amax(d)    # ...allowing diagonals.


class Grid2D(Grid):
    """
    2D grid world
    """
    def __init__(self, widthOrArray, height=None, maxRange=None):
        """
        Examples:
          w = Grid2D(3,4, 8)  # A 3x4 grid
          w = Grid2D(np.array([[1,2],
                               [2,3]]))
        """
        self.terminalState  = 'TERMINAL_STATE'
        self.state_dim      = 2  # The number of coordinates needed to unambiguously define an agent's state in the world.
        self.max_action_dim = 8  # 3*3 - 1

        if isinstance(widthOrArray, np.ndarray):
            self.width, self.height = widthOrArray.shape[0:2]
            self.data = widthOrArray
        else:
            self.bounds         = np.array([widthOrArray, height])
            self.width          = widthOrArray
            self.height         = height
            r = np.random.uniform
            rCenterScale = r(13.0)  # The factor to reduce noise by in the center.
            rEdgeScale = r(low=rCenterScale, high=r(31.0))  # The factor to reduce noise by along edges/faces.
            d = 2**ceil(log2(max(widthOrArray, height)))
            space = r(maxRange, size=(d+1,d+1))
            rw.mountains(space, d, 0, 0, r, nsc=rCenterScale, nse=rEdgeScale)
            self.data = np.round(space[0:widthOrArray,0:height])

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

    def adjacent(self, s):
        """Return the actions avalaible from state s and the states they reach."""
        adj = []
        act = []
        d = [-1,0,1]
        for dy in d:
            for dx in d:
                action = np.array([dx, dy])
                s_prime = s + action
                if self.distance(s, s_prime) == 0 or not self.contains(s_prime):
                    continue
                act.append(np.array([dx,dy]))
                adj.append(s_prime)
        return act,adj

    def _getLegacyText(self):
        t = [[self.data[x][y] for x in range(self.width)] for y in range(self.height)]
        t.reverse()
        return t

    def __str__(self):
        return str(self._getLegacyText())

    def randomState(self):
        """Returns the coordinates of a random state."""
        return (np.random.randint(self.width),np.random.randint(self.height))

    def enumerateStates(self):
        """Returns a list of all possible state coordinates [(x, y), ...]"""
        states = []
        for i in range(self.width):
            for j in range(self.height):
                states.append((i, j))
        return states


class Grid3D(Grid):
    """
        3D gridworld
    """
    def __init__(self, widthOrArray, heightOrDeep=None, depth=None, maxRange=None):
        """
        Examples:
          w = Grid3D(3,4,5, 8)
          w = Grid3D(np.array([[[1,2], [2,2]],
                               [[1,2], [3,4]]]))
        """
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
            r = np.random.uniform
            rCenterScale = r(13.0)  # The factor to reduce noise by in the center.
            rEdgeScale = r(low=rCenterScale, high=r(31.0))  # The factor to reduce noise by along edges/faces.
            d = 2**ceil(log2(max(widthOrArray, height, depth)))
            space = r(maxRange, size=(d+1, d+1, d+1))
            rw.clouds(space, d, nsc=rCenterScale, nse=rEdgeScale)
            self.data = np.round(space[0:widthOrArray, 0:height, 0:depth])

    def copy(self):
        """Copy the data of a 3D grid into a new instance of Grid3D."""
        return Grid3D(self.data, True)

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        return Grid3D(self.data, False)

    def adjacent(self, s):
        """Return the actions available from state s and the states they reach."""
        act = []
        adj = []
        d = [-1,0,1]
        for dz in d:
            for dy in d:
                for dx in d:
                    if dx == dy == dz == 0: continue
                    action = np.array([dx,dy,dz])
                    s_prime = s + action
                    if np.amin(s_prime) < 0: continue
                    if s_prime[0] >= self.width or s_prime[1] >= self.height or s_prime[2] >= self.depth:
                        continue
                    act.append(action)
                    adj.append(s_prime)
        return act,adj

    def __str__(self):
        return str(self.data)

    def randomState(self):
        return np.array([np.random.randint(self.width),
                         np.random.randint(self.height),
                         np.random.randint(self.depth)])


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
        # Store environment, state and action dimension
        self.env            = env
        self.state_dim      = env.state_dim
        self.max_action_dim = env.max_action_dim
        self.state          = env.randomState()

    def get_action(self):
        """Get the action to perform."""
        # Randomly do stuff
        return np.random.choice(self.env.adjacent(self.state)[0])

    def train(self, memory):
        pass

    def reward(self, world, state, time):
        """Get the reward for an agent in a particular state of the world at a given time."""
        pass

class EpsilonGreedyAgent(Agent):
    def __init__(self, env):
        super(EpsilonGreedyAgent, self).__init__(env)
        # Agent learning parameters
        self.epsilon       = 1.0   # initial exploration probability
        self.epsilon_decay = 0.99  # epsilon decay after each episode
        self.beta          = 0.99  # learning rate
        self.gamma         = 0.99  # reward discount factor

        # Initialize Q[s,a] table
        self.Q = {}
        states = env.enumerateStates()
        for s in states:
            for a in env.actions_allowed(s):
                self.Q[(s, a)] = 0

    def get_action(self):
        actions_allowed = self.env.actions_allowed(self.state)
        # Epsilon-greedy agent policy
        if np.random.uniform(0, 1) < self.epsilon:
            # explore
            return actions_allowed[np.random.choice(len(actions_allowed))]
        else:
            # exploit on allowed actions
            Q_s             = self.Q[self.state, actions_allowed]
            actions_greedy  = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def reward(self, world, state, time):
        return world[state]

    def train(self, memory):
        """
            Update:

                Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])

            Where:

                R[s,a] = reward for taking action a from state s
                beta   = learning rate
                gamma  = discount factor
        """
        (state, action, state_next, reward, done) = memory
        sa = (state, action)
        max_next_value = np.max(list(self.Q[(state_next, a)] for a in self.env.actions_allowed(state_next)))
        self.Q[sa] += self.beta * (reward + self.gamma*max_next_value - self.Q[sa])

    def display_greedy_policy(self):
        # greedy policy = argmax[a'] Q[s,a']
        for k, v in self.Q.items():
            s, a = k
            print("State: ", s, "; Action: ", a, "; Q-Value", v)


def Simulate(agent, env, T, sequential=False):
    print(env.data)

    # Set initial state.
    if sequential:
        next_state = agent.state

    for t_i in range(T):
        # Determine the state the agent begins in at this timestep.
        if not sequential:
            agent.state = env.randomState()
        else:
            agent.state = next_state

        # Get the agent's action at this timestep.
        a_i = agent.get_action()

        # Get the next state by transitioning in the environment
        next_state = env.transition(agent.state, a_i)

        # Get the reward for performing that action at this timestep.
        r_i = agent.reward(env, next_state, t_i)

        # Do one round of training.
        memory = (agent.state, a_i, next_state, r_i, False)
        print("State, Action, New State, Reward at t = %d:" % t_i, memory)
        agent.train(memory)
    print("Here are the agent's final Q values:")
    agent.display_greedy_policy()


if __name__ == "__main__":
    env = Grid2D(3, 3, 10)
    Simulate(EpsilonGreedyAgent(env), env, 100)

