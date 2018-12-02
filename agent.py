import numpy as np

class Agent:
    def __init__(self, env):
        # Store environment, state and action dimension
        self.env            = env
        self.state_dim      = env.state_dim
        self.max_action_dim = env.max_action_dim
        self.state          = env.randomState()

    def get_action(self, t):
        """Get the action to perform."""
        # Randomly do stuff
        return np.random.choice(self.env.adjacent(self.state)[0])

    def train(self, memory):
        pass

    def reward(self, state, time):
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
                self.Q[(s, tuple(a))] = 0

    def get_action(self, t):
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

    def reward(self, state, time):
        print(state)
        return self.env[tuple(int(i) for i in state)]

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
        max_next_value = np.amax(list(self.Q[(state_next, a)] for a in self.env.actions_allowed(state_next)))
        self.Q[sa] += self.beta * (reward + self.gamma*max_next_value - self.Q[sa])

    def display_q_values(self):
        # greedy policy = argmax[a'] Q[s,a']
        for k, v in self.Q.items():
            s, a = k
            print("State: ", s, "; Action: ", a, "; Q-Value", v)
