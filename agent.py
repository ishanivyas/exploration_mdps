import numpy as np

class Agent:
    def __init__(self, env, r=np.random):
        # Store environment, state and action dimension
        self.r              = r
        self.env            = env
        self.state_dim      = env.state_dim
        self.max_action_dim = env.max_action_dim
        self.state          = env.randomState()

    def get_action(self, t):
        """Get the action to perform."""
        pass

    def train(self, memory):
        pass

    def reward(self, state, time):
        """Get the reward for an agent in a particular state of the world at a given time."""
        pass


class RandomAgent(Agent):
    def __init__(self, env, r=np.random):
        super(RandomAgent, self).__init__(env, r)

    def get_action(self, t):
        return self.r.choice(self.env.actions_allowed(self.state))

class GreedyAgent(Agent):
    def __init__(self, env, r=np.random):
        super(GreedyAgent, self).__init__(env, r)
        self.beta          = 0.99  # learning rate
        self.gamma         = 0.99  # reward discount factor

        # Initialize Q[s,a] table
        self.Q = {}
        states = env.enumerateStates()
        for s in states:
            for a in env.actions_allowed(s):
                self.Q[(s, tuple(a))] = 0

    def reward(self, state, time):
        print(state)
        return self.env[tuple(int(i) for i in state)]

    def get_action(self, t):
        actions_allowed = self.env.actions_allowed(self.state)
        Q_s             = self.Q[(self.state, tuple(actions_allowed))]
        actions_greedy  = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
        return self.r.choice(actions_greedy)

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

    def display_q_values(self):
        # greedy policy = argmax[a'] Q[s,a']
        for k, v in self.Q.items():
            s, a = k
            print("State: ", s, "; Action: ", a, "; Q-Value", v)


class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, env, eps=1.0, decay=1.0, r=np.random):
        super(EpsilonGreedyAgent, self).__init__(env, r)
        # Epsilon learning parameters
        self.epsilon       = eps   # initial exploration probability
        self.epsilon_decay = decay # epsilon decay after each episode

    def get_action(self, t):
        # Epsilon-greedy agent policy
        if self.r.uniform(0, 1) < self.epsilon:
            # Explore
            self.epsilon *= self.epsilon_decay
            actions_allowed = self.env.actions_allowed(self.state)
            return actions_allowed[int(self.r.uniform(len(actions_allowed)))]
        else:
            # Exploit on allowed actions
            return super(EpsilonGreedyAgent, self).get_action(t)
