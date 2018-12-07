import numpy as np

def choose(r, array):
    i = r.choice(len(array))
    return array[i]

def sample(r, array, probs):
    i = r.choice(len(array), p=probs)
    return array[i]

class Agent:
    def __init__(self, env, start=None, r=np.random):
        # Store environment, state and action dimension
        self.r              = r
        self.env            = env
        self.state_dim      = env.state_dim
        self.max_action_dim = env.max_action_dim
        self.state          = start or env.randomState()

    def get_action(self, t):
        """Get the action to perform."""
        pass

    def train(self, memory):
        pass

    def reward(self, state, time):
        """Get the reward for an agent in a particular state of the world at a given time."""
        pass


class RandomAgent(Agent):
    def __init__(self, env, start=None, r=np.random):
        super(RandomAgent, self).__init__(env, start, r)

    def get_action(self, t):
        actions = np.array(self.env.actions_allowed(self.state))
        return choose(self.r, actions)

    def display_q_values(self):
        pass

class QAgent(RandomAgent):
    def __init__(self, env, start=None, r=np.random):
        super(QAgent, self).__init__(env, start, r)
        self.beta          = 0.98  # learning rate
        self.gamma         = 0.95  # reward discount factor

        # Initialize Q[s,a] table
        self.Q = {}
        for s in env.enumerateStates():
            for a in env.actions_allowed(s):
                self.Q[(s, tuple(a))] = 0

    def reward(self, state, time):
        print(state)
        return self.env[tuple(int(i) for i in state)]

    def train(self, memory):
        """
            Update Q values:

                Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])

            Where:

                R[s,a] = reward for taking action a from state s
                beta   = learning rate
                gamma  = discount factor
        """
        (state, action, state_next, reward, done) = memory
        max_next_value = np.max(list(self.Q[(state_next, a)] for a in self.env.actions_allowed(state_next)))
        sa = (state, action)
        self.Q[sa] += self.beta * (reward + self.gamma*max_next_value - self.Q[sa])

    def display_q_values(self):
        # greedy policy = argmax[a'] Q[s,a']
        for k, v in self.Q.items():
            s, a = k
            print("State: ", s, "; Action: ", a, "; Q-Value", v)

class GreedyAgent(QAgent):
    def __init__(self, env, start=None, r=np.random):
        super(GreedyAgent, self).__init__(env, start, r)

    def get_action(self, t):
        actions_allowed = self.env.actions_allowed(self.state)
        Q_s             = [self.Q[(self.state, a)] for a in actions_allowed]

        amax = np.flatnonzero(Q_s == np.max(Q_s))
        actions_allowed = np.array(actions_allowed)
        actions_greedy  = actions_allowed[amax]
        action = tuple(choose(self.r, actions_greedy))
        return action

class EpsilonGreedyAgent(QAgent):
    def __init__(self, env, eps=1.0, decay=1.0, start=None, r=np.random):
        super(EpsilonGreedyAgent, self).__init__(env, start, r)
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

class BoltzmannAgent(QAgent):
    def __init__(self, env, decay=1.0, min_temp = 0.1, start=None, r=np.random):
        super(BoltzmannAgent, self).__init__(env, start, r)
        self.decay = decay
        self.min_temp = min_temp
        self.temp = 1/min_temp

    def get_action(self, t):
        actions = self.env.actions_allowed(self.state)
        Qsa       = np.array([self.Q[(self.state, a)] for a in actions])
        probs     = np.exp(Qsa/self.temp)
        probs     = probs/np.sum(probs)
        choice    = sample(self.r, np.arange(len(probs)), probs)
        self.temp = max(self.temp*self.decay, self.min_temp)
        return actions[choice]
