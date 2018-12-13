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
        self.state          = env.randomState() if start == None else start

    def get_action(self, t):
        """Get the action to perform."""
        pass

    def train(self, memory):
        pass

    def reward(self, state, time):
        """Get the reward for an agent in a particular state of the world at a given time."""
        pass

# An agent that keeps no memory and uniformly randomly chooses actions.
class RandomAgent(Agent):
    def __init__(self, env, start=None, r=np.random):
        super(RandomAgent, self).__init__(env, start, r)

    def get_action(self, t):
        actions = np.array(self.env.actions_allowed(self.state))
        c = choose(self.r, actions)
        if c != self.env.exitAction:
            c = tuple(c)
        return c

    def display_q_values(self):
        pass

# An agent that uniformly randomly chooses actions, but also does Q value updates.
# It learns Q values, but ignores them and just plays available uniformly at random.
class QAgent(RandomAgent):
    def __init__(self, env, start=None, r=np.random, reward_prior=0):
        super(QAgent, self).__init__(env, start, r)
        self.beta          = 0.98  # learning rate
        self.gamma         = 0.5  # reward discount factor

        # Initialize Q[s,a] table and keep counts of how many times we perform each [s, a]
        self.Q      = {}
        self.counts = {}
        for s in env.enumerateStates():
            for a in env.actions_allowed(s):
                if a != env.exitAction:
                    a = tuple(a)
                self.Q[(s, a)] = reward_prior
                self.counts[(s, a)] = 0

    def reward(self, state, time):
        return self.env[tuple(int(i) for i in state)]

    def train(self, memory, verbose=False):
        """
            Update Q values:

                Q[s,a] <- Q[s,a] + beta * (R[s,a] + gamma * max(Q[s,:]) - Q[s,a])

            Where:

                R[s,a] = reward for taking action a from state s
                beta   = learning rate
                gamma  = discount factor
        """
        (state, action, state_next, reward, done) = memory
        max_next_value = 0
        if state_next != self.env.terminalState:
            max_next_value = np.max(list(self.Q[(state_next, a)] for a in self.env.actions_allowed(state_next)))
        sa = (state, action)
        oldq = self.Q[sa]
        self.Q[sa] += self.beta * (reward + self.gamma*max_next_value - self.Q[sa])
        if verbose:
            print("Q value for State:", state, "; Action:", action, "; went from", oldq, "to", self.Q[sa])
        self.counts[sa] += 1

    def display_q_values(self):
        # greedy policy = argmax[a'] Q[s,a']
        for k, v in self.Q.items():
            s, a = k
            print("State: ", s, "; Action: ", a, "; Q-Value", v, "; Visit Count: ", self.counts[k])
        return self.Q, self.counts

# An agent that does Q value updates and plays the best known action deterministically.
class GreedyAgent(QAgent):
    def __init__(self, env, start=None, r=np.random, reward_prior=0):
        super(GreedyAgent, self).__init__(env, start, r, reward_prior)

    def get_action(self, t):
        actions_allowed = self.env.actions_allowed(self.state)
        Q_s             = [self.Q[(self.state, a)] for a in actions_allowed]

        amax = np.flatnonzero(Q_s == np.max(Q_s))
        actions_allowed = np.array(actions_allowed)
        actions_greedy  = actions_allowed[amax]
        action = choose(self.r, actions_greedy)
        if action != self.env.exitAction:
            action = tuple(action)
        return action

# Plays randomly with probability `eps`, which decays over time.
class EpsilonGreedyAgent(GreedyAgent):
    def __init__(self, env, eps=1.0, decay=1.0, start=None, r=np.random, reward_prior=0):
        super(EpsilonGreedyAgent, self).__init__(env, start, r, reward_prior)
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

# Picks actions according to a soft-max-like distribution based on Q values.
class BoltzmannAgent(QAgent):
    def __init__(self, env, decay=1.0, min_temp = 0.1, start=None, r=np.random, reward_prior=0):
        super(BoltzmannAgent, self).__init__(env, start, r, reward_prior)
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

# An agent that gives Q value boosts to relatively unexplored states when selecting actions.
class GreedyUCBAgent(GreedyAgent):
    def __init__(self, env, alpha=10, start=None, r=np.random, reward_prior=0):
        super(GreedyUCBAgent, self).__init__(env, start, r)

        # Determines how large the confidence interval will be
        self.alpha = alpha 

    def get_action(self, t, verbose=False):
        actions_allowed = self.env.actions_allowed(self.state)
        Q_s = []
        delta = t ** -self.alpha
        for a in actions_allowed:
            q = self.Q[(self.state, a)]
            count = self.counts[(self.state, a)] + 0.01
            q_boosted = q + np.sqrt(8*np.log(1/delta) / count)
            if verbose:
                print("Count = %d: q was boosted from" % count, q, "to", q_boosted)
            Q_s.append(q_boosted)

        amax = np.flatnonzero(Q_s == np.max(Q_s))
        actions_allowed = np.array(actions_allowed)
        actions_greedy  = actions_allowed[amax]
        action = choose(self.r, actions_greedy)
        if action != self.env.exitAction:
            action = tuple(action)
        return action