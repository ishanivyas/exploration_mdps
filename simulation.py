import grid_world as gw
import numpy as np
class Simulation():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.cumulative_reward = 0
        print("Here is the data of the environment: ")
        print(env.data)


    def simulate_random(self, T, sequential=False, verbose=False):
        # Set initial state.
        next_state = self.agent.state

        for t_i in range(1, T + 1):
            # Determine the state the agent begins in at this timestep.
            self.agent.state = next_state if sequential and next_state != self.env.terminalState \
                else self.env.randomState()
            if next_state == self.env.terminalState:
                self.agent.state = (np.random.randint(3), 0)

            # Get the agent's action at this timestep.
            a_i = self.agent.get_action(t_i)

            # Get the next state and reward by transitioning in the environment
            next_state, r_i = self.env.transition(self.agent.state, a_i)
            self.cumulative_reward += r_i
            if next_state == self.env.terminalState and verbose:
                print("Reached terminal state.")
            # Do one round of training.
            memory = (self.agent.state, a_i, next_state, r_i, False)

            if verbose:
                print("State, Action, New State, Reward at t = %d: %s" % (t_i, memory))
            self.agent.train(memory, verbose)
        print("Cumulative reward after training was", self.cumulative_reward)
        return self.cumulative_reward

    def display_agent_Q(self):
        print("Here are the agent's final Q values:")
        return self.agent.display_q_values()

    def simulate(self, memories):
        for memory in memories:
            agent.train(memory)
