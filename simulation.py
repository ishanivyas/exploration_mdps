import grid_world as gw

class Simulation():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        print("Here is the data of the environment: ")
        print(env.data)

    def simulate_random(self, T, sequential=False):
        # Set initial state.
        if sequential:
            next_state = self.agent.state

        for t_i in range(T):
            # Determine the state the agent begins in at this timestep.
            self.agent.state = next_state if sequential else self.env.randomState()

            # Get the agent's action at this timestep.
            a_i = self.agent.get_action(t_i)

            # Get the next state by transitioning in the environment
            next_state = self.env.transition(self.agent.state, a_i)

            # Get the reward for performing that action at this timestep.
            r_i = self.agent.reward(next_state, t_i)

            # Do one round of training.
            memory = (self.agent.state, a_i, next_state, r_i, False)
            print("State, Action, New State, Reward at t = %d:" % t_i, memory)
            self.agent.train(memory)

    def display_agent_Q(self):
        print("Here are the agent's final Q values:")
        self.agent.display_q_values()

    def simulate(self, memories):
        for memory in memories:
            agent.train(memory)
