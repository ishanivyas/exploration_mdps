import numpy as np
import agent as a
import simulation as s
import grid_world as gw

if __name__ == "__main__":
    rs = np.random.RandomState(seed=1)
    env = gw.Grid2D(3, 3, maxRange=10, r=rs)
    simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()
