import agent as a
import simulation as s
import grid_world as gw

if __name__ == "__main__":
    env = gw.Grid2D(3, 3, 10)
    simulation = s.Simulation(a.EpsilonGreedyAgent(env), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()
