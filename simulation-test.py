import numpy as np
import agent as a
import simulation as s
import grid_world as gw

if __name__ == "__main__":
    rs = np.random.RandomState(seed=1)
    # env = gw.Grid2D(3, 3, maxRange=10, r=rs)
    env = gw.OneWayGrid2D(10)
    print(env.exitAction)
    print("\n\n\nGreedyUCBAgent #################################################")
    rs = np.random.RandomState(seed=1)
    simulation = s.Simulation(a.GreedyUCBAgent(env, r=rs), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()

    print("Sequential Simulation:")
    simulation = s.Simulation(a.QAgent(env, r=rs), env)
    simulation.simulate_random(25, True, True)
    simulation.display_agent_Q()

    print("\n\n\nQAgent #################################################")
    rs = np.random.RandomState(seed=1)
    simulation = s.Simulation(a.QAgent(env, r=rs), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()

    print("Sequential Simulation:")
    simulation = s.Simulation(a.QAgent(env, r=rs), env)
    simulation.simulate_random(25, True, True)
    simulation.display_agent_Q()

    print("\n\n\nGreedyAgent #################################################")
    rs = np.random.RandomState(seed=1)
    agent = a.GreedyAgent(env, r=rs)
    simulation = s.Simulation(agent, env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()

    print("Sequential Simulation:")
    simulation = s.Simulation(a.GreedyAgent(env, r=rs), env)
    simulation.simulate_random(25, True, True)
    simulation.display_agent_Q()

    print("\n\n\nEpsilonGreedyAgent ##########################################")
    rs = np.random.RandomState(seed=1)
    simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=1.0, decay=.95), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()

    print("Sequential Simulation:")
    simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=1.0, decay=.95), env)
    simulation.simulate_random(25, True, True)
    simulation.display_agent_Q()

    print("\n\n\nBoltzmannAgent ##############################################")
    rs = np.random.RandomState(seed=1)
    simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
    simulation.simulate_random(100)
    simulation.display_agent_Q()

    print("Sequential Simulation:")
    simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
    simulation.simulate_random(25, True, True)
    simulation.display_agent_Q()
