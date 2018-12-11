import numpy as np
import agent as a
import simulation as s
import grid_world as gw

def MSE(qs, ground_truth_q):
	MSE = 0
	for sa, qv in qs.items():
		true_q = ground_truth_q[sa]
		MSE += (qv - true_q)**2
	return MSE / len(qs)

if __name__ == "__main__":
    rs = np.random.RandomState(seed=1)
    # env = gw.Grid2D(3, 3, maxRange=10, r=rs)
    env = gw.OneWayGrid2D(10)

    print("\n\n\nQAgent #################################################")
    rs = np.random.RandomState(seed=1)
    simulation = s.Simulation(a.QAgent(env, r=rs), env)
    simulation.simulate_random(100000)
    ground_truth_q, blah = simulation.display_agent_Q()

    # print("Sequential Simulation:")
    # simulation = s.Simulation(a.QAgent(env, r=rs), env)
    # simulation.simulate_random(25, True, True)
    # simulation.display_agent_Q()

    print("\n\n\nGreedyUCBAgent #################################################")
    rs = np.random.RandomState(seed=1)
    # simulation = s.Simulation(a.GreedyUCBAgent(env, r=rs), env)
    # simulation.simulate_random(100)
    # simulation.display_agent_Q()

    # print("Sequential Simulation:")
    simulation = s.Simulation(a.GreedyUCBAgent(env, r=rs), env)
    ucb_cumulative_reward = simulation.simulate_random(100, sequential=True)
    ucb_q, ucb_counts = simulation.display_agent_Q()

    print("\n\n\nGreedyAgent #################################################")
    rs = np.random.RandomState(seed=1)
    # # agent = a.GreedyAgent(env, r=rs)
    # # simulation = s.Simulation(agent, env)
    # # simulation.simulate_random(100)
    # # simulation.display_agent_Q()

    # print("Sequential Simulation:")
    simulation = s.Simulation(a.GreedyAgent(env, r=rs), env)
    greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
    greedy_q, greedy_counts = simulation.display_agent_Q()

    

    print("\n\n\nEpsilonGreedyAgent ##########################################")
    rs = np.random.RandomState(seed=1)
    # # simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=1.0, decay=.95), env)
    # # simulation.simulate_random(100)
    # # simulation.display_agent_Q()

    # print("Sequential Simulation:")
    simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=1.0, decay=.95), env)
    epsilon_greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
    eps_greedy_q, eps_greedy_counts = simulation.display_agent_Q()


    print("\n\n\nBoltzmannAgent ##############################################")
    rs = np.random.RandomState(seed=1)
    # # simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
    # # simulation.simulate_random(100)
    # # simulation.display_agent_Q()

    # print("Sequential Simulation:")
    simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
    boltzmann_cumulative_reward = simulation.simulate_random(100, sequential=True)
    boltz_q, boltz_counts = simulation.display_agent_Q()

    print("Here are the Mean Squared Differences between Q values after 100 transitions and the converged values")
    m = MSE(ucb_q, ground_truth_q)
    print("MSE for UCB agent was ", m)
    print("Cumulative reward was ", ucb_cumulative_reward)
    print("")

    m = MSE(greedy_q, ground_truth_q)
    print("MSE for Greedy agent was ", m)
    print("Cumulative reward was ", greedy_cumulative_reward)
    print("")

    m = MSE(eps_greedy_q, ground_truth_q)
    print("MSE for Epsilon Greedy agent was ", m)
    print("Cumulative reward was ", epsilon_greedy_cumulative_reward)
    print("")

    m = MSE(boltz_q, ground_truth_q)
    print("MSE for Boltzmann agent was ", m)
    print("Cumulative reward was ", boltzmann_cumulative_reward)
    print("")
