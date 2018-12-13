import numpy as np
import agent as a
import simulation as s
import grid_world as gw
import os
import copy

def MSE(qs, ground_truth_q):
	MSE = 0
	for sa, qv in qs.items():
		true_q = ground_truth_q[sa]
		MSE += (qv - true_q)**2
	return MSE / len(qs)

def confidence_interval_90(lst):
	_sorted = sorted(lst)
	lower_bound = _sorted[len(lst) // 20]
	upper_bound = _sorted[len(lst)*19 // 20]
	median = _sorted[len(lst) // 2]
	# print("90% Confidence Interval and Median: ", (lower_bound, median, upper_bound))
	return lower_bound, median, upper_bound

def dump(s, a, agent_results):
	print("Confidence intervals for State:", s, "; Action:", a)
	conf_int = []
	counts = []
	for i in range(len(agent_results)):
		valuez, countz = agent_results[i][(s, a)]
		c1 = confidence_interval_90(valuez)
		conf_int.append(tuple(c1))
		c2 = confidence_interval_90(countz)
		counts.append(tuple(c2))
	print(conf_int)
	print(counts)
	print("#################################")
	print("")

if __name__ == "__main__":
	rs = np.random.RandomState(seed=1)
	# env = gw.Grid2D(3, 3, maxRange=10, r=rs)
	env = gw.OneWayGrid2D(10)
	ground_truth_conf = {}
	for st in env.enumerateStates():
		for ac in env.actions_allowed(st):
			if ac != env.exitAction:
				ac = tuple(ac)
			ground_truth_conf[(st, ac)] = ([], [])
	ground_truth_conf["cumulative"] = []
	ground_truth_conf["MSE"] = []

	uniform_conf = copy.deepcopy(ground_truth_conf)
	greedy_conf = copy.deepcopy(ground_truth_conf)
	eps25_greedy_conf = copy.deepcopy(ground_truth_conf)
	eps50_greedy_conf = copy.deepcopy(ground_truth_conf)
	eps75_greedy_conf = copy.deepcopy(ground_truth_conf)
	bolzmann_no_decay_conf = copy.deepcopy(ground_truth_conf)
	bolzmann_decay_conf = copy.deepcopy(ground_truth_conf)
	ucb5_conf = copy.deepcopy(ground_truth_conf)
	ucb10_conf = copy.deepcopy(ground_truth_conf)

	agent_results = [ground_truth_conf, uniform_conf, greedy_conf, eps25_greedy_conf, eps50_greedy_conf, \
		eps75_greedy_conf, bolzmann_no_decay_conf, bolzmann_decay_conf, ucb5_conf, ucb10_conf]
	print("\n\n\nGroundTruth #################################################")
	rs = np.random.RandomState(seed=1)
	simulation = s.Simulation(a.QAgent(env, r=rs), env)
	simulation.simulate_random(100000)
	ground_truth_q, ground_truth_count = simulation.display_agent_Q()
	for k, v in ground_truth_q.items():
		ground_truth_conf[k][0].append(v)
		ground_truth_conf[k][1].append(ground_truth_count[k])
	ground_truth_conf["cumulative"].append(0)
	ground_truth_conf["MSE"].append(0)
	
	for i in range(100):
		print("Iteration: %d" % i)
		print("\n\n\nQAgent #################################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.QAgent(env, r=rs), env)
		uniform_cumulative_reward = simulation.simulate_random(100, sequential=True)
		uniform, uniform_count = simulation.agent.Q, simulation.agent.counts
		for k, v in uniform.items():
			uniform_conf[k][0].append(v)
			uniform_conf[k][1].append(uniform_count[k])
		uniform_conf["cumulative"].append(uniform_cumulative_reward)
		uniform_conf["MSE"].append(MSE(uniform, ground_truth_q))

		print("\n\n\nUCB5Agent #################################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.GreedyUCBAgent(env, alpha=5, r=rs), env)
		ucb_cumulative_reward = simulation.simulate_random(100, sequential=True)
		ucb_q, ucb_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in ucb_q.items():
			ucb5_conf[k][0].append(v)
			ucb5_conf[k][1].append(ucb_counts[k])
		ucb5_conf["cumulative"].append(ucb_cumulative_reward)
		ucb5_conf["MSE"].append(MSE(ucb_q, ground_truth_q))

		print("\n\n\nUCB10Agent #################################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.GreedyUCBAgent(env, r=rs), env)
		ucb_cumulative_reward = simulation.simulate_random(100, sequential=True)
		ucb_q, ucb_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in ucb_q.items():
			ucb10_conf[k][0].append(v)
			ucb10_conf[k][1].append(ucb_counts[k])
		ucb10_conf["cumulative"].append(ucb_cumulative_reward)
		ucb10_conf["MSE"].append(MSE(ucb_q, ground_truth_q))

		print("\n\n\nGreedyAgent #################################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.GreedyAgent(env, r=rs), env)
		greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
		greedy_q, greedy_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in greedy_q.items():
			greedy_conf[k][0].append(v)
			greedy_conf[k][1].append(greedy_counts[k])
		greedy_conf["cumulative"].append(greedy_cumulative_reward)
		greedy_conf["MSE"].append(MSE(greedy_q, ground_truth_q))

		print("\n\n\nEpsilonGreedyAgent25 ##########################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=.25, decay=.95), env)
		epsilon_greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
		eps_greedy_q, eps_greedy_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in eps_greedy_q.items():
			eps25_greedy_conf[k][0].append(v)
			eps25_greedy_conf[k][1].append(eps_greedy_counts[k])
		eps25_greedy_conf["cumulative"].append(epsilon_greedy_cumulative_reward)
		eps25_greedy_conf["MSE"].append(MSE(eps_greedy_q, ground_truth_q))

		print("\n\n\nEpsilonGreedyAgent50 ##########################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=.50, decay=.95), env)
		epsilon_greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
		eps_greedy_q, eps_greedy_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in eps_greedy_q.items():
			eps50_greedy_conf[k][0].append(v)
			eps50_greedy_conf[k][1].append(eps_greedy_counts[k])
		eps50_greedy_conf["cumulative"].append(epsilon_greedy_cumulative_reward)
		eps50_greedy_conf["MSE"].append(MSE(eps_greedy_q, ground_truth_q))

		print("\n\n\nEpsilonGreedyAgent75 ##########################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.EpsilonGreedyAgent(env, r=rs, eps=.75, decay=.95), env)
		epsilon_greedy_cumulative_reward = simulation.simulate_random(100, sequential=True)
		eps_greedy_q, eps_greedy_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in eps_greedy_q.items():
			eps75_greedy_conf[k][0].append(v)
			eps75_greedy_conf[k][1].append(eps_greedy_counts[k])
		eps75_greedy_conf["cumulative"].append(epsilon_greedy_cumulative_reward)
		eps75_greedy_conf["MSE"].append(MSE(eps_greedy_q, ground_truth_q))

		print("\n\n\nBoltzmannAgentNoDecay ##############################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
		boltzmann_cumulative_reward = simulation.simulate_random(100, sequential=True)
		boltz_q, boltz_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in boltz_q.items():
			bolzmann_no_decay_conf[k][0].append(v)
			bolzmann_no_decay_conf[k][1].append(boltz_counts[k])
		bolzmann_no_decay_conf["cumulative"].append(boltzmann_cumulative_reward)
		bolzmann_no_decay_conf["MSE"].append(MSE(boltz_q, ground_truth_q))

		print("\n\n\nBoltzmannAgentNoDecay ##############################################")
		rs = np.random.RandomState(seed=1)
		simulation = s.Simulation(a.BoltzmannAgent(env, r=rs), env)
		simulation.agent.decay = 0.95
		boltzmann_cumulative_reward = simulation.simulate_random(100, sequential=True)
		boltz_q, boltz_counts = simulation.agent.Q, simulation.agent.counts
		for k, v in boltz_q.items():
			bolzmann_decay_conf[k][0].append(v)
			bolzmann_decay_conf[k][1].append(boltz_counts[k])
		bolzmann_decay_conf["cumulative"].append(boltzmann_cumulative_reward)
		bolzmann_decay_conf["MSE"].append(MSE(boltz_q, ground_truth_q))

	print("Done, finishing")
	for s, a in uniform.keys():
		dump(s, a, agent_results)

	print("\nUniform")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(uniform_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(uniform_conf["MSE"]))

	print("\nUCB alpha=5")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(ucb5_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(ucb5_conf["MSE"]))

	print("\nUCB alpha=10")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(ucb10_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(ucb10_conf["MSE"]))

	print("\nGreedy")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(greedy_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(greedy_conf["MSE"]))

	print("\nEpsilon Greedy 0.25")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(eps25_greedy_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(eps25_greedy_conf["MSE"]))

	print("\nEpsilon Greedy 0.50")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(eps50_greedy_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(eps50_greedy_conf["MSE"]))

	print("\nEpsilon Greedy 0.75")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(eps75_greedy_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(eps75_greedy_conf["MSE"]))

	print("\nBoltzmann Without Decay")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(bolzmann_no_decay_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(bolzmann_no_decay_conf["MSE"]))

	print("\nBoltzmann With Decay = 0.95")
	print("Confidence interval of cumulative reward:")
	print(confidence_interval_90(bolzmann_decay_conf["cumulative"]))
	print("Confidence interval of MSE:")
	print(confidence_interval_90(bolzmann_decay_conf["MSE"]))