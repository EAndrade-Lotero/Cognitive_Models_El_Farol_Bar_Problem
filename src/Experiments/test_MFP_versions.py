import numpy as np
from time import time
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from random import randint, seed, choice

from Classes.bar import Bar
from Classes.agents import AgentMFP_Multi, AgentMFPMultiSameTransProb, epsilon_greedy
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from Classes.agent_utils import ProxyDict, TransitionsFrequencyMatrix


def test():
	# Define simulation parameters
	num_agents = 2
	threshold = .5
	num_rounds = 100
	epsilon = 0
	belief_strength = 0.1
	# states = StatesContainer(num_agents)
	states = list(product([0,1], repeat=num_agents))
	alphas = {(x,y):1/len(states) for x in states for y in states}
	# Parameters to define old agents
	parameters = {
		"belief_strength":belief_strength,\
		"alphas":alphas,\
		"num_agents":num_agents,\
		"threshold":threshold,\
		"epsilon":epsilon,
	}
	# Define old agents
	# agents_old = [epsilon_greedy(parameters, n) for n in range(num_agents)]
	agents_old = [AgentMFP_Multi(parameters, n) for n in range(num_agents)]
	# Parameters to define new agents
	# trans_probs = TransitionsFrequencyMatrix(num_agents)
	# count_states = BinaryCounter1D(num_agents)
	# count_transitions = BinaryCounter2D(num_agents)
	# count_states = {state:0 for state in states}
	# count_transitions = {(prev_s,new_s):0 for prev_s in states for new_s in states}
	# trans_probs = deepcopy(alphas)
	count_states = ProxyDict(
		keys=states,
		initial_val=0
	)
	count_transitions = ProxyDict(
		keys=list(product(states, repeat=2)),
		initial_val=0
	)
	trans_probs = ProxyDict(
		keys=list(product(states, repeat=2)),
		initial_val=1/len(states)
	)
	parameters = {
		"belief_strength":belief_strength,\
		"alphas":alphas,\
		"num_agents":num_agents,\
		"threshold":threshold,\
		"epsilon":epsilon,
		"trans_probs":trans_probs,
		"count_states":count_states,
		"count_transitions":count_transitions,
		"states":states,
		"designated_agent":False
	}
	# Define new agents
	# agents_new = [AgentMFPMultiSameTransProb(parameters, n) for n in range(num_agents)]
	agents_new = [epsilon_greedy(parameters, n) for n in range(num_agents)]
	agents_new[0].designated_agent = True
	# Create bar
	bar = Bar(num_agents=num_agents, threshold=threshold)
	# Create simulation
	episode = Episode(
		environment=bar,\
		agents=agents_old,\
		model='MFP',\
		num_rounds=num_rounds
	)
	list_attendances = list()
	# print('')
	# print('-'*50)
	# print('Agents backup')
	# print('-'*50)
	for _ in range(num_rounds):
		# attendances = states.choice()
		attendances = choice(states)
		list_attendances.append(attendances)
	old_eus = list()
	start_time = time()
	for k, attendances in enumerate(list_attendances):
		if k == int(num_rounds / 2):
			time_to_reset = True
		else:
			time_to_reset = False
		attendance, scores = bar.step(attendances)
		# print(f'\tAttendance = {attendance}\n')
		for i, agent in enumerate(agents_old):
			if time_to_reset:
				agent.reset()
			score = scores[i]
			# Learning rule is applied
			agent.update(score, attendances)
			# decision = attendances[i]
			# decision = agent.make_decision()
			eus = tuple([agent.exp_util(attendances, a) for a in range(2)])
			old_eus.append(eus)
			# decision_ = 'go' if decision == 1 else 'no go'
			# print(f'\t\tAgent_{i} => {decision_}')
			# agent.print_agent()
	# 		print('')		
	end_time = time()
	time_old = end_time - start_time
	# print('')
	# print('-'*50)
	# print('Agents new')
	# print('-'*50)
	new_eus = list()
	start_time = time()
	for k, attendances in enumerate(list_attendances):
		if k == int(num_rounds / 2):
			time_to_reset = True
		else:
			time_to_reset = False
		attendance, scores = bar.step(attendances)
		# print(f'\tAttendance = {attendance}\n')
		for i, agent in enumerate(agents_new):
			if time_to_reset:
				agent.reset()
			score = scores[i]
			# Learning rule is applied
			agent.update(score, attendances)
			# decision = attendances[i]
			# decision = agent.make_decision()
			eus = tuple([agent.exp_util(attendances, a) for a in range(2)])
			new_eus.append(eus)
			# decision_ = 'go' if decision == 1 else 'no go'
			# print(f'\t\tAgent_{i} => {decision_}')
			# agent.print_agent()
			# print('')		
	end_time = time()
	time_new = end_time - start_time
	old_eus = np.array(old_eus)
	new_eus = np.array(new_eus)
	# print(old_eus)
	# print(new_eus)
	print(np.where(old_eus != new_eus))
	A = TransitionsFrequencyMatrix(num_agents)
	A.from_dict(agents_old[0].trans_probs)
	B = TransitionsFrequencyMatrix(num_agents)	
	B.from_proxydict(agents_new[1].trans_probs)
	# A = agents_old[0].trans_probs
	# B = agents_new[1].trans_probs
	table_A = PrettyTable([''] + [str(s) for s in states])
	table_B = PrettyTable([''] + [str(s) for s in states])
	for prev_state in states:
		dummies = [round(A((prev_state, state)),2) for state in states]
		table_A.add_row([str(prev_state)] + dummies)
		dummies = [round(B((prev_state, state)),2) for state in states]
		table_B.add_row([str(prev_state)] + dummies)
		# dummies = [round(A[(prev_state, state)],2) for state in states]
		# table_A.add_row([str(prev_state)] + dummies)
		# dummies = [round(B[(prev_state, state)],2) for state in states]
		# table_B.add_row([str(prev_state)] + dummies)
	print(table_A)
	print(table_B)
	time_dif_percentage = (time_new / time_old) * 100
	time_speedup = 100 - time_dif_percentage
	print(f'Time speedup new vs old: {round(time_speedup,2)}%')
	positions_different = np.where(A.trans_freqs != B.trans_freqs)
	print(positions_different)
	pos_dif_percentage = (len(positions_different[0]) / (num_agents ** 4)) * 100
	print(f'Difference in content new vs old: {round(pos_dif_percentage,2)}%')
