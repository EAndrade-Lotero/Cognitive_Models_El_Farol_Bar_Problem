'''
Classes with agents' rules
'''
import torch
import numpy as np
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from random import randint, uniform, choice
from typing import Optional, Union, Dict, List

from Classes.agent_utils import MLP

DICT_STATES = {
	(0,0):0,
	(0,1):1,
	(1,0):2,
	(1,1):3
}

class Agent :
	'''
	Defines the basic methods for a best response agent.
	'''

	def __init__(
				self, 
				parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		self.parameters = parameters
		assert('threshold' in parameters.keys())
		self.threshold = parameters['threshold']
		self.epsilon = parameters['epsilon']
		self.decisions = []
		self.scores = []
		self.number = n
		self.prev_state_ = None
		if self.epsilon is not None:
			self.cool_down = False
		else:
			self.cool_down = True
		self.turn = -1

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		self.turn += 1
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			best_action = self.greedy_action(self.prev_state_)
			# Include exploration with a factor of epsilon
			if self.cool_down:
				epsilon = self.get_epsilon_from_cool_down()
			else:
				epsilon = self.epsilon
			if epsilon is not None:
				if uniform(0, 1) < epsilon:
					other_action = 1 - best_action
					# return other_action
					return randint(0, 1)
				else:
					return best_action
			else:
				raise Exception('Code error: epsilon should not be None!')
		else:
			# no previous data, so make random decision
			return randint(0, 1)

	def greedy_action(self, prev_state:tuple) -> int:
		'''
		Returns the action with higher expected utility.
		Break ties uniformly.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
		Output:
			- a decision 0 or 1
		'''
		eus = [self.exp_util(prev_state, action) for action in [0,1]]
		max_eu = max(eus)
		max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
		return choice(max_actions)

	def go_probability(self) -> float:
		'''
		Agent returns the probability of going to the bar
		according to its model.

		Output:
			- p, float representing the probability that the
				 agent goes to the bar.
		'''
		# Check if agent recalls previous round
		if self.prev_state_ is not None:
			# Obtain expected utility of go and no go
			eus = [self.exp_util(self.prev_state, action) for action in [0,1]]
			if eus[1] > eus[0]:
				# Return 1 if go has higher expected utility
				return 1
			elif eus[1] < eus[0]:
				# Return 0 if go has lower expected utility
				return 0
			else:
				# Return 0.5 for breaking ties randomly
				return 0.5
		else:
			# Agent does not recall previous round, so choice is random
			return 0.5

	def payoff(self, action:int, partners:List[int]) -> int:
		'''
		Returns the payoff according to the El Farol bar problem payoff matrix
		'''
		if action == 0:
			return 0
		elif sum(partners) + 1 <= self.threshold * self.num_agents:
			return 1
		else:
			return -1

	def reset(self) -> None:
		'''
		Restarts the agent's data for a new trial.
		'''
		self.decisions = []
		self.scores = []
		if self.epsilon is not None:
			self.cool_down = False
		else:
			self.cool_down = True
		self.turn = -1

	def exp_util(self, prev_state:List[int], action:int) -> float:
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		# To be defined by subclass
		pass

	def update(self, score:int, obs_state_:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						 where each argument is 0 or 1.
		'''
		# To be defined by subclass
		pass

	def print_agent(self, ronda:Optional[Union[int, None]]=None) -> str:
		'''
		Returns a string with the state of the agent on a given round.
		Input:
			- ronda, integer with the number of the round.
		Output:
			- string with a representation of the agent at given round.
		'''
		# To be defined by subclass
		pass

	def get_epsilon_from_cool_down(self) -> float:
		round = self.turn
		# Check value of epsilong accoring to cooling down protocol.
		if round < 50:
			epsilon = 0.5
		elif round < 100:
			epsilon = 0.3
		elif round < 150:
			epsilon = 0.1
		elif round < 200:
			epsilon = 0.05
		elif round < 250:
			epsilon = 0.01
		else:
			epsilon = 0
		return epsilon



class Agent_BKUP :
	'''
	Defines the basic methods for each agent.
	'''

	def __init__(self, parameters:Dict[str, any], n:int):
		self.parameters = parameters
		assert('threshold' in parameters.keys())
		self.threshold = parameters['threshold']
		self.decisions = []
		self.scores = []
		self.number = n

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# To be defined by subclass
		pass

	def greedy_action(self, prev_state:tuple) -> int:
		'''
		Returns the action with higher expected utility.
		Break ties uniformly.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
		Output:
			- a decision 0 or 1
		'''
		eus = [self.exp_util(prev_state, action) for action in [0,1]]
		max_eu = max(eus)
		max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
		return choice(max_actions)

	def payoff(self, action:int, partners:List[int]) -> int:
		'''
		Returns the payoff according to the El Farol bar problem payoff matrix
		'''
		if action == 0:
			return 0
		elif sum(partners) + 1 <= self.threshold * self.num_agents:
			return 1
		else:
			return -1

	def update(self, score:int, obs_state_:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		# To be defined by subclass
		pass

	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent goes to the bar.
		'''
		# To be defined by subclass
		pass

	def reset(self):
		'''
		Restarts the agent's data for a new trial.
		'''
		self.decisions = []
		self.scores = []
		
	def print_agent(self, ronda:int=None) -> str:
			'''
			Returns a string with the state of the agent on a given round.
			Input:
				- ronda, integer with the number of the round.
			Output:
				- string with a representation of the agent at given round.
			'''
			if ronda is None:
				try:
					ronda = len(self.decisions) - 1
				except:
					ronda = 0
			try:
				decision = self.decisions[ronda]
			except:
				decision = "nan"
			try:
				score = self.scores[ronda]
			except:
				score = "nan"
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}")



class Random(Agent) :
	'''
	Implements a random rule of go/no go with equal probability.
	'''

	def __init__(
				self, 
				parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		self.p = parameters["go_probability"]
		self.number = n
		self.decisions = []
		self.scores = []

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		return 1 if uniform(0, 1) < self.p else 0
	
	def update(
				self, 
				score:int, 
				obs_state_:tuple
			) -> None:
		self.decisions.append(obs_state_[self.number])
		self.scores.append(score)

	def go_probability(self) -> float:
		'''
		Agent returns the probability of going to the bar
		according to its model.

		Output:
			- p, float representing the probability that the
				 agent goes to the bar.
		'''
		return self.p

	def print_agent(self, ronda:int=None) -> str:
			'''
			Returns a string with the state of the agent on a given round.
			Input:
				- ronda, integer with the number of the round.
			Output:
				- string with a representation of the agent at given round.
			'''
			if ronda is None:
				try:
					ronda = len(self.decisions) - 1
				except:
					ronda = 0
			try:
				decision = self.decisions[ronda]
			except:
				decision = "nan"
			try:
				score = self.scores[ronda]
			except:
				score = "nan"
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}")

	def reset(self):
		self.decisions = []
		self.scores = []



class AgentMFP_Multi(Agent) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule for multiple players.
	It uses the following parameters:
		* alphas
		* num_agents
		* threshold
		* belief_strength
	It also requires an id_number n
	'''

	def __init__(
				self, 
				parameters: Dict[str,any], 
				n: int
			) -> None:
		super().__init__(parameters, n)
		assert(parameters["num_agents"] is not None)
		self.num_agents = parameters["num_agents"]
		self.states = list(product([0,1], repeat=self.num_agents))
		assert(parameters["alphas"] is not None)
		self.alphas = deepcopy(parameters["alphas"])
		assert(parameters["belief_strength"] is not None)
		self.belief_strength = parameters["belief_strength"]
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(parameters["alphas"])
		# self.trans_probs = parameters['trans_probs']
		# self.count_states = parameters['count_states']
		# self.count_transitions = parameters['count_transitions']
		# self.convergence = [max([value for key, value in self.trans_probs.items()])]
		# self.convergence = trans_probs.get_convergence()
		self.designated_agent = parameters['designated_agent']
		self.prev_state_ = None

	def update(self, score:int, obs_state:tuple) -> None:
		'''
		Agent updates its model using the Markov Fictitious Play rule.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						 where each argument is 0 or 1.
		Input:
		'''
		if isinstance(obs_state, list):
			obs_state = tuple(obs_state)
		# Update records
		self.scores.append(score)
		assert(isinstance(self.number, int)), f'=>{self.number}'
		self.decisions.append(obs_state[self.number])
		if self.designated_agent:
			# # Register to calculate convergence
			# trans_probs = deepcopy(self.trans_probs)
			# Agent recalls previous state?
			if self.prev_state_ is not None:
				prev_state = self.prev_state_
				# Update transtion counts
				observed_transition = (tuple(prev_state), tuple(obs_state))
				# self.count_transitions.update(observed_transition)
				self.count_transitions[observed_transition] += 1
				# Loop over states and update transition probabilities
				# print('='*50)
				for new_state in self.states:
					transition = (tuple(prev_state), tuple(new_state))
					# numerator = self.count_transitions(transition) + self.belief_strength * self.alphas[transition]
					# denominator = self.count_states(tuple(prev_state)) + self.belief_strength
					# new_prob = numerator / denominator
					# assert(0 <= new_prob <= 1), f'\nTransition:{transition}\nTransition counts:{self.count_transitions(transition)}\nState counts:{self.count_states(tuple(prev_state))}'
					# print(f'agente {self.number} --- transición {prev_state}=>{new_state} --- pasa de {round(self.trans_probs(transition))} a {round(new_prob,2)}')
					# self.trans_probs.update(transition, new_prob)
					numerator = self.count_transitions[transition] + self.belief_strength * self.alphas[transition]
					denominator = self.count_states[tuple(prev_state)] + self.belief_strength
					new_prob = numerator / denominator
					assert(0 <= new_prob <= 1), f'\nTransition:{transition}\nTransition counts:{self.count_transitions[transition]}\nState counts:{self.count_states[tuple(prev_state)]}'
					# # print(f'agente {self.number} --- transición {prev_state}=>{new_state} --- pasa de {round(self.trans_probs(transition))} a {round(new_prob,2)}')
					self.trans_probs[transition] = new_prob
			# Update state counts
			# self.count_states.update(tuple(obs_state))
			self.count_states[tuple(obs_state)] += 1
		# Update previous state
		self.prev_state_ = obs_state
		# # Update convergence
		# differences = np.abs([trans_probs[k] - self.trans_probs[k] for k in trans_probs.keys()])
		# self.convergence.append(max(differences))

	def reset(self) -> None:
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None
		if self.designated_agent:
			# self.count_states.reset()
			# self.count_transitions.reset()
			# self.trans_probs.reset()
			self.trans_probs = deepcopy(self.alphas)
			self.count_states = {state:0 for state in self.states}
			self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}

	def exp_util(self, prev_state:List[int], action:int) -> float:
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		eu = 0
		list_partners = list(product([0,1], repeat = self.num_agents-1))
		# print(list_partners)
		# print(f'Agente {self.number}')
		# print(f'prev_state:{prev_state} ---- accion:{action}')
		for partners_ in list_partners:
			partners = list(partners_)
			state = partners[:self.number] + [action] + partners[self.number:]
			# print('state -->', state)
			v = self.payoff(action, partners)
			# p = self.trans_probs((tuple(prev_state), tuple(state)))
			p = self.trans_probs[(tuple(prev_state), tuple(state))]
			# print(f'accion:{action} --- state:{state} ---- utilidad:{v} --- probabilidad:{p}')
			eu += v*p
		# print(f'Expected utility of action {action} is {eu}')
		return eu

	def print_agent(self, ronda:Optional[Union[int, None]]=None) -> str:
		'''
		Returns a string with the state of the agent on a given round.
		Input:
			- ronda, integer with the number of the round.
		Output:
			- string with a representation of the agent at given round.
		'''
		if ronda is None:
			try:
				ronda = len(self.decisions) - 1
			except:
				ronda = 0
		try:
			decision = self.decisions[ronda]
		except:
			decision = "nan"
		try:
			score = self.scores[ronda]
		except:
			score = "nan"
		states = self.states
		if self.designated_agent:
			table = PrettyTable([''] + [str(s) for s in states])
			for prev_state in states:
				dummies = [round(self.trans_probs((prev_state, state)),2) for state in states]
				table.add_row([str(prev_state)] + dummies)
			probs = str(table)
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")
		else:
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}")

	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.

		Output:
			- p, float representing the probability that the
					agent goes to the bar.
		'''
		# Check if agent recalls previous round
		if self.prev_state_ is not None:
			# Obtain expected utility of go and no go
			eus = [self.exp_util(self.prev_state, action) for action in [0,1]]
			if eus[1] > eus[0]:
				# Return 1 - epsilon if go has higher expected utility
				return 1 - self.epsilon
			elif eus[1] < eus[0]:
				# Return epsilon if go has lower expected utility
				return self.epsilon
			else:
				# Return 0.5 for breaking ties randomly
				return 0.5
		else:
			# Agent does not recall previous round, so choice is random
			return 0.5



class AgentNN(AgentMFP_Multi) :
	'''
	Implements an agent using an approximation function
	based on a Neural Network
	'''
	def __init__(
				self, 
				parameters: Dict[str,any], 
				n: int,
				designated_agent: bool,
			) -> None:
		super().__init__(parameters, n, designated_agent)

	def exp_util(self, prev_state: List[int], action: int) -> float:
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		# print(f'prev_state:{prev_state} ---- accion:{action}')
		eu = 0
		list_partners = list(product([0,1], repeat = self.num_agents-1))
		# print(list_partners)
		for partners_ in list_partners:
			partners = list(partners_)
			state = partners[:self.number] + [action] + partners[self.number:]
			v = self.payoff(action, partners)
			p = np.dot(self.trans_probs.values_vector(prev_state), state)
			# print(f'state:{state} ---- utilidad:{v} --- probabilidad:{p}')
			eu += v * p
		# print(f'Expected utility of action {action} is {eu}')
		return eu

	def update(self, score:int, obs_state:List[int]) -> None:
		'''
		Agent updates its neural network.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						 where each argument is 0 or 1.
		'''
		# Update records
		self.scores.append(score)
		self.decisions.append(obs_state[self.number])
		if self.designated_agent:
			# # Register to calculate convergence
			# trans_probs = deepcopy(self.trans_probs)
			# Agent recalls previous state?
			if self.prev_state_ is not None:
				prev_state = self.prev_state_
				# Update transtion counts
				self.trans_probs.update(prev_state, obs_state)
		# Update previous state
		self.prev_state_ = obs_state
		# # Update convergence
		# differences = np.abs([trans_probs[k] - self.trans_probs[k] for k in trans_probs.keys()])
		# self.convergence.append(max(differences))

	def reset(self) -> None:
		'''
		Restarts the agent's data for a new trial.
		'''
		self.decisions = []
		self.scores = []
		self.prev_state_ = None
		if self.designated_agent:
			self.trans_probs.reset()

	def print_agent(self, ronda:Optional[Union[int, None]]=None) -> str:
		'''
		Returns a string with the state of the agent on a given round.
		Input:
			- ronda, integer with the number of the round.
		Output:
			- string with a representation of the agent at given round.
		'''
		if ronda is None:
			try:
				ronda = len(self.decisions) - 1
			except:
				ronda = 0
		try:
			decision = self.decisions[ronda]
		except:
			decision = "nan"
		try:
			score = self.scores[ronda]
		except:
			score = "nan"
		states = self.states
		if self.designated_agent:
			probs = str(self.trans_probs)
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")
		else:
			print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}")


	def exp_util_DEPRECATED(self, prev_state:List[int], action:int) -> float:
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		# Get probabilities of aggregate state and action
		probabilities = self.NN(prev_state)
		eu = 0
		for aggregated_partners in [0,1]:
			state = (action, aggregated_partners)
			v = self.payoff[action, aggregated_partners]
			p = probabilities[DICT_STATES[state]]
			eu += v*p
		return eu

	def update_DEPRECATED(self, score:int, obs_state:List[int]) -> None:
		'''Entrena la red'''
		# Crear el dataset
		action = self.actions[-1]
		# Aggregate state
		new_state = self._get_agg_state(action, obs_state)
		index_probability = DICT_STATES[new_state]
		probability = [0] * 4
		probability[index_probability] = 1
		self.probabilities.append(probability)
		mask = np.random.choice(list(range(len(self.states))), self.len_exp)
		states = self.states[mask]
		probabilities = self.probabilities[mask]
		ds = ExperienceDataset(
			states=[torch.tensor(states, dtype=torch.float32)],
			probabilities=[torch.tensor(probabilities, dtype=torch.float32)]
		)
		# Crear el dataloader
		ds_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
		# Entrena por num epocas
		for epoch in range(self.num_epochs):
			self.NN.learn(ds_loader)



class AgentMFPAgg(Agent) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule
	with aggregated states.
	It uses the following parameters:
		* belief_strength (int)
		* alphas (dict, initial probability for each transition)
	It also requires an id_number n
	'''

	def __init__(
				self, 
				parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		super().__init__(parameters, n)
		assert(parameters["alphas"] is not None)
		self.alphas = deepcopy(parameters["alphas"])
		assert(parameters["belief_strength"] is not None)
		self.belief_strength = parameters["belief_strength"]
		self.prev_state_ = None
		# states: (a, b) where:
		#		 a == 1 the agent goes; a == 0 the agent doesn't go
		#		 b == 1 no seats available for agent; b == 0 at least one seat available
		self.states = [(a,b) for a in range(2) for b in range(2)]
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(parameters["alphas"])
		self.payoff = np.matrix([[0, 0], [1, -1]])
		self.convergence = [max([value for key, value in self.trans_probs.items()])]

	def update(self, score:int, obs_state:List[int]):
		'''
		Agent updates its model using the Markov Fictitious Play rule.
		Input:
			- score, a number 0 or 1.
			- obs_state, a tuple with the sate of current round,
						 where each argument is 0 or 1.
		Input:
		'''
		# Update records
		self.scores.append(score)
		action = obs_state[self.number]
		self.decisions.append(action)
		# Register to calculate convergence
		trans_probs = deepcopy(self.trans_probs)
		# Aggregate state
		state = self._get_agg_state(action, obs_state)
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			prev_state = self.prev_state_
			# Update transtion counts
			observed_transition = (prev_state, state)
			self.count_transitions[observed_transition] += 1
			# Loop over states and update transition probabilities
			for new_state in self.states:
				transition = (prev_state, new_state)
				numerator = self.count_transitions[transition] + self.belief_strength * self.alphas[transition]
				denominator = self.count_states[prev_state] + self.belief_strength
				new_prob = numerator / denominator
				assert(new_prob <= 1), f'\nTransition:{transition}\nTransition counts:{self.count_transitions[transition]}\nState counts:{self.count_states[prev_state]}'
				self.trans_probs[transition] = new_prob
		# Update state counts
		self.count_states[state] += 1
		# Update previous state
		self.prev_state_ = state
		# Update convergence
		differences = np.abs([trans_probs[k] - self.trans_probs[k] for k in trans_probs.keys()])
		self.convergence.append(max(differences))

	def _get_agg_state(self, action:int, obs_state:List[int]) -> tuple:
		agg_partners = sum([obs_state[i] for i in range(len(obs_state)) if i != self.number])
		agg_state = 1 if agg_partners >= int(self.threshold * len(obs_state)) else 0
		state = (action, agg_state)  
		return state

	def reset(self) -> None:
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(self.alphas)

	def sample_action(self, prev_state:List[int]) -> int:
		'''
		Samples an action with a softmax over expected utilities.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
		Output:
			- a decision 0 or 1
		'''
		logits = [self.exp_util(prev_state, action) for action in [0,1]]
		weights = np.exp(logits) / np.exp(logits).sum()
		return np.random.choice([0,1], p=weights)

	def exp_util(self, prev_state:List[int], action:int) -> float:
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						  where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		eu = 0
		for aggregated_partners in [0,1]:
			state = (action, aggregated_partners)
			v = self.payoff[action, aggregated_partners]
			p = self.trans_probs[(prev_state, state)]
			eu += v*p
		return eu

	def print_agent(self, ronda:Optional[Union[int, None]]=None) -> str:
		'''
		Returns a string with the state of the agent on a given round.
		Input:
			- ronda, integer with the number of the round.
		Output:
			- string with a representation of the agent at given round.
		'''
		if ronda is None:
			try:
				ronda = len(self.decisions) - 1
			except:
				ronda = 0
		try:
			decision = self.decisions[ronda]
		except:
			decision = "nan"
		try:
			score = self.scores[ronda]
		except:
			score = "nan"
		states = [(a,b) for a in [0,1] for b in [0,1]]
		probs = '		' + ' '.join([str(s) for s in states])
		print(probs)
		for prev_state in states:
			probs += '\n' + str(prev_state)
			for state in states:
				dummy = str(round(self.trans_probs[(prev_state, state)],2))
				if len(dummy) < 4:
					dummy += '0'*(4-len(dummy))
				probs += '   ' + dummy
		print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")


