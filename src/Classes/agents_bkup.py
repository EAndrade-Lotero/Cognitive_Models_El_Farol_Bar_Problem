'''
Classes with agents' rules
'''
import numpy as np
from copy import deepcopy
from itertools import product
from random import randint, uniform, choice
from typing import Dict

class Agent :
	'''
	Defines the basic methods for each agent.
	'''
	def __init__(self, n:int, parameters:list=[]):
		self.parameters = parameters
		self.number = n
		self.decisions = []
		self.scores = []
		self.prev_state_ = None
		self.debug = False
		
	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# To be defined by subclass
		pass

	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		# Update records
		self.scores.append(score)
		action = obs_state[self.number]
		assert(isinstance(action, int)), f'action:{action} ({type(action)})\n{obs_state}, ({type(obs_state)})'
		self.decisions.append(action)
		self.prev_state_ = obs_state

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
		self.prev_state_ = None

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
	def __init__(self, parameters: Dict[str, any], n:int) -> None:
		super().__init__(n, parameters)
		self.go_prob = parameters["go_prob"]

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		return 1 if uniform(0, 1) < self.go_prob else 0
	
	def update(self, score: int, obs_state_: tuple):
		action = obs_state_[self.number]
		assert(isinstance(action, int)), f'action:{action} ({type(action)})\n{obs_state_}, ({type(obs_state_)})'
		self.decisions.append(action)
		self.scores.append(score)
		self.prev_state_ = obs_state_
		
	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent goes to the bar.
		'''
		return self.go_prob


class AgentMFP(Agent) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule.
	It uses the following parameters:
		* rate
		* alphas
	It also requires an id_number n
	'''
	
	def __init__(self, parameters:dict, n:int) :
		super().__init__(parameters)
		assert(parameters["alphas"] is not None)
		self.alphas = deepcopy(parameters["alphas"])
		self.prev_state_ = None
		self.states = [(a,b) for a in range(2) for b in range(2)]
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(parameters["alphas"])
		assert(parameters["belief_strength"] is not None)
		self.belief_strength = parameters["belief_strength"]
		self.payoff = np.matrix([[0, 0], [1, -1]])
		self.number = n
		
	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			return self.greedy_action(self.prev_state_)
		else:
			# no previous data, so make random decision
			return randint(0, 1)
		
	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model using the Markov Fictitious Play rule.
		Input:
			- score, a number 0 or 1.
			- obs_state, a tuple with the sate of current round,
						where each argument is 0 or 1.
		Input:
		'''
		if isinstance(obs_state, list):
			obs_state = tuple(obs_state)
		# Update records
		self.scores.append(score)
		self.decisions.append(obs_state[self.number])
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			prev_state = self.prev_state_
			# Update transtion counts
			observed_transition = (prev_state, obs_state)
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
		self.count_states[obs_state] += 1
		# Update previous state
		self.prev_state_ = obs_state
	
	def reset(self) :
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(self.alphas)
	
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
	
	def exp_util(self, prev_state:tuple, action:int) -> float:
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
		state = [np.nan] * 2
		state[self.number] = action
		for partner in [0,1]:
			state[1 - self.number] = partner
			v = self.payoff[action, partner]
			p = self.trans_probs[(prev_state, tuple(state))]
			eu += v*p
		return eu
	
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
		states = [(a,b) for a in [0,1] for b in [0,1]]
		probs = '        ' + ' '.join([str(s) for s in states])
		print(probs)
		for prev_state in states:
			probs += '\n' + str(prev_state)
			for state in states:
				dummy = str(round(self.trans_probs[(prev_state, state)],2))
				if len(dummy) < 4:
					dummy += '0'*(4-len(dummy))
				probs += '   ' + dummy
		print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")
	
	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent goes to the bar.
		'''
		# If greedy action is go (a=1), then probability of going is 1.
		# If greedy action is no go (a=0), then probability of going is 0.
		return self.greedy_action()


class AgentMFP_Multi(Agent) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule for multiple players.
	It uses the following parameters:
		* num_agents
		* rate
		* alphas
	It also requires an id_number n
	'''
	
	def __init__(self, parameters:dict, n:int) :
		super().__init__(parameters)
		assert(parameters["alphas"] is not None)
		self.alphas = parameters["alphas"]
		assert(parameters["num_agents"] is not None)
		self.num_agents = parameters["num_agents"]
		assert(parameters["threshold"] is not None)
		self.treshold = parameters["threshold"]
		assert(parameters["belief_strength"] is not None)
		self.belief_strength = parameters["belief_strength"]
		self.number = n
		self.prev_state_ = None
		self.states = list(product([0,1], repeat=self.num_agents))
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(parameters["alphas"])

	def payoff(self, action, partners):
		if action == 0:
			return 0
		elif sum(partners)+1 <= self.treshold*self.num_agents:
			return 1
		else:
			return -1

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			return self.greedy_action(self.prev_state_)
		else:
			# no previous data, so make random decision
			return randint(0, 1)

	def update(self, score:int, obs_state:tuple):
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
		self.decisions.append(obs_state[self.number])
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			prev_state = self.prev_state_
			# Update transtion counts
			observed_transition = (prev_state, obs_state)
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
		self.count_states[obs_state] += 1
		# Update previous state
		self.prev_state_ = obs_state
		
	def reset(self) :
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.trans_probs = deepcopy(self.alphas)

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
		# print(f'{prev_state} -- 0:{eus[0]} --- 1:{eus[1]}')
		max_eu = max(eus)
		max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
		return choice(max_actions)
	
	def exp_util(self, prev_state, action):
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
		#print(list_partners)
		# print(f'prev_state:{prev_state} ---- accion:{action}')
		for partners in list_partners:
			state = list(partners)
			state.insert(self.number, action)
			v = self.payoff(action, partners)
			p = self.trans_probs[(prev_state, tuple(state))]
			# print(f'state:{state} ---- utilidad:{v} --- probabilidad:{p}')
			eu += v*p
		return eu
	
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
		# states = [(a,b) for a in [0,1] for b in [0,1]]
		states = self.states
		probs = '        ' + ' '.join([str(s) for s in states])
		print(probs)
		for prev_state in states:
			probs += '\n' + str(prev_state)
			for state in states:
				dummy = str(round(self.trans_probs[(prev_state, state)],2))
				if len(dummy) < 4:
					dummy += '0'*(4-len(dummy))
				probs += '   ' + dummy
		print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")



class AgentMFPMultiSameTransProb(Agent) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule for multiple players.
	Can use a designated agent to hold the transition probability, shared across players.
	'''
	
	def __init__(self, parameters:dict, n:int) :
		super().__init__(parameters, n)
		assert(parameters["num_agents"] is not None)
		self.num_agents = parameters["num_agents"]
		assert(parameters["threshold"] is not None)
		self.treshold = parameters["threshold"]
		assert(parameters["alphas"] is not None)
		self.alphas = parameters["alphas"]
		assert(parameters["belief_strength"] is not None)
		self.belief_strength = parameters["belief_strength"]
		self.trans_probs = parameters['trans_probs']
		self.count_states = parameters['count_states']
		self.count_transitions = parameters['count_transitions']
		self.states = parameters['states']
		self.designated_agent = parameters['designated_agent']
		self.number = n
		self.prev_state_ = None

	def payoff(self, action, partners):
		if action == 0:
			return 0
		elif sum(partners)+1 <= self.treshold*self.num_agents:
			return 1
		else:
			return -1

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			return self.greedy_action(self.prev_state_)
		else:
			# no previous data, so make random decision
			return randint(0, 1)

	def update(self, score:int, obs_state:tuple):
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
		self.decisions.append(obs_state[self.number])
		if self.designated_agent:
			# Agent recalls previous state?
			if self.prev_state_ is not None:
				prev_state = self.prev_state_
				# Update transtion counts
				observed_transition = (prev_state, obs_state)
				self.count_transitions.increment(observed_transition)
				# Loop over states and update transition probabilities
				for new_state in self.states:
					transition = (prev_state, new_state)
					numerator = self.count_transitions(transition) + self.belief_strength * self.alphas[transition]
					denominator = self.count_states(tuple(prev_state)) + self.belief_strength
					new_prob = numerator / denominator
					assert(0 <= new_prob <= 1), f'\nTransition:{transition}\nTransition counts:{self.count_transitions(transition)}\nState counts:{self.count_states(tuple(prev_state))}'
					# print(f'agente {self.number} --- transición {prev_state}=>{new_state} --- pasa de {round(self.trans_probs(transition))} a {round(new_prob,2)}')
					self.trans_probs.update(transition, new_prob)
			# Update state counts
			self.count_states.increment(obs_state)
		# Update previous state
		self.prev_state_ = obs_state

	def reset(self) :
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None
		if self.designated_agent:
			self.count_states.reset()
			self.count_transitions.reset()
			self.trans_probs.reset()

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
		# print(f'{prev_state} -- 0:{eus[0]} --- 1:{eus[1]}')
		max_eu = max(eus)
		max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
		return choice(max_actions)
	
	def exp_util(self, prev_state, action):
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
		#print(list_partners)
		# print(f'prev_state:{prev_state} ---- accion:{action}')
		for partners in list_partners:
			state = list(partners)
			state.insert(self.number, action)
			v = self.payoff(action, partners)
			p = self.trans_probs((prev_state, tuple(state)))
			# p = self.trans_probs[(prev_state, tuple(state))]
			# print(f'state:{state} ---- utilidad:{v} --- probabilidad:{p}')
			eu += v*p
		return eu
	
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
		# states = [(a,b) for a in [0,1] for b in [0,1]]
		states = self.states
		probs = '        ' + ' '.join([str(s) for s in states])
		print(probs)
		for prev_state in states:
			probs += '\n' + str(prev_state)
			for state in states:
				dummy = str(round(self.trans_probs((prev_state, state)),2))
				if len(dummy) < 4:
					dummy += '0'*(4-len(dummy))
				probs += '   ' + dummy
		print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")



class epsilon_greedy(AgentMFPMultiSameTransProb):
	
	def __init__(self, parameters: dict, n: int):
		super().__init__(parameters, n)
		self.epsilon = parameters["epsilon"]
		# If self.epsilon is None, then cooling down protocol applies.
	
	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# Agent recalls previous state?
		if self.prev_state_ is None:
			return randint(0, 1)
		else:
			if self.epsilon is None:
				# Get the round number
				round = len(self.decisions)
				step = 100
				num_steps = 15
				if round > step * num_steps:
					epsilon = 0
				else:
					k = np.digitize(
						x=round, 
						bins=[i * step for i in range(num_steps)]
					)
					# # Halves epsilon each time
					# epsilon = 1 / (2**k)
					# From 0.5 to 0 in num_steps
					initial_epsilon = 0.8
					epsilon_step = initial_epsilon / num_steps
					epsilon = (num_steps - k) * epsilon_step
				# # Check value of epsilong accoring to cooling protocol.
				# if round < step:
				# 	epsilon = 0.5
				# elif round < 2 * step:
				# 	epsilon = 0.3
				# elif round < 150:
				# 	epsilon = 0.1
				# elif round < 200:
				# 	epsilon = 0.05
				# elif round < 250:
				# 	epsilon = 0.01
				# else:
				# 	epsilon = 0
			else:
				# No cooling protocol. Use stored epsilon.
				epsilon = self.epsilon
			# Explore with probability epsilon.
			if uniform(0, 1) < epsilon:
				return randint(0, 1)
			else:
				# No exploration. Return greedy action.
				return self.greedy_action(self.prev_state_)
	
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
