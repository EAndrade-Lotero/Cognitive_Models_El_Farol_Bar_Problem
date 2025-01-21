'''
Classes with cognitive model agents' rules
'''
import numpy as np
from math import comb
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from random import randint, uniform
from typing import Optional, Union, Dict, List, Tuple

from Classes.agent_utils import ProxyDict, TransitionsFrequencyMatrix


class CogMod () :
	'''
	Basic class for cognitive agents
	'''

	def __init__(
				self, 
				free_parameters: Optional[Dict[str,any]]={}, 
				fixed_parameters: Optional[Dict[str,any]]={}, 
				n: Optional[int]=1,
				fix_overflow: Optional[bool]=True
			) -> None:
		#----------------------
		# Initialize lists
		#----------------------
		self.decisions = []
		self.scores = []
		self.prev_state_ = None
		self.debug = False
		self.number = n
		#----------------------
		# Parameter bookkeeping
		#----------------------
		self.fixed_parameters = fixed_parameters
		self.free_parameters = free_parameters
		self.threshold = fixed_parameters["threshold"]
		self.num_agents = int(fixed_parameters["num_agents"])
		if "inverse_temperature" in free_parameters:
			self.inverse_temperature = free_parameters["inverse_temperature"]
		#----------------------
		# Dealing with softmax overflow
		#----------------------
		self.fix_overflow = fix_overflow

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			go_prob = self.go_probability()
			probabilities = [1 - go_prob, go_prob]
			decision = np.random.choice(
				a=[0,1],
				size=1,
				p=probabilities
			)[0]
			if isinstance(decision, np.int64):
				decision = int(decision)
			return decision
		else:
			# no previous data, so make random decision
			return randint(0, 1)
		
	def go_probability(self) -> float:
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent decides to go to the bar.
		'''
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			# determine action preferences given previous state
			preferences = self.determine_action_preferences(self.prev_state_)
			probabilities = self.softmax(preferences)
			if self.debug:
				print('Probabilities:')
				print(f'no go:{probabilities[0]} ---- go:{probabilities[1]}')
			return probabilities[1]
		else:
			# no previous data
			return 0.5
		
	def payoff(
				self, 
				action: int, 
				state: List[int]
			) -> int:
		'''
		Determines the payoff of an action given the bar's attendance.
		Input:
			- action, go = 1 or no_go = 0
			- state, list of decisions of all agents
		Output:
			- List with payoff for the action
		'''
		attendance = sum([x == 1 or x == '1' for x in state])
		if action == 0 or action == '0':
			return 0
		elif attendance <= self.threshold * self.num_agents:
			return 1
		else:
			return -1
		
	def softmax(
				self,
				preferences: List[float]
			) -> List[float]:
		'''
		Determines the softmax of the vector of preferences.
		Input:
			- preferences, list of preferences for actions
		Output:
			- List with softmax preferences
		'''
		# Broadcast inverse temperature and exponential
		# print('\tPreferences:', preferences)
		numerator = np.exp(self.inverse_temperature * np.array(preferences))
		num_inf = [np.isinf(x) for x in numerator]
		if sum(num_inf) == 1:
			softmax_values = [1 if np.isinf(x) else 0 for x in numerator]
			return softmax_values
		elif sum(num_inf) > 1:
			if self.fix_overflow:
				if self.debug:
					print(f'Overflow warning: {num_inf}')
				numerator = np.array([1 if np.isinf(x) else 0 for x in numerator])
			else:
				raise Exception(f'Error: softmax gives rise to numerical overflow (numerator: {numerator})!')
		# print('\tNumerator:', numerator)
		# find sum of values
		denominator = sum(numerator)
		assert(not np.isinf(denominator))
		if np.isclose(denominator, 0):
			if self.fix_overflow:
				if self.debug:
					print('Underflow warning:')
				softmax_values = [1 / len(numerator) for _ in numerator]
				return softmax_values
			else:
				raise Exception(f'Error: softmax gives rise to numerical overflow (denominator: {denominator})!')
		# print('\tDenominator:', denominator)
		# Return softmax using broadcast
		softmax_values = numerator / denominator
		assert(np.all([not np.isnan(n) for n in softmax_values])), f'numerator:{numerator} --- denominator: {denominator} --- preferences:{preferences}'
		return softmax_values

	def determine_action_preferences(self, prev_state:any) -> List[float]:
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
		assert(isinstance(action, int) or isinstance(action, np.int16)), f'Error: action of type {type(action)}. Type int was expected. (previous actions: {self.decisions})'
		self.decisions.append(action)
		self.prev_state_ = obs_state

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

	@staticmethod
	def name():
		return 'CogMod'


class Random(CogMod) :
	'''
	Implements a random rule of go/no go with equal probability.
	'''
	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.go_prob = free_parameters["go_prob"]

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		return 1 if uniform(0, 1) < self.go_prob else 0
	
	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent goes to the bar.
		'''
		return self.go_prob

	def __str__(self) -> str:
			'''
			Returns a string with the state of the agent on a given round.
			Input:
				- ronda, integer with the number of the round.
			Output:
				- string with a representation of the agent at given round.
			'''
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
			return f"No.agent:{self.number}, go_prob:{self.go_prob}\nDecision:{decision}, Score:{score}"

	@staticmethod
	def name():
		return 'Random'


class WSLS(CogMod) :
	'''
	Defines the model of go drive plus Win-Stay, Lose-Shift.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.go_drive = free_parameters["go_drive"]
		self.wsls_strength = free_parameters["wsls_strength"]

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		# Get previous action
		action = self.decisions[-1]
		assert(isinstance(action, int) or isinstance(action, np.int16)), f'Error: action of type {type(action)}. Type int was expected. (previous actions: {self.decisions})'
		# Use model to determine preferences
		payoff = self.payoff(action, previous_state)
		go_preference = self.go_drive + self.wsls_strength * payoff
		no_go_preference = 1 - self.go_drive
		# Return preferences
		preferences = [no_go_preference, go_preference]
		return preferences

	@staticmethod
	def name():
		return 'WSLS'


class PayoffRescorlaWagner(CogMod) :
	'''
	Defines the model of reinforcement learning that estimates
	actions payoffs using the Rescorla Wagner rule.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.reward_go = free_parameters["initial_reward_estimate_go"]
		self.reward_no_go = free_parameters["initial_reward_estimate_no_go"]
		self.learning_rate = free_parameters["learning_rate"]
		#----------------------
		# Bookkeeping for go preference
		#----------------------
		self.action_preferences = [
			self.reward_no_go, 
			self.reward_go
		]

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		# Return preferences
		return self.action_preferences
	
	def update(self, score:int, obs_state:tuple) -> None:
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		# Agent learns
		action = obs_state[self.number]
		delta = score - self.action_preferences[action]
		if self.debug:
			print('Learning rule:')
			print(f'Q[{action}] <- {self.action_preferences[action]} + {self.learning_rate} * ({score} - {self.action_preferences[action]})')
		self.action_preferences[action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[{action}] <- {self.action_preferences[action]}\n')
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state
	
	def __str__(self) -> str:
		table = PrettyTable()
		table.field_names = ['Agent', self.number]
		table.add_row(['reward_go', self.action_preferences[1]])
		table.add_row(['reward_no_go', self.action_preferences[0]])
		table.add_row(['learning_rate', self.learning_rate])
		table.add_row(['inverse_temperature', self.inverse_temperature])
		return str(table)

	@staticmethod
	def name():
		return 'PRW'


class AttendanceRescorlaWagner(CogMod) :
	'''
	Defines the model of reinforcement learning that estimates
	bar's free space using the Rescorla Wagner rule.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.luft_estimate = free_parameters["initial_luft_estimate"]
		self.learning_rate = free_parameters["learning_rate"]
		#----------------------
		# Bookkeeping for go preference
		#----------------------
		self.action_preferences = [
			0.5, 
			self.luft_estimate
		]

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		self.action_preferences = [0, self.luft_estimate]
		return self.action_preferences

	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		# Agent learns
		action = obs_state[self.number]
		attendance_without_myself = sum(obs_state) - action
		available_space = self.threshold * self.num_agents - attendance_without_myself
		delta = available_space - self.luft_estimate
		if self.debug:
			print('Learning rule:')
			print(f'Q_k+1 <- {self.luft_estimate} + {self.learning_rate} * ({available_space} - {self.luft_estimate})')
		self.luft_estimate += self.learning_rate * delta
		if self.debug:
			print(f'Q_k+1 <- {self.luft_estimate}\n')
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state

	@staticmethod
	def name():
		return 'ARW'


class Q_learning(CogMod) :
	'''
	Defines the model of reinforcement learning that estimates
	long term actions payoffs using the q_learning rule.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.learning_rate = free_parameters["learning_rate"]
		self.discount_factor = free_parameters["discount_factor"]
		go_drive = free_parameters["go_drive"]
		#----------------------
		# Bookkeeping for go preference
		#----------------------
		# Q = parameters["Q"]
		# assert(isinstance(Q, np.ndarray))
		# expected_num_rows = 2 ** self.num_agents
		# expected_num_cols = 2
		# assert(Q.shape == (expected_num_rows, expected_num_cols)), f'Error: Q.shape is {Q.shape} but should be ({expected_num_rows}, {expected_num_cols})'
		self.backup_Q = np.zeros((2 ** self.num_agents, 2))
		self.backup_Q[:,1] = go_drive
		self.backup_Q[:,0] = 0
		self.Q = deepcopy(self.backup_Q)

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		if previous_state is None:
			return [0, 0]
		else:
			index_previous_state = self._get_index(previous_state)
			return self.Q[index_previous_state, :]

	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		action = obs_state[self.number]
		if self.prev_state_ is not None:
			# Agent learns
			self.learn(
				action=action,
				previous_state=self.prev_state_,
				new_state=obs_state
			)
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state

	def learn(
				self,
				action: int,
				previous_state: List[int],
				new_state: List[int]
			) -> None:
		'''
		Agent updates their action preferences
		Input:
			- action, go = 1 or no_go = 0
			- previous_state, list of decisions on previous round
			- new_state, list of decisions obtained after decisions
		'''
		# Bootstrap max expected long term reward
		max_bootstrap = self.maxQ(new_state)
		# Get round payoff
		payoff = self.payoff(action, new_state)
		# Estimage long term reward for state-action pair
		G = payoff + self.discount_factor * max_bootstrap
		# Determine error prediction
		index_previous_state = self._get_index(previous_state)
		delta = G - self.Q[index_previous_state, action]
		# Update Q table
		if self.debug:
			print('Learning rule:')
			print(f'Q[{previous_state},{action}] <- {self.Q[index_previous_state, action]} + {self.learning_rate} * ({G} - {self.Q[index_previous_state, action]})')
		self.Q[index_previous_state, action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[{previous_state},{action}] = {self.Q[index_previous_state, action]}')

	def maxQ(self, state: List[int]) -> float:
		'''
		Determines the max over actions of the estimated long term reward given a state
		Input:
			- state, list of decisions
		Output:
			- maximum of the estimated long term rewards
		'''
		state_index = self._get_index(state)
		estimated_rewards = self.Q[state_index, :]
		return max(estimated_rewards)

	def _get_index(self, state: List[int]) -> int:
		'''
		Determines the index of a state in a Q table
		Input:
			- state, list of decisions
		Output:
			- index, integer with the index of the Q table corresponding to the state
		'''
		if isinstance(state, dict):
			state_ = list(state.values())
		else:
			state_ = state
		# Convert the state into a binary sequence
		binary_sequence = "".join(str(x) for x in state_)
		# Interpret the sequence as a decimal integer
		index =  int(binary_sequence, 2)
		# Return index
		return index
	
	def reset(self) -> None:
		super().reset()
		self.Q = deepcopy(self.backup_Q)

	@staticmethod
	def name():
		return 'Qlearning'


class QAttendance(CogMod) :
	'''
	Defines the model of reinforcement learning that estimates
	long term actions payoffs using the q_learning rule.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.learning_rate = free_parameters["learning_rate"]
		self.discount_factor = free_parameters["discount_factor"]
		self.go_drive = free_parameters["go_drive"]
		#----------------------
		# Bookkeeping for go preference
		#----------------------
		# Q = parameters["Q"]
		# assert(isinstance(Q, np.ndarray))
		# expected_num_rows = 2 ** self.num_agents
		# expected_num_cols = 2
		# assert(Q.shape == (expected_num_rows, expected_num_cols)), f'Error: Q.shape is {Q.shape} but should be ({expected_num_rows}, {expected_num_cols})'
		self.backup_Q = np.zeros((2 ** self.num_agents, 2))
		self.backup_Q[:,1] = self.go_drive
		self.backup_Q[:,0] = 0
		self.Q = deepcopy(self.backup_Q)

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		if previous_state is None:
			return [0, 0]
		else:
			index_previous_state = self._get_index(previous_state)
			return self.Q[index_previous_state, :]

	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		action = obs_state[self.number]
		if self.prev_state_ is not None:
			# Agent learns
			self.learn(
				action=action,
				previous_state=self.prev_state_,
				new_state=obs_state
			)
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state

	def learn(
				self,
				action: int,
				previous_state: List[int],
				new_state: List[int]
			) -> None:
		'''
		Agent updates their action preferences
		Reward signal contains average go frequency
		Input:
			- action, go = 1 or no_go = 0
			- previous_state, list of decisions on previous round
			- new_state, list of decisions obtained after decisions
		'''
		# Get average go frequency
		average_go = sum([self.discount_factor**(i+1) * x  for i, x in enumerate(self.decisions[::-1])])
		# Get round payoff
		payoff = self.payoff(action, previous_state)
		# Bootstrap max expected long term reward
		max_bootstrap = self.maxQ(new_state)
		# Estimage long term reward for state-action pair
		long_term_reward = payoff + self.discount_factor * max_bootstrap
		# Update
		G = average_go + long_term_reward
		# Determine error prediction
		index_previous_state = self._get_index(previous_state)
		delta = G - self.Q[index_previous_state, action]
		# Update Q table
		if self.debug:
			print(f'Discounted average go frequency: {average_go}')
			print(f'Reward: {payoff}')
			print(f'Reward with average go frequency: {G}')
			print('Learning rule:')
			print(f'Q[{previous_state},{action}] <- {self.Q[index_previous_state, action]} + {self.learning_rate} * ({G} - {self.Q[index_previous_state, action]})')
		self.Q[index_previous_state, action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[{previous_state},{action}] = {self.Q[index_previous_state, action]}')

	def maxQ(self, state: List[int]) -> float:
		'''
		Determines the max over actions of the estimated long term reward given a state
		Input:
			- state, list of decisions
		Output:
			- maximum of the estimated long term rewards
		'''
		state_index = self._get_index(state)
		estimated_rewards = self.Q[state_index, :]
		return max(estimated_rewards)

	def _get_index(self, state: List[int]) -> int:
		'''
		Determines the index of a state in a Q table
		Input:
			- state, list of decisions
		Output:
			- index, integer with the index of the Q table corresponding to the state
		'''
		if isinstance(state, dict):
			state_ = list(state.values())
		else:
			state_ = state
		# Convert the state into a binary sequence
		binary_sequence = "".join(str(x) for x in state_)
		# Interpret the sequence as a decimal integer
		index =  int(binary_sequence, 2)
		# Return index
		return index
	
	def reset(self) -> None:
		super().reset()
		self.Q = deepcopy(self.backup_Q)

	@staticmethod
	def name():
		return 'Qlearning'


class QFairness(CogMod) :
	'''
	Defines the model of reinforcement learning that estimates
	long term actions payoffs using the q_learning rule.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.learning_rate = free_parameters["learning_rate"]
		self.discount_factor = free_parameters["discount_factor"]
		self.go_drive = free_parameters["go_drive"]
		#----------------------
		# Bookkeeping for go preference
		#----------------------
		# Q = parameters["Q"]
		# assert(isinstance(Q, np.ndarray))
		# expected_num_rows = 2 ** self.num_agents
		# expected_num_cols = 2
		# assert(Q.shape == (expected_num_rows, expected_num_cols)), f'Error: Q.shape is {Q.shape} but should be ({expected_num_rows}, {expected_num_cols})'
		self.backup_Q = np.zeros((2 ** self.num_agents, 2))
		self.backup_Q[:,1] = self.go_drive
		self.backup_Q[:,0] = 0
		self.Q = deepcopy(self.backup_Q)

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		if previous_state is None:
			return [0, 0]
		else:
			index_previous_state = self._get_index(previous_state)
			return self.Q[index_previous_state, :]

	def update(self, score:int, obs_state:tuple):
		'''
		Agent updates its model.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		action = obs_state[self.number]
		if self.prev_state_ is not None:
			# Agent learns
			self.learn(
				action=action,
				previous_state=self.prev_state_,
				new_state=obs_state
			)
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state

	def learn(
				self,
				action: int,
				previous_state: List[int],
				new_state: List[int]
			) -> None:
		'''
		Agent updates their action preferences
		Reward signal contains fair go average
		Input:
			- action, go = 1 or no_go = 0
			- previous_state, list of decisions on previous round
			- new_state, list of decisions obtained after decisions
		'''
		# Get average go frequency
		average_go = sum([self.discount_factor**(i) * x  for i, x in enumerate(self.decisions[::-1])])
		# Get fair go average
		fair_go = self.threshold - average_go
		# Get round payoff
		payoff = self.payoff(action, previous_state)
		# Bootstrap max expected long term reward
		max_bootstrap = self.maxQ(new_state)
		# Estimage long term reward for state-action pair
		long_term_reward = payoff + self.discount_factor * max_bootstrap
		# Update
		G = fair_go + long_term_reward
		# Determine error prediction
		index_previous_state = self._get_index(previous_state)
		delta = G - self.Q[index_previous_state, action]
		# Update Q table
		if self.debug:
			print(f'Discounted average go frequency: {average_go}')
			print(f'Go fairness: {fair_go}')
			print(f'Reward: {payoff}')
			print(f'Reward with average go frequency: {G}')
			print('Learning rule:')
			print(f'Q[{previous_state},{action}] <- {self.Q[index_previous_state, action]} + {self.learning_rate} * ({G} - {self.Q[index_previous_state, action]})')
		self.Q[index_previous_state, action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[{previous_state},{action}] = {self.Q[index_previous_state, action]}')

	def maxQ(self, state: List[int]) -> float:
		'''
		Determines the max over actions of the estimated long term reward given a state
		Input:
			- state, list of decisions
		Output:
			- maximum of the estimated long term rewards
		'''
		state_index = self._get_index(state)
		estimated_rewards = self.Q[state_index, :]
		return max(estimated_rewards)

	def _get_index(self, state: List[int]) -> int:
		'''
		Determines the index of a state in a Q table
		Input:
			- state, list of decisions
		Output:
			- index, integer with the index of the Q table corresponding to the state
		'''
		if isinstance(state, dict):
			state_ = list(state.values())
		else:
			state_ = state
		# Convert the state into a binary sequence
		binary_sequence = "".join(str(x) for x in state_)
		# Interpret the sequence as a decimal integer
		index =  int(binary_sequence, 2)
		# Return index
		return index
	
	def reset(self) -> None:
		super().reset()
		self.Q = deepcopy(self.backup_Q)

	@staticmethod
	def name():
		return 'QFairness'


class MFP(CogMod) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule for multiple players.
	'''

	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping free parameters
		#----------------------
		self.belief_strength = free_parameters["belief_strength"]
		self.go_drive = free_parameters["go_drive"]
		#----------------------
		# Bookkeeping fixed parameters
		#----------------------
		self.states = fixed_parameters['states']
		self.count_states = fixed_parameters['count_states']
		self.count_transitions = fixed_parameters['count_transitions']
		self.designated_agent = fixed_parameters['designated_agent']
		self.list_partners = list(product([0,1], repeat = self.num_agents-1))
		self.number = n
		self.prev_state_ = None
		#----------------------
		# Create transition probabilities
		#----------------------		
		self.alphas = self.create_alphas()
		self.trans_probs = ProxyDict(
			keys=list(product(self.states, repeat=2)),
			initial_val=1
		)
		self.trans_probs.from_dict(self.alphas)

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		eus = [self.exp_util(previous_state, action) for action in [0,1]]
		if self.debug:
			print('Expected utilities:')
			print(f'no go:{eus[0]} ---- go:{eus[1]}')
		return eus
	
	def update(self, score:int, obs_state:Tuple[int]):
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
					assert(0 <= new_prob <= 1), f'Error: Improper probability value {new_prob}.\nTransition:{transition}\nTransition counts:{self.count_transitions(transition)}\nPrev. state counts:{self.count_states(tuple(prev_state))}'
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
			self.trans_probs.from_dict(self.alphas)
	
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
		# print(f'prev_state:{prev_state} ---- accion:{action}')
		for partners in self.list_partners:
			state = list(partners)
			state.insert(self.number, action)
			v = self.payoff(action, state)
			p = self.trans_probs((tuple(prev_state), tuple(state)))
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
		print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}, Go_d:{self.go_drive}")
		tp = TransitionsFrequencyMatrix(num_agents=self.num_agents)
		tp.from_proxydict(self.trans_probs)
		print(tp)

	def create_alphas(self) -> None:

		def get_drive(y: List[int]) -> float:
			if y[self.number] == 1 and sum(y) <= self.threshold * self.num_agents:
				return self.go_drive
			else:
				return 0
			
		def get_combinatorial_tuples(B: int) -> int:
			aux = 0
			for i in range(B + 1):
				aux += comb(self.num_agents -1, i)
			return aux

		combinatorial_tuples = get_combinatorial_tuples(int(self.threshold * self.num_agents - 1))
		# print(f'combinatorial_tuples: ({self.num_agents - 1},{int(self.threshold * self.num_agents - 1)}) = {combinatorial_tuples}')
		scalling_factor = 2 ** self.num_agents + combinatorial_tuples * self.go_drive
		# print(f'scalling_factor: 2^{self.num_agents} + {combinatorial_tuples}*{self.go_drive} = {scalling_factor}')
		# y = list(self.states[0])
		# y[self.number] = 1
		# print(f'Numerator with drive:', 1 + get_drive(y))
		# wdtg = (1 + get_drive(y)) / scalling_factor
		# print('With drive to go:', wdtg)
		# print('Without drive to go:', 1 / scalling_factor)
		alphas = {(x,y):(1 + get_drive(y))/scalling_factor for x in self.states for y in self.states}
		# x = self.states[0]
		# values_alphas = [alphas[(x,y)] for y in self.states]
		# print('Number of columns:', len(values_alphas))
		# values_alphas_wdtg = [v for v in values_alphas if v == wdtg]
		# print('Number of values with drive to go:', len(values_alphas_wdtg))
		# print('alphas:')
		# print(alphas)
		for x in self.states:
			check_list = [alphas[(x, y)] for y in self.states]
			# print('check_list:', check_list)
			# print('sum:', sum(check_list))
			assert(np.isclose(sum(check_list), 1))
		return alphas
	
	@staticmethod
	def name():
		return 'MFP'


class MFPAgg(CogMod):
		
	def __init__(
				self, 
				free_parameters:Optional[Dict[str,any]]={}, 
				fixed_parameters:Optional[Dict[str,any]]={}, 
				n:Optional[int]=1
			) -> None:
		#----------------------
		# Initialize superclass
		#----------------------
		super().__init__(
			free_parameters=free_parameters, 
			fixed_parameters=fixed_parameters, 
			n=n
		)
		#----------------------
		# Bookkeeping free parameters
		#----------------------
		self.belief_strength = free_parameters["belief_strength"]
		self.go_drive = free_parameters["go_drive"]
		#----------------------
		# Bookkeeping important variables
		#----------------------
		# states: (a, b) where:
		#		 a == 1 the agent goes; a == 0 the agent doesn't go
		#		 b == 1 no seats available for agent; b == 0 at least one seat available
		self.states = [(a,b) for a in range(2) for b in range(2)]
		self.count_states = {state:0 for state in self.states}
		self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
		self.payoff_matrix = np.matrix([[0, 0], [1, -1]])
		self.prev_state_ = None
		#----------------------
		# Create transition probabilities
		#----------------------		
		self.alphas = self.create_alphas()
		self.trans_probs = deepcopy(self.alphas)

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
			v = self.payoff_matrix[action, aggregated_partners]
			p = self.trans_probs[(tuple(prev_state), state)]
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

	def determine_action_preferences(
				self,
				previous_state: List[int]
			) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		eus = [self.exp_util(previous_state, action) for action in [0,1]]
		if self.debug:
			print('Expected utilities:')
			print(f'no go:{eus[0]} ---- go:{eus[1]}')
		return eus
	
	def create_alphas(self) -> None:

		def get_drive(y: List[int]) -> float:
			if y[1] == 1:
				return self.go_drive
			else:
				return 0

		scalling_factor = 4 + 2 * self.go_drive
		# print('scalling_factor:', scalling_factor)
		alphas = {(x,y):(1 + get_drive(y))/scalling_factor for x in self.states for y in self.states}
		# print('alphas:')
		# print(alphas)
		for x in self.states:
			check_list = [alphas[(x, y)] for y in self.states]
			# print('check_list:', check_list)
			assert(np.isclose(sum(check_list), 1))
		return alphas

	@staticmethod
	def name():
		return 'MFPAgg'


MODELS = {
	'Random': {
		'class': Random,
		'free_parameters': {
			'go_prob':0,
		}
	}, 
	'WSLS': {
		'class': WSLS, 
		'free_parameters': {
			'inverse_temperature':10,
			'go_drive':0,
			'wsls_strength':0
		}
	}, 
	'QL': {
		'class': Q_learning,
		'free_parameters': {
			'inverse_temperature':10,
			"go_drive": 0,
			"learning_rate": 0.001,
			"discount_factor": 0.8
		}
	}, 
	'MFP': {
		'class': MFP,
		'free_parameters': {
			"inverse_temperature":1,
			'belief_strength':1,
			"go_drive":0.5,
		}
	}, 
	'MFPAgg': {
		'class': MFPAgg,
		'free_parameters': {
			"inverse_temperature":1,
			'belief_strength':1,
			"go_drive":0.5,
		}
	},
	'PRW': {
		'class': PayoffRescorlaWagner,
		'free_parameters': {
			'inverse_temperature':10,
			'initial_reward_estimate_go':0,
			'initial_reward_estimate_no_go':0,
			'learning_rate':0.1
		}
	}, 
	'ARW': {
		'class': AttendanceRescorlaWagner,
		'free_parameters': {
			'inverse_temperature':10,
			'initial_luft_estimate':0,
			'learning_rate':0.1
		}
	}
}
