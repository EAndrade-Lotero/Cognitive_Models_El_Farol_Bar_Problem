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
		self.ingest_parameters(fixed_parameters, free_parameters)
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
			preferences = self.determine_action_preferences()
			probabilities = self.softmax(preferences)
			if self.debug:
				print('Action probabilities:')
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
		# numerator = np.exp(self.inverse_temperature * np.array(preferences))
		numerator = np.exp(self.inverse_temperature * np.array(preferences) - np.max(preferences)) # <= subtracted max for numerical stability
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

	def determine_action_preferences(self) -> List[float]:
		# To be defined by subclass
		pass

	def update(self, score:int, obs_state:List[int]) -> None:
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
		try:
			action = int(action)
		except Exception as e:
			print(f'Error: action of type {type(action)}. Type int was expected. (previous actions: {self.decisions})')
			raise Exception(e)
		self.decisions.append(action)
		self.prev_state_ = tuple(obs_state)

	def reset(self) -> None:
		'''
		Restarts the agent's data for a new trial.
		'''
		self.decisions = []
		self.scores = []
		self.prev_state_ = None

	def restart(self) -> None:
		# To be defined by subclass
		pass

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

	def ingest_parameters(
				self, 
				fixed_parameters:Dict[str,any], 
				free_parameters:Dict[str,any]
			) -> None:
		'''
		Ingests parameters from the model.
		Input:
			- fixed_parameters, dictionary with fixed parameters
			- free_parameters, dictionary with free parameters
		'''
		self.fixed_parameters = fixed_parameters
		self.free_parameters = free_parameters
		self.threshold = fixed_parameters["threshold"]
		self.num_agents = int(fixed_parameters["num_agents"])
		if "inverse_temperature" in free_parameters:
			self.inverse_temperature = free_parameters["inverse_temperature"]

	@staticmethod
	def name():
		return 'CogMod'


class Random(CogMod) :
	'''
	Implements a random rule of go/no go with probability given by go_prob.
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
		go_prob = self.go_probability()
		return 1 if uniform(0, 1) < go_prob else 0
	
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


class RandomM1(CogMod) :
	'''
	Implements a random rule of go/no go with probability given by go_prob.
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
		#--------------------------------------------
		# Create counters
		#--------------------------------------------
		self.states = np.array([0])
		self.restart()
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.ingest_parameters(fixed_parameters, free_parameters)

	def ingest_parameters(
				self, 
				fixed_parameters:Dict[str,any], 
				free_parameters:Dict[str,any]
			) -> None:
		super().ingest_parameters(fixed_parameters, free_parameters)
		self.go_prob = np.zeros(self.states.shape)
		for state in self.states:
			self.go_prob[state] = free_parameters[f"go_prob_{state}"]

	def make_decision(self) -> int:
		'''
		Agent decides whether to go to the bar or not.
		Output:
			- A decision 0 or 1
		'''
		go_prob = self.go_probability()
		return 1 if uniform(0, 1) < go_prob else 0
	
	def go_probability(self):
		'''
		Agent returns the probability of going to the bar
		according to its model.
		Output:
			- p, float representing the probability that the
				agent goes to the bar.
		'''
		return self.go_prob[self.prev_state_]

	def update(self, score:int, obs_state:List[int]) -> None:
		self.prev_state_ = 0

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
		return 'Random-M1'


class RandomM2(RandomM1) :
	'''
	Implements a random rule of go/no go with probability given by go_prob.
	This models conditions the probability on the previous action and aggregate state.
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
		#--------------------------------------------
		# Create counters
		#--------------------------------------------
		self.states = list(product([0, 1], np.arange(self.num_agents)))
		self.restart()
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.ingest_parameters(fixed_parameters, free_parameters)

	def update(self, score:int, obs_state:List[int]) -> None:
		action = obs_state[self.number]
		attendance_others = np.sum(obs_state) - action
		self.prev_state_ = (action, attendance_others)


class RandomM3(RandomM1) :
	'''
	Implements a random rule of go/no go with probability given by go_prob.
	This models conditions the probability on the previous actions vector, the full-state.
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
		#--------------------------------------------
		# Create counters
		#--------------------------------------------
		self.states = list(product([0, 1], repeat=self.num_agents))
		self.restart()
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.ingest_parameters(fixed_parameters, free_parameters)

	def update(self, score:int, obs_state:List[int]) -> None:
		self.prev_state_ = tuple(obs_state)


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
		self.ingest_parameters(fixed_parameters, free_parameters)

	def ingest_parameters(
				self, 
				fixed_parameters:Dict[str,any], 
				free_parameters:Dict[str,any]
			) -> None:
		super().ingest_parameters(fixed_parameters, free_parameters)		
		self.go_drive = free_parameters["go_drive"]
		self.wsls_strength = free_parameters["wsls_strength"]

	def determine_action_preferences(self) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Output:
			- List with no go preference followed by go preference
		'''
		# Get previous action
		previous_state = self.prev_state_
		action = previous_state[self.number]
		# Use model to determine preferences
		payoff = self.payoff(action, previous_state)
		go_preference = self.go_drive + self.wsls_strength * payoff
		no_go_preference = 1 - self.go_drive
		if self.debug:
			print(f'payoff: {payoff}')
			print(f'go_preference = {self.go_drive} + {self.wsls_strength} * {payoff}')
		# Return preferences
		preferences = [no_go_preference, go_preference]
		return preferences

	@staticmethod
	def name():
		return 'WSLS'


class PayoffM1(CogMod) :
	'''
	Defines the error-driven learning rule based on payoffs.
	This is the unconditioned model.
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
		# Bookkeeping for go preference
		#----------------------
		self.backup_Q = np.zeros(2)
		self.Q = deepcopy(self.backup_Q)
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.ingest_parameters(fixed_parameters, free_parameters)

	def ingest_parameters(
				self, 
				fixed_parameters:Dict[str,any], 
				free_parameters:Dict[str,any]
			) -> None:
		super().ingest_parameters(fixed_parameters, free_parameters)
		self.learning_rate = free_parameters["learning_rate"]

	def determine_action_preferences(self) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Output:
			- List with no go preference followed by go preference
		'''
		if self.prev_state_ is None:
			return [0, 0]
		else:
			return self.Q

	def update(
				self, 
				score: int, 
				obs_state: Tuple[int]
			) -> None:
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
			self.learn(obs_state)
		# Update records
		self.scores.append(score)
		self.decisions.append(action)
		self.prev_state_ = obs_state

	def learn(
				self,
				obs_state: Tuple[int],
			) -> None:
		'''
		Agent updates their action preferences
		Input:
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		'''
		# Get action
		action = obs_state[self.number]
		# Observe G
		G = self._get_G(obs_state)
		# Determine error prediction
		delta = G - self.Q[action]
		# Update Q table
		if self.debug:
			print('Learning rule:')
			print(f'Q[{action}] <- {self.Q[action]} + {self.learning_rate} * ({G} - {self.Q[action]})')
		self.Q[action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[{action}] = {self.Q[action]}')

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		G = self.payoff(action, obs_state)
		if self.debug:
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	def reset(self) -> None:
		super().reset()

	def restart(self) -> None:
		super().reset()
		self.Q = deepcopy(self.backup_Q)

	@staticmethod
	def name():
		return 'Payoff-M1'


class PayoffM2(PayoffM1) :
	'''
	Defines the error-driven learning rule based on payoffs.
	This model conditions G on the previous action and aggregate state.
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
		# Bookkeeping for go preference
		#----------------------
		self.backup_Q = np.zeros((2, self.num_agents, 2))
		self.Q = deepcopy(self.backup_Q)

	def determine_action_preferences(self) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Output:
			- List with no go preference followed by go preference
		'''
		if self.prev_state_ is None:
			return [0, 0]
		else:
			prev_action, attendance = self._get_index(self.prev_state_)
			return self.Q[prev_action, attendance, :]

	def learn(
				self,
				obs_state: Tuple[int],
			) -> None:
		'''
		Agent updates their action preferences
		Input:
			- action, go = 1 or no_go = 0
			- previous_state, list of decisions on previous round
			- new_state, list of decisions obtained after decisions
		'''
		# Get previous state
		previous_state = self.prev_state_
		# Get action
		action = obs_state[self.number]
		# Observe G 
		G = self._get_G(obs_state)
		# Determine error prediction
		prev_action, attendance = self._get_index(previous_state)
		delta = G - self.Q[prev_action, attendance, action]
		# Update Q table
		if self.debug:
			print('Learning rule:')
			print(f'Q[({prev_action}, {attendance}), {action}] <- {self.Q[prev_action, attendance, action]} + {self.learning_rate} * ({G} - {self.Q[prev_action, attendance, action]})')
		self.Q[prev_action, attendance, action] += self.learning_rate * delta
		if self.debug:
			print(f'Q[({prev_action}, {attendance}), {action}] = {self.Q[prev_action, attendance, action]}')

	def _get_index(self, state: List[int]) -> int:
		'''
		Determines the index of a state in a Q table
		Input:
			- state, list of decisions
		Output:
			- tuple, integers corresponding to the action and attendance
		'''
		if isinstance(state, dict):
			state_ = list(state.values())
		else:
			state_ = state
		action = state_[self.number]
		attendance_others = np.sum(state_) - action
		# Return index
		return action, attendance_others
	
	@staticmethod
	def name():
		return 'Payoff-M2'
	

class PayoffM3(PayoffM1) :
	'''
	Defines the error-driven learning rule based on payoffs.
	This model conditions G on the previous actions vector, the full-state.
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
		# Bookkeeping for go preference
		#----------------------
		self.backup_Q = np.zeros((2 ** self.num_agents, 2))
		self.Q = deepcopy(self.backup_Q)

	def determine_action_preferences(self) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Output:
			- List with no go preference followed by go preference
		'''
		if self.prev_state_ is None:
			return [0, 0]
		else:
			index_previous_state = self._get_index(self.prev_state_)
			return self.Q[index_previous_state, :]

	def learn(
				self,
				obs_state: Tuple[int],
			) -> None:
		'''
		Agent updates their action preferences
		Input:
			- action, go = 1 or no_go = 0
			- previous_state, list of decisions on previous round
			- new_state, list of decisions obtained after decisions
		'''
		# Get previous state
		previous_state = self.prev_state_
		# Get action
		action = obs_state[self.number]
		# Observe G 
		G = self._get_G(obs_state)
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
	
	@staticmethod
	def name():
		return 'Payoff-M3'


class AvailableSpaceM1(PayoffM1) :
	'''
	Defines the error-driven learning rule based on 
	available space in the bar.
	This is the unconditioned model.
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

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get attendance other players
		attendance_others = np.sum(obs_state) - action
		G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
		if self.debug:
			print(f'Attendance other players: {attendance_others}')
			if action == 0:
				print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
			elif action == 1:
				print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'AvailableSpace-M1'


class AvailableSpaceM2(PayoffM2) :
	'''
	Defines the error-driven learning rule based on 
	available space in the bar.
	This model conditions G on the previous action 
	and aggregate state.
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

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get attendance other players
		attendance_others = np.sum(obs_state) - action
		G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
		if self.debug:
			print(f'Attendance other players: {attendance_others}')
			if action == 0:
				print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
			elif action == 1:
				print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'AvailableSpace-M2'


class AvailableSpaceM3(PayoffM3) :
	'''
	Defines the error-driven learning rule based on 
	available space in the bar.
	This model conditions G on the previous actions vector, the full-state.
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

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get attendance other players
		attendance_others = np.sum(obs_state) - action
		G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
		if self.debug:
			print(f'Attendance other players: {attendance_others}')
			if action == 0:
				print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
			elif action == 1:
				print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G


	@staticmethod
	def name():
		return 'AvailableSpace-M3'


class AttendanceM1(PayoffM1) :
	'''
	Defines the error-driven learning rule based on 
	weighted combination of average go and payoff.
	This is the unconditioned model.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_go = np.mean(self.decisions + [action])
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_go + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average go: {average_go}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Attendance-M1'


class AttendanceM2(PayoffM2) :
	'''
	Defines the error-driven learning rule based on 
	weighted combination of average go and payoff.
	This model conditions G on the previous action 
	and aggregate state.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_go = np.mean(self.decisions + [action])
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_go + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average go: {average_go}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Attendance-M2'


class AttendanceM3(PayoffM3) :
	'''
	Defines the error-driven learning rule based on weighted 
	combination of average go and payoff.
	This model conditions G on the previous actions vector, the full-state.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_go = np.mean(self.decisions + [action])
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_go + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average go: {average_go}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Attendance-M3'


class FairnessM1(PayoffM1) :
	'''
	Defines the error-driven learning rule based 
	on weighted combination of fair amount of go
	and payoff.
	This is the unconditioned model.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_fairness = np.mean(self.decisions + [action]) - self.threshold
		average_fairness = average_fairness * (1 - 2 * action)
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_fairness + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average fairness: {average_fairness}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Fairness-M1'


class FairnessM2(PayoffM2) :
	'''
	Defines the error-driven learning rule based 
	on weighted combination of fair amount of go
	and payoff.
	This model conditions G on the previous action 
	and aggregate state.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_fairness = np.mean(self.decisions + [action]) - self.threshold
		average_fairness = average_fairness * (1 - 2 * action)
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_fairness + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average fairness: {average_fairness}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Fairness-M2'
	

class FairnessM3(PayoffM3) :
	'''
	Defines the error-driven learning rule based 
	on weighted combination of fair amount of go
	and payoff.
	This model conditions G on the previous actions vector, the full-state.
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
		self.bias = free_parameters['bias']

	def _get_G(self, obs_state: Tuple[int]) -> float:
		action = obs_state[self.number]
		# Get go frequency
		average_fairness = np.mean(self.decisions + [action]) - self.threshold
		average_fairness = average_fairness * (1 - 2 * action)
		# Get payoff
		payoff = self.payoff(action, obs_state)
		G = self.bias * average_fairness + (1 - self.bias) * payoff
		if self.debug:
			print(f'Average fairness: {average_fairness}')
			print(f'Payoff: {payoff}')
			print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
		return G

	@staticmethod
	def name():
		return 'Fairness-M3'
	

class MFPM1(CogMod) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule 
	for multiple players.
	This model conditions G on the previous actions vector, the full-state.
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
		#--------------------------------------------
		# Create counters
		#--------------------------------------------
		self.states = [0]
		self.restart()
		#----------------------
		# Bookkeeping for model parameters
		#----------------------
		self.ingest_parameters(fixed_parameters, free_parameters)

	def ingest_parameters(
				self, 
				fixed_parameters:Dict[str,any], 
				free_parameters:Dict[str,any]
			) -> None:
		super().ingest_parameters(fixed_parameters, free_parameters)
		self.belief_strength = free_parameters['belief_strength']
		assert(self.belief_strength > 0)

	def determine_action_preferences(self) -> List[float]:
		'''
		Agent determines their preferences to go to the bar or not.
		Input:
			- state, list of decisions of all agents
		Output:
			- List with no go preference followed by go preference
		'''
		if self.prev_state_ is not None:
			eus = [self.exp_util(action) for action in [0,1]]
		else:
			eus = [0, 0]
		if self.debug:
			print('Expected utilities:')
			print(f'no go:{eus[0]} ---- go:{eus[1]}')
		return eus
	
	def exp_util(self, action):
		'''
		Evaluates the expected utility of an action.
		Input:
			- prev_state, a tuple with the state of the previous round, 
						where each argument is 0 or 1.
			- action, which is a possible decision 0 or 1.
		Output:
			- The expected utility (float).
		'''
		if action == 0:
			return 0
		else:
			prev_sate = self.get_prev_state()
			numerator = self.count_bar_with_capacity(prev_sate) + self.belief_strength
			denominator = self.count_states(prev_sate) + 2 * self.belief_strength
			prob_capacity = numerator / denominator
			prob_crowded = 1 - prob_capacity
			eu = prob_capacity - prob_crowded
			if self.debug:
				print(f'{prob_capacity=} --- {prob_crowded=}')
		return eu

	def update(self, score:int, obs_state:Tuple[int]):
		'''
		Agent updates its model using the observed frequencies.
		Input:
			- score, a number 0 or 1.
			- obs_state_, a tuple with the sate of current round,
						where each argument is 0 or 1.
		Input:
		'''
		# Update records
		self.scores.append(score)
		self.decisions.append(obs_state[self.number])
		# Agent recalls previous state?
		if self.prev_state_ is not None:
			prev_state = self.get_prev_state()
			# Increment count of states
			self.count_states.increment(prev_state)
			# Find other player's attendance
			action = obs_state[self.number]
			other_players_attendance = sum(obs_state) - action
			if other_players_attendance < int(self.threshold * self.num_agents):
				# Increment count of bar with capacity given previous state
				self.count_bar_with_capacity.increment(prev_state)
		if self.debug:
			print(f'I see the previous state: {prev_state}')
			print('I recall the following frequencies of states:')
			print(self.count_states)
			print('I recall the following frequencies of bar with capacity:')
			print(self.count_bar_with_capacity)
		# Update previous state
		self.prev_state_ = obs_state

	def _get_error_message(
				self, 
				new_prob: float, 
				transition: Dict[any, any], 
				prev_state: Tuple[int]
			) -> str:
		error_message = f'Error: Improper probability value {new_prob}.\n'
		error_message += f'Transition:{transition}\n'
		error_message += f'Transition counts:{self.count_transitions(transition)}\n'
		error_message += f'Prev. state counts:{self.count_states(tuple(prev_state))}'
		return error_message	

	def reset(self) :
		'''
		Restarts the agent's data for a new trial.
		'''
		super().reset()
		self.prev_state_ = None

	def restart(self) -> None:
		'''
		Restarts the agent memory.
		'''
		self.reset()
		self.count_states = ProxyDict(keys=self.states, initial_val=0)
		self.count_bar_with_capacity = ProxyDict(
			keys=self.states,
			initial_val=0
		)

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
		tp = TransitionsFrequencyMatrix(num_agents=self.num_agents)
		tp.from_proxydict(self.trans_probs)
		print(tp)

	def get_prev_state(self):
		return 0

	@staticmethod
	def name():
		return 'MFP-M1'


class MFPM2(MFPM1) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule 
	for multiple players.
	This is the unconditioned model.
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
		self.states = list(product([0, 1], np.arange(self.num_agents)))
		self.restart()

	def get_prev_state(self):
		assert(self.prev_state_ is not None)
		action = self.prev_state_[self.number]
		others_attendance = sum(self.prev_state_) - action 
		prev_state = (action, others_attendance)
		return prev_state

	@staticmethod
	def name():
		return 'MFP-M2'
	

class MFPM3(MFPM1) :
	'''
	Implements an agent using the Markov Fictitious Play learning rule 
	for multiple players.
	This model conditions G on the previous actions vector, the full-state.
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
		self.states = list(product([0, 1], repeat=self.num_agents))
		self.restart()

	def get_prev_state(self):
		assert(len(self.prev_state_) == self.num_agents)
		if isinstance(self.prev_state_, list):
			prev_state = tuple(self.prev_state_)
		else:
			prev_state = self.prev_state_
		return prev_state

	@staticmethod
	def name():
		return 'MFP-M3'


free_parameters_error_driven_1 = {
	'inverse_temperature':10,
	'learning_rate': 0.001,
}
free_parameters_error_driven_2 = {
	'inverse_temperature':10,
	'learning_rate': 0.001,
	'bias': 0.5
}
free_parameters_MFP = {
	'inverse_temperature':10,
	'belief_strength': 10
}

MODELS = {
	'Random': {
		'class': Random,
		'free_parameters': {
			'go_prob':0,
		}
	# 'Random-M1': {
	# 	'class': RandomM1,
	# 	'free_parameters': {
	# 		'go_prob_1':0,
	# 	}
	# }, 
	# 'Random-M2': {
	# 	'class': RandomM2,
	# 	'free_parameters': {
	# 		'go_prob_2':0,
	# 	}
	# }, 
	# 'Random-M3': {
	# 	'class': RandomM3,
	# 	'free_parameters': {
	# 		'go_prob_3':0,
	# 	}
	}, 
	'WSLS': {
		'class': WSLS, 
		'free_parameters': {
			'inverse_temperature':10,
			'go_drive':0,
			'wsls_strength':0
		}
	}, 
	'Payoff-M1': {
		'class': PayoffM1,
		'free_parameters': free_parameters_error_driven_1
	},
	'Payoff-M2': {
		'class': PayoffM2,
		'free_parameters': free_parameters_error_driven_1
	},
	'Payoff-M3': {
		'class': PayoffM3,
		'free_parameters': free_parameters_error_driven_1
	},
	'AvailableSpace-M1': {
		'class': AvailableSpaceM1,
		'free_parameters': free_parameters_error_driven_1
	},
	'AvailableSpace-M2': {
		'class': AvailableSpaceM2,
		'free_parameters': free_parameters_error_driven_1
	},
	'AvailableSpace-M3': {
		'class': AvailableSpaceM3,
		'free_parameters': free_parameters_error_driven_1
	},
	'Attendance-M1': {
		'class': AttendanceM1,
		'free_parameters': free_parameters_error_driven_2
	},
	'Attendance-M2': {
		'class': AttendanceM2,
		'free_parameters': free_parameters_error_driven_2
	},
	'Attendance-M3': {
		'class': AttendanceM3,
		'free_parameters': free_parameters_error_driven_2
	},
	'Fairness-M1': {
		'class': FairnessM1,
		'free_parameters': free_parameters_error_driven_2
	},
	'Fairness-M2': {
		'class': FairnessM2,
		'free_parameters': free_parameters_error_driven_2
	},
	'Fairness-M3': {
		'class': FairnessM3,
		'free_parameters': free_parameters_error_driven_2
	},
	'MFP-M1': {
		'class': MFPM1,
		'free_parameters': free_parameters_MFP
	},
	'MFP-M2': {
		'class': MFPM2,
		'free_parameters': free_parameters_MFP
	},
	'MFP-M3': {
		'class': MFPM3,
		'free_parameters': free_parameters_MFP
	}
}
