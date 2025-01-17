import pandas as pd
from pathlib import Path
from bayes_opt import BayesianOptimization
from typing import (
	List, Dict, Tuple,
	Union, Optional, Callable
)

from Classes.bar import Bar
from Classes.cognitive_model_agents import *
from Utils.interaction import Episode
from Utils.utils import PPT


class GetMeasure:
	'''
	Class to obtain a measure from a model given some 
	free parameters.

	Input:
		- agent_class, an agent class
		- fixed_parameters, a dictionary with the model's fixed parameters
		- free_parameters, a dictionary with the model's free parameters
	'''
	def __init__(
				self, 
				agent_class: CogMod, 
				free_parameters: Dict[str, any],
				fixed_parameters: Dict[str, any],
				measure: Optional[Union[str, None]]=None,
			) -> None:
		self.agent_class = agent_class
		self.free_parameters = free_parameters
		self.fixed_parameters = fixed_parameters
		self.num_rounds = 100
		self.num_episodes = 50
		self.T = 20
		# Select measure
		if measure is None:
			self.measure = 'efficiency'
		else:
			self.measure = measure
		# For debugging
		self.debug = False

	def create_loss_function(self, method_measure: str) -> Callable:
		assert(hasattr(self, method_measure))
		arguments = ', '.join([f'{key}=None' for key in self.free_parameters.keys()])
		function_definition = f'''
from types import MethodType

def def_funct(self, {arguments}):
	parameters = locals()
	measure_value = self.{method_measure}(parameters)
	return measure_value

self.black_box_function = MethodType(def_funct, self)
'''
		# print(function_definition)
		exec(function_definition)

	def efficiency(self, free_parameters: Dict[str, any]) -> float:
		# Generate simulated data
		if self.debug:
			print('Parameters for simulation:')
			print(free_parameters)
		data = self.generate_simulated_data(free_parameters)
		# Get only last T rounds from data
		num_rounds = max(data["round"].unique())
		data = pd.DataFrame(data[data["round"] >= num_rounds - self.T])
		if self.debug:
			print('')
			print('-'*50)
			# Get average score per simulation
			for sim, grp in data.groupby('id_sim'):
				m = grp['score'].mean()
				print(f'Sim:{sim} => mean score:{m}')
		# Get average score
		return data.score.mean()

	def generate_simulated_data(
				self, 
				free_parameters: Dict[str, any]
			) -> None:
		'''Generate a dataset with simulated data with the 
		given free parameters'''
		bar, agents = self.initialize(free_parameters)
		episode = Episode(
			environment=bar,
			agents=agents,
			model=self.agent_class.name(),
			num_rounds=self.num_rounds
		)
		df = episode.simulate(num_episodes=self.num_episodes)
		return df

	def initialize(
				self,
				free_parameters: Dict[str, any]
			) -> Tuple[Bar, List[CogMod]]:
		bar = Bar(
			num_agents=self.fixed_parameters['num_agents'],
			threshold=self.fixed_parameters['threshold']
		)
		fixed_parameters = self.fixed_parameters.copy()
		fixed_parameters['num_agents']
		if self.agent_class == MFP:
			states = list(product([0,1], repeat=self.fixed_parameters['num_agents']))
			count_states = ProxyDict(
				keys=states,
				initial_val=0
			)
			count_transitions = ProxyDict(
				keys=list(product(states, repeat=2)),
				initial_val=0
			)
			fixed_parameters['states'] = states
			fixed_parameters['count_states'] = count_states
			fixed_parameters['count_transitions'] = count_transitions
		agents = [
			self.agent_class(
				free_parameters=free_parameters, 
				fixed_parameters=self.fixed_parameters, 
				n=n
			) for n in range(self.fixed_parameters['num_agents'])
		]
		agents[0].designated_agent = True
		return bar, agents



class ParameterOptimization :
	'''
	Class for finding the parameters that maximize 
	a given measure of a given agent.

	Input:
		- agent_class, an agent class
		- model_name, the name of the model
		- fixed_parameters (list), a list with the name 
				of the fixed parameters used by the class
		- free_parameters (dict), a dictionary with the value 
				of the free parameters used by the class
		- measure_class, a class that takes the model_class and returns
				a float representing the measure to evaluate the free parameters
		- optimizer, str with the name of the optimizer. Two options available:
			* bayesian
			* skitlearn
	'''

	def __init__(
				self, 
				agent_class: CogMod, 
				free_parameters: Dict[str, any],
				fixed_parameters: Dict[str, any],
				measure_class: any, 
				measure: str, 
				optimizer_name: Union[BayesianOptimization, None],
				hyperparameters: Optional[Union[Dict[str,any],None]]=None
			) -> None:
		# --------------------------
		# Checking measure class
		# --------------------------
		try:
			pr = measure_class(
				agent_class=agent_class,
				free_parameters=free_parameters,
				fixed_parameters=fixed_parameters
			)
		except Exception as e:
			print(f'\t{e}')
			raise Exception('Error: measure_class should initialize with "agent_class", "free_parameters" and "fixed_parameters"!')
		assert(hasattr(measure_class, 'create_loss_function'))
		self.measure_class = measure_class
		# --------------------------
		# Bookkeeping
		# --------------------------
		self.free_parameters = free_parameters
		self.fixed_parameters = fixed_parameters
		self.agent_class = agent_class
		self.measure = measure
		# --------------------------
		# Initialize optimizer
		# --------------------------
		self.optimizer_name = optimizer_name
		if optimizer_name == 'bayesian':
			self.optimizer = self.create_bayesian_optimizer()
		else:
			raise Exception('Oooops')
		if hyperparameters is None:
				self.hyperparameters = {
					'init_points':4,
					'n_iter':16
				}
		else:
			self.hyperparameters = hyperparameters
		self.verbose = False

	def get_optimal_parameters(self) -> Tuple[float]:
		'''
		Returns the parameters that minimize dev(parameters)
		'''
		self.optimizer.maximize(**self.hyperparameters)
		return self.optimizer.max

	def create_bayesian_optimizer(self):
		# Initialize function to get deviance from model        
		pr = self.measure_class(
			agent_class=self.agent_class,
			free_parameters=self.free_parameters,
			fixed_parameters=self.fixed_parameters
		)
		pr.create_loss_function(self.measure)
		# Bounded region of parameter space
		pbounds = dict()
		for parameter in self.free_parameters:
			extend_dict = self.get_saved_bounds(parameter)
			if extend_dict is not None:
				pbounds.update(extend_dict)
		print('pbounds:', pbounds)
		# Initialize optimizer
		optimizer = BayesianOptimization(
			f=pr.black_box_function,
			pbounds=pbounds,
			random_state=1,
			allow_duplicate_points=True,
			verbose=False
		)
		return optimizer

	def get_saved_bounds(self, parameter:str) -> Tuple[float]:
		'''Returns the known bounds for given parameter'''
		#------------------------------------------
		# Random
		#------------------------------------------
		if parameter == 'go_prob':
			return {'go_prob':(0, 1)}
		#------------------------------------------
		#Other cognitive models
		#------------------------------------------
		elif parameter == 'inverse_temperature':
			return {'inverse_temperature':(1, 64)}
		#------------------------------------------
		# WSLS
		#------------------------------------------
		if parameter == 'go_drive':
			return {'go_drive':(0, 1)}
		if parameter == 'wsls_strength':
			return {'wsls_strength':(0, 10)}
		#------------------------------------------
		# PRW
		#------------------------------------------
		if parameter == 'initial_reward_estimate_go':
			return {'initial_reward_estimate_go':(0, 1)}
		if parameter == 'initial_reward_estimate_no_go':
			return {'initial_reward_estimate_no_go':(0, 1)}
		if parameter == 'learning_rate':
			return {'learning_rate':(0, 1)}
		#------------------------------------------
		# ARW
		#------------------------------------------
		if parameter == 'initial_luft_estimate':
			return {'initial_luft_estimate':(0, 1)}
		if parameter == 'learning_rate':
			return {'learning_rate':(0, 1)}
		#------------------------------------------
		# Qlearning
		#------------------------------------------
		if parameter == 'go_drive':
			return {'go_drive':(0, 1)}
		if parameter == 'discount_factor':
			return {'discount_factor':(0, 1)}
		if parameter == 'learning_rate':
			return {'learning_rate':(0, 1)}
		#------------------------------------------
		# MFP
		#------------------------------------------
		if parameter == 'belief_strength':
			return {'belief_strength':(0, 1)}
		if parameter == 'epsilon':
				return {'epsilon':(0, 1)}
		else:
			raise Exception(f'Parameter {parameter} not known!')



