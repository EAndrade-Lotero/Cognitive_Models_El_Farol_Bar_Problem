'''
Helper functions to gather and process data
'''

import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
from pathlib import Path
from copy import deepcopy
from itertools import product
from random import seed, choices
from IPython.display import clear_output
from typing import List, Dict, Tuple, Union, Optional

from Classes.bar import Bar
from Utils.LaTeX_utils import PrintLaTeX
from Classes.agent_utils import ProxyDict
from Utils.plot_utils import PlotsAndMeasures
from Classes.cognitive_model_agents import *
from Classes.agents import *


class Episode :
	'''
	Runs the problem for a number of rounds and keeps tally of everything.
	'''

	def __init__(self, environment:Bar, agents:List[any], model:str, num_rounds:int):
		'''
		Input:
			- environment, object with the environment on which to test the agents.
			- agents, list with the agents.
			- num_rounds, int with the number of rounds.
		'''
		self.environment = environment
		self.agents = agents
		self.model = model
		self.num_rounds = num_rounds
		self.id = uuid.uuid1()
		self.sleep_time = 0.3
	
	def play_round(self, verbose:Optional[bool]=False):
		'''
		Plays one round of the game.
		Input:
			- verbose, True/False to print information for the round.
		'''
		attendances = list()
		# Ask each agent to make a decision
		for i, agent in enumerate(self.agents):
			# Make decision
			decision = agent.make_decision()
			# Add to list of attendances
			attendances.append(decision)
		# Compute attendance and scores
		attendance, scores = self.environment.step(attendances)
		if verbose:
			print(f'\tAttendance = {attendance}\n')
		for i, agent in enumerate(self.agents):
			score = scores[i]
			# Learning rule is applied
			agent.update(score, attendances)
			if verbose:
				decision = attendances[i]
				decision_ = 'go' if decision == 1 else 'no go'
				print(f'\t\tAgent_{i} => {decision_}')
				agent.print_agent()
				print('')

	def run(self, verbose:bool=False):
		'''
		Runs the trial for the specified number of rounds.
		Input:
			- verbose, True/False to print information for the round.
		'''
		# Run the given number of rounds
		for round in range(self.num_rounds):
			if verbose:
				print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
			self.play_round(verbose=verbose)

	def to_pandas(self) -> pd.DataFrame:
		'''
		Creates a pandas dataframe with the information from the current objects.
		Output:
			- pandas dataframe with the following six variables:
			
			Variables:
				* id_sim: a unique identifier for the simulation
				* threshold: the bar's threshold
				* round: the round number
				* attendance: the round's attendance
				* id_player: the player's number
				* decision: the player's decision
				* score: the player's score
				* model: the model's name
				* convergence: the maximum difference between 
							two previous approximations of 
							probability estimates
		'''
		data = {}
		data["id_sim"]= list()
		data["round"]= list()
		data["attendance"]= list()
		data["id_player"]= list()
		data["decision"]= list()
		data["score"]= list()
		if hasattr(self.agents[0], 'convergence'):
			include_convergence = True
			data["convergence"]= list()
		else:
			include_convergence = False
		for r in range(self.num_rounds):
			for i, a in enumerate(self.agents):
				data["id_sim"].append(self.id)
				data["round"].append(r)
				data["attendance"].append(self.environment.history[r])
				data["id_player"].append(a.number)
				data["decision"].append(a.decisions[r])
				data["score"].append(a.scores[r])
				if include_convergence:
					data["convergence"].append(a.convergence[r])
		df = pd.DataFrame.from_dict(data)		
		df["model"] = self.model
		df["threshold"] = self.environment.threshold
		df["num_agents"] = self.environment.num_agents
		return df
	
	def simulate(self, num_episodes:int=1, file:str=None, verbose:bool=False):
		'''
		Runs a certain number of episodes.
		Input:
			- num_episodes, int with the number of episodes.
			- file, string with the name of file to save the data on.
			- verbose, True/False to print information.
		Output:
			- Pandas dataframe with the following variables:
				Variables:
					* id_sim: a unique identifier for the simulation
					* round: the round number
					* attendance: the round's attendance
					* id_player: the player's number
					* decision: the player's decision
					* score: the player's score
					* model: the model's name
		'''		
		data_frames= list()
		# Run the number of episodes
		for t in tqdm(range(num_episodes)):
			self.id = uuid.uuid1()			
			for agent in self.agents:
				agent.reset()
			if verbose:
				print('\n' + '='*10 + f'Episode {t}' + '='*10 + '\n')
			# Reset environment for new episode
			self.environment.reset()
			# Run the episode
			self.run(verbose=verbose)
			data_frames.append(self.to_pandas())
		data = pd.concat(data_frames, ignore_index=True)
		if file is not None:
			data.to_csv(file, index=False)
		return data

	def renderize(self, folder:str=None):
		'''
		Plots the per round history as a grid.
		Input:
			- folder, string with the name of folder to save the data on.
		'''
		for round in range(self.num_rounds):
			self.play_round(verbose=0)				
			clear_output(wait=True)
			self.environment.render(folder=folder)
			sleep(self.sleep_time)



class Experiment :
	'''
	Compares given models on a number of measures.
	'''

	def __init__(
				self, 
				agent_class: CogMod, 
				fixed_parameters: Dict[str, any],
				free_parameters: Dict[str, any],
				simulation_parameters: Dict[str, any], 
				measures: Optional[List[str]]=[], 
				verbose: Optional[bool]=False,
		) -> None:
		'''
		Input:
			- agent_class, class to create the agents.
			- fixed_parameters, a dictionary with the 
				fixed parameters of the class
			- free_parameters, a dictionary with the
				free parameters of the class
			- simulation_parameters, a diccionary with
				the number of rounds and episodes
			- measures, list of measures
			- verbose, True/False to print information.
		'''
		self.agent_class = agent_class
		self.fixed_parameters = fixed_parameters
		self.free_parameters = free_parameters
		bar, agents = self.initialize()
		self.environment = bar
		self.agents = agents
		self.num_rounds = simulation_parameters['num_rounds']
		self.num_episodes = simulation_parameters['num_episodes']
		self.measures = measures
		self.verbose = verbose
		self.data = None
		self.latex_string = ''

	def initialize(
			self,
			num_agents: Optional[Union[int, None]]=None,
			agent_class: Optional[Union[int, None]]=None
		) -> Tuple[Bar, List[CogMod]]:
		if num_agents is None:
			num_agents = self.fixed_parameters['num_agents']
		if agent_class is None:
			agent_class = self.agent_class
			free_parameters = self.free_parameters.copy()
		else:
			free_parameters = self.free_parameters[agent_class.name()]
		bar = Bar(
			num_agents=num_agents,
			threshold=self.fixed_parameters['threshold']
		)
		fixed_parameters = self.fixed_parameters.copy()
		fixed_parameters['num_agents'] = num_agents
		if agent_class in [MFP, AgentMFPMultiSameTransProb, softmax_greedy]:
			states = list(product([0,1], repeat=num_agents))
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
			if agent_class in [AgentMFPMultiSameTransProb, softmax_greedy]:
				alphas = {(x,y):1/len(states) for x in states for y in states}
				trans_probs = ProxyDict(
					keys=list(product(states, repeat=2)),
					initial_val=1/len(states)
				)
				free_parameters['alphas'] = alphas
				fixed_parameters['trans_probs'] = trans_probs
		agents = [
			agent_class(
				free_parameters=free_parameters, 
				fixed_parameters=fixed_parameters, 
				n=n
			) for n in range(num_agents)
		]
		agents[0].designated_agent = True
		return bar, agents

	def run_sweep1(
				self, \
				parameter: str, \
				values: List[float], \
				folder_plots: Optional[str]=None,
				file_data: Optional[str]=None,
				kwargs: Optional[Dict[str,any]]={}
				) -> None:
		'''
		Runs a parameter sweep of one parameter, 
		obtains the data and shows the plots on the given measures.
		Input:
			- parameter, a string with the name of the parameter.
			- values, a list with the parameter's values.
			- file, string with the name of file to save the plot on.
			- kwargs: dict with additional setup values for plots
		'''
		# Creates list of dataframes
		df_list= list()
		# Iterate over parameter values
		for value in tqdm(values, desc=f'Running models for each {parameter}'):
			if parameter == 'num_agents':
				bar, agents = self.initialize(num_agents=value)
				self.environment = bar
				self.agents = agents
			if parameter == 'agent_class':
				bar, agents = self.initialize(agent_class=value)
				self.environment = bar
				self.agents = agents
			# Creates list for containing the modified agents
			if parameter not in ['agent_class']:
				# Iterate over agents
				for agent_ in self.agents:
					# Modify agent's parameter with value
					instruction = f'agent_.{parameter} = {value}'
					exec(instruction)
			# Check if parameter modifies environment
			if parameter == 'threshold':
				self.environment.threshold = value
			# Create name
			name = f'{parameter}={value}'
			# Check exception for cool_down
			if parameter == 'epsilon' and value is None:
				name = f'{parameter}=cool_down'
			if parameter == 'agent_class':
				name = value.name()
			# Create simulation
			episode = Episode(
				environment=self.environment,\
				agents=self.agents,\
				model=name,\
				num_rounds=self.num_rounds
			)
			# Run simulation
			df = episode.simulate(
				num_episodes=self.num_episodes, 
				verbose=0
			)
			# Append dataframe
			df_list.append(df)
		# Concatenate dataframes
		self.data = pd.concat(df_list, ignore_index=True)
		if file_data is not None:
			self.data.to_csv(file_data, index=False)
			print(f'Data saved to {file_data}')
		# Create plot object
		p = PlotsAndMeasures(self.data)
		# Plot on each given measure
		list_of_paths = p.plot_measures(
			measures=self.measures,
			folder=folder_plots,
			kwargs=kwargs,
			suffix=parameter
		)
		free_parameters = self.agents[0].free_parameters
		latex_string = PrintLaTeX.print_sweep(
			parameters=free_parameters,
			sweep_parameter=parameter,
			values=values,
			list_of_paths=list_of_paths
		)
		self.latex_string = latex_string

	def run_sweep2(self, \
					parameter1:str, \
					values1:list,\
					parameter2:str, \
					values2:list, \
					file:str=None
						):
		'''
		Runs a parameter sweep of one parameter, 
		obtains the data and shows the plots on the given measures.
		Input:
			- parameter1, a string with the name of the first parameter.
			- values1, a list with the first parameter's values.
			- parameter2, a string with the name of the second parameter.
			- values2, a list with the second parameter's values.
			- file, string with the name of file to save the plot on.
		'''
		# Creates list of dataframes
		df_list= list()
		# Creates list of agents
		for value1 in tqdm(values1):
			for value2 in tqdm(values2):
				agents_parameter= list()
				for agent in self.agents:
					agent_ = deepcopy(agent)
					instruction = f'agent_.{parameter1} = {value1}'
					exec(instruction)
					instruction = f'agent_.{parameter2} = {value2}'
					exec(instruction)
					agents_parameter.append(agent_)
				# Creates name
				name = f'{parameter1}={value1}, {parameter2}={value2}'
				# Create simulation
				episode = Episode(environment=self.environment,\
							agents=agents_parameter,\
							model=name,\
							num_rounds=self.num_rounds)
				# Run simulation
				df = episode.simulate(num_episodes=self.num_episodes, verbose=False)
				df[parameter1] = value1
				df[parameter2] = value2
				# Append dataframe
				df_list.append(df)
		# Concatenate dataframes
		self.data = pd.concat(df_list, ignore_index=True)
		p = PlotsAndMeasures(self.data)
		for m in self.measures:
			if m == 'score':
				ax = p.plot_scores_sweep2(parameter1=parameter1,\
										parameter2=parameter2,\
										file=file
										)
			plt.show()



class Performer :

	@staticmethod
	def examine_simulation(
				agent_class: CogMod,
				fixed_parameters: Dict[str, any],
				free_parameters: Dict[str, any],
				simulation_parameters: Dict[str, any],
				semilla: int,
			) -> None:
		num_agents = fixed_parameters['num_agents']
		threshold = fixed_parameters['threshold']
		num_rounds = simulation_parameters['num_rounds']
		verbose = simulation_parameters['verbose']
		agents = [
			agent_class(
				free_parameters=free_parameters, 
				fixed_parameters=fixed_parameters, 
				n=n
			) for n in range(num_agents)
		]
		agents[0].designated_agent = True
		for agent in agents:
			agent.debug = True
		#-------------------------------
		# Create bar
		#-------------------------------
		bar = Bar(num_agents=num_agents, threshold=threshold)
		#-------------------------------
		# Create simulation
		#-------------------------------
		episode = Episode(
			environment=bar,\
			agents=agents,\
			model='MFP',\
			num_rounds=num_rounds
		)
		#-------------------------------
		# Run simulation with seed
		#-------------------------------
		seed(semilla)
		np.random.seed(semilla)
		df = episode.simulate(verbose=verbose)
		for agent in episode.agents:
			print(agent)

	@staticmethod
	def simple_run(
				agent_class: CogMod,
				fixed_parameters: Dict[str, any],
				free_parameters: Dict[str, any],
				simulation_parameters: Dict[str, any],
				measures: Optional[Union[List[str], None]]=None,
				image_folder: Optional[Union[None, Path]]=None,
				data_folder: Optional[Union[None, Path]]=None,
				seeds: Optional[Union[None, List[int]]]=None,
				kwargs: Optional[Union[Dict[str, any], None]]=None
			) -> None:
		num_agents = fixed_parameters['num_agents']
		threshold = fixed_parameters['threshold']
		num_rounds = simulation_parameters['num_rounds']
		verbose = simulation_parameters['verbose']
		agents = [
			agent_class(
				free_parameters=free_parameters, 
				fixed_parameters=fixed_parameters, 
				n=n
			) for n in range(num_agents)
		]
		agents[0].designated_agent = True
		#-------------------------------
		# Create bar
		#-------------------------------
		bar = Bar(num_agents=num_agents, threshold=threshold)
		#-------------------------------
		# Create simulation
		#-------------------------------
		episode = Episode(
			environment=bar,\
			agents=agents,\
			model='',\
			num_rounds=num_rounds
		)
		#-------------------------------
		# Run simulation per seed
		#-------------------------------
		list_images = list()
		if seeds is not None:
			semillas = seeds
		else:
			semillas = choices(list(range(100)), k=4)
			print('Seeds chosen for simple simulation:', semillas)
		for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
			seed(semilla)
			np.random.seed(semilla)
			# print('Seed:', semilla)
			#-------------------------------
			# Run simulation
			#-------------------------------
			df = episode.simulate(verbose=verbose)
			if data_folder is not None:
				data_file = Path.joinpath(data_folder, f'{agent_class.name()}-{semilla}.csv')
				df.to_csv(data_file)
				print(f'Data saved to {data_file}')
			if image_folder is not None and measures is not None:
				if 'render' in measures:
					file_att = Path.joinpath(image_folder, f'render_{semilla}.png')
					list_images.append(file_att)
					episode.environment.render(
						file=file_att,
						num_rounds=10
					)
				measures_ = [m for m in measures if m != 'render']
				if len(measures) > 0:
					p = PlotsAndMeasures(df)
					list_p = p.plot_measures(					
						folder=image_folder,
						measures=measures_,
						kwargs=kwargs
					)
					list_images += list_p
		if len(list_images) > 0:
			#-------------------------------
			# Create latex string
			#-------------------------------
			# Presenting fixed parameters
			latex_string = '\n\n' + r'\noindent\textbf{Fixed parameters:}' + '\n\n'
			latex_string += PrintLaTeX.print_parameters(fixed_parameters, are_free=False)
			# Presenting free parameters
			latex_string += PrintLaTeX.print_parameters(free_parameters)
			# Add list of images
			latex_string += PrintLaTeX.print_table_from_figs(list_images)
			# Clean plot memory
			plt.close()
			return latex_string

	@staticmethod
	def sweep(
				agent_class: CogMod,
				fixed_parameters: Dict[str, any],
				free_parameters: Dict[str, any],
				simulation_parameters: Dict[str, any],
				sweep_parameter: str,
				values: List[any],
				image_folder: Path,
				measures: Optional[Union[List[str], None]]=None,
				kwargs: Optional[Union[Dict[str, str], None]]=None,
			) -> None:
		#-------------------------------
		# Create experiment
		#-------------------------------
		if measures is None:
			measures=[
				'attendance', 
				'deviation', 
				'efficiency', 
				'inequality', 
				'conditional_entropy',
				'entropy'
			]			
		experiment = Experiment(
			agent_class=agent_class,
			fixed_parameters=fixed_parameters,
			free_parameters=free_parameters,
			simulation_parameters=simulation_parameters,
			measures=measures
		)
		#-------------------------------
		# Run sweep
		#-------------------------------
		kwargs_ = {
			'x_label':sweep_parameter,
			'only_value':True,
			'title_size':16,
			'x_label_size':14,
			'y_label_size':14,
		}
		if kwargs is not None:
			kwargs_.update(kwargs)
		experiment.run_sweep1(
			parameter=sweep_parameter,
			values=values,
			folder_plots=image_folder,
			kwargs=kwargs_
		)
		# Clean plot memory
		plt.close()
		return experiment.latex_string

	@staticmethod
	def simple_vs(
				list_dicts: List[Dict[str,any]],
				image_folder: Path,
				measures: List[str],
				kwargs: Optional[Union[Dict[str, any], None]]=None
			) -> None:
		df_list = list()
		for dict_ in list_dicts:
			fixed_parameters = dict_['fixed_parameters']
			free_parameters = dict_['free_parameters']
			simulation_parameters = dict_['simulation_parameters']
			num_agents = fixed_parameters['num_agents']
			threshold = fixed_parameters['threshold']
			num_rounds = simulation_parameters['num_rounds']
			verbose = simulation_parameters['verbose']
			agent_class = dict_['agent_class']
			agents = [
				agent_class(
					free_parameters=free_parameters, 
					fixed_parameters=fixed_parameters, 
					n=n
				) for n in range(num_agents)
			]
			agents[0].designated_agent = True
			#-------------------------------
			# Create bar
			#-------------------------------
			bar = Bar(num_agents=num_agents, threshold=threshold)
			#-------------------------------
			# Create simulation
			#-------------------------------
			episode = Episode(
				environment=bar,\
				agents=agents,\
				model='',\
				num_rounds=num_rounds
			)
			#-------------------------------
			# Run simulation per seed
			#-------------------------------
			semilla = dict_['seed']
			seed(semilla)
			np.random.seed(semilla)
			# print('Seed:', semilla)
			#-------------------------------
			# Run simulation
			#-------------------------------
			df = episode.simulate(verbose=verbose)
			df['model'] = f'{agent_class.name()}-{semilla}'
			df_list.append(df)
		df = pd.concat(df_list, ignore_index=True)
		p = PlotsAndMeasures(df)
		list_images = p.plot_measures(					
			folder=image_folder,
			measures=measures,
			kwargs=kwargs
		)
		#-------------------------------
		# Create latex string
		#-------------------------------
		# Presenting fixed parameters
		latex_string = '\n\n' + r'\noindent\textbf{Fixed parameters:}' + '\n\n'
		latex_string += PrintLaTeX.print_parameters(fixed_parameters, are_free=False)
		# Presenting free parameters
		latex_string += PrintLaTeX.print_parameters(free_parameters)
		# Add list of images
		latex_string += PrintLaTeX.print_table_from_figs(list_images)
		# Clean plot memory
		plt.close()
		return latex_string

	@staticmethod
	def simple_plots(
				agent_class: CogMod,
				fixed_parameters: Dict[str, any],
				free_parameters: Dict[str, any],
				simulation_parameters: Dict[str, any],
				measures: Optional[Union[List[str], None]]=None,
				image_folder: Optional[Union[None, Path]]=None,
				data_folder: Optional[Union[None, Path]]=None,
				kwargs: Optional[Union[Dict[str, str], None]]=None
			) -> None:
		num_agents = fixed_parameters['num_agents']
		threshold = fixed_parameters['threshold']
		num_rounds = simulation_parameters['num_rounds']
		num_episodes = simulation_parameters['num_episodes']
		verbose = simulation_parameters['verbose']
		agents = [
			agent_class(
				free_parameters=free_parameters, 
				fixed_parameters=fixed_parameters, 
				n=n
			) for n in range(num_agents)
		]
		agents[0].designated_agent = True
		#-------------------------------
		# Create bar
		#-------------------------------
		bar = Bar(num_agents=num_agents, threshold=threshold)
		#-------------------------------
		# Create simulation
		#-------------------------------
		episode = Episode(
			environment=bar,\
			agents=agents,\
			model='',\
			num_rounds=num_rounds
		)
		#-------------------------------
		# Run simulation
		#-------------------------------
		df = episode.simulate(
			num_episodes=num_episodes,
			verbose=verbose
		)
		if data_folder is not None:
			data_file = Path.joinpath(data_folder, f'{agent_class.name()}.csv')
			df.to_csv(data_file)
			print(f'Data saved to {data_file}')
		if image_folder is not None and measures is not None:
			#-------------------------------
			# Plot
			#-------------------------------
			p = PlotsAndMeasures(df)
			kwargs_ = {'title': agent_class.name()}
			if kwargs is not None:
				kwargs_.update(kwargs)
			p.plot_measures(
				measures=measures, 
				folder=image_folder,
				kwargs=kwargs_
			)
		elif image_folder is None or measures is not None:
			print('Warning: In order to save plots, both arguments "image_folder" and "measures" should be given.')