import numpy as np
from pathlib import Path

from Utils.interaction import Performer
from Classes.cognitive_model_agents import Q_learning, Random

folder_name = 'Fig1'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)

fixed_parameters = {
	'num_agents': 4,
	'threshold': 0.5,
}
simulation_parameters = {
	'num_rounds': 5000,
	'num_episodes': 1,
	'verbose': 0,
}
QL_free_parameters = {
	"inverse_temperature": 32,
	"go_drive": 0,
	"learning_rate": 0.01,
	"discount_factor": 0.9
}
R_free_parameters = {
	# "go_prob": 0.26, # Optimal go_prob for four players
	"go_prob": 0.5, # Optimal go_prob for four players
}

def top_panel():
	#-------------------------------
	# Create renders Qlearning
	#-------------------------------
	perf = Performer.simple_run(
		agent_class=Q_learning,
		fixed_parameters=fixed_parameters,
		free_parameters=QL_free_parameters,
		simulation_parameters=simulation_parameters,
		image_folder=image_folder,
		measures=['render'],
		seeds=[7,9],
		kwargs={'T':10}
	)
	#-------------------------------
	# Create renders Random
	#-------------------------------
	perf = Performer.simple_run(
		agent_class=Random,
		fixed_parameters=fixed_parameters,
		free_parameters=R_free_parameters,
		simulation_parameters=simulation_parameters,
		image_folder=image_folder,
		measures=['render'],
		seeds=[0],
		kwargs={'T':10}
	)

def center_panel():
	#-------------------------------
	# Define simulation parameters
	#-------------------------------
	list_dicts = [
		{
			'agent_class': Q_learning,
			'fixed_parameters': fixed_parameters,
			'free_parameters': QL_free_parameters,
			'simulation_parameters': simulation_parameters,
			'seed': 7
		},
		{
			'agent_class': Q_learning,
			'fixed_parameters': fixed_parameters,
			'free_parameters': QL_free_parameters,
			'simulation_parameters': simulation_parameters,
			'seed': 9
		},
		{
			'agent_class': Random,
			'fixed_parameters': fixed_parameters,
			'free_parameters': R_free_parameters,
			'simulation_parameters': simulation_parameters,
			'seed': 0
		}
	]	
	#-------------------------------
	# Create plots
	#-------------------------------
	perf = Performer.simple_vs(
		list_dicts=list_dicts,
		image_folder=image_folder,
		measures=['hist_states'],
		kwargs={
			'T': 1000,
			'model_names': {
				'Qlearning-7': 'A',
				'Qlearning-9': 'B',
				'Random-0': 'C'
			},
			'title': 'Distribution of states',
			'legend': 'Situation'
		}
	)
