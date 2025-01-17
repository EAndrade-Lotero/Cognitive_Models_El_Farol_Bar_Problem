from pathlib import Path

from Utils.interaction import Performer
from Classes.cognitive_model_agents import *

folder_name = 'Figs'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)


def para_fig_random():
	#-------------------------------
	# Define simulation parameters
	#-------------------------------
	fixed_parameters = {
		'num_agents': 2,
		'threshold': 0.5,
	}
	free_parameters = {
		"go_prob": 0.5,
	}
	simulation_parameters = {
		'num_rounds':10000,
		'num_episodes': 1,
		'verbose': 0,
	}
	agent_class = Random
	Performer.sweep(
		agent_class=agent_class,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters,
		simulation_parameters=simulation_parameters,
		sweep_parameter='go_prob',
		values=[0, 0.1, 0.3, 0.5],
		image_folder=image_folder,
		measures=['efficiency']
	)
	Performer.sweep(
		agent_class=agent_class,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters,
		simulation_parameters=simulation_parameters,
		sweep_parameter='num_agents',
		values=[2, 8],
		image_folder=image_folder,
		measures=['entropy', 'conditional_entropy'],
		kwargs={'T':np.infty}
	)
	Performer.simple_plots(
		agent_class=Random,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters,
		simulation_parameters=simulation_parameters,
		measures=['hist_states', 'hist_state_transitions'],
		kwargs={'T':np.infty},
		image_folder=image_folder
	)


def para_fig_PRW_vs_ARW():
	#-------------------------------
	# Define simulation parameters
	#-------------------------------
	fixed_parameters = {
		'num_agents': 8,
		'threshold': 0.5,
	}
	free_parameters = {
		'PRW': {
			"inverse_temperature": 16,
			"initial_reward_estimate_go": 0,
			"initial_reward_estimate_no_go": 0,
			"learning_rate": 0.1
		},
		'ARW': {
			"inverse_temperature": 16,
			"initial_luft_estimate": -1,
			"learning_rate": 0.01
		}
	}
	simulation_parameters = {
		'num_rounds': 100,
		'num_episodes': 100,
		'verbose': 0,
	}
	agent_class = CogMod
	# Performer.sweep(
	# 	agent_class=agent_class,
	# 	fixed_parameters=fixed_parameters,
	# 	free_parameters=free_parameters,
	# 	simulation_parameters=simulation_parameters,
	# 	sweep_parameter='agent_class',
	# 	values=[PayoffRescorlaWagner, AttendanceRescorlaWagner],
	# 	image_folder=image_folder,
	# 	measures=[
	# 		'efficiency', 
	# 		'inequality',
	# 		'entropy', 
	# 		'conditional_entropy'
	# 	]
	# )
	simulation_parameters['num_episodes'] = 1
	Performer.simple_plots(
		agent_class=PayoffRescorlaWagner,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters['PRW'],
		simulation_parameters=simulation_parameters,
		measures=['round_attendance'],
		image_folder=image_folder
	)
	Performer.simple_plots(
		agent_class=AttendanceRescorlaWagner,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters['ARW'],
		simulation_parameters=simulation_parameters,
		measures=['round_attendance'],
		image_folder=image_folder
	)


# Dibujo de rondas

# Plots de measures

# Tabla de MLEs

# Plot de comparación con QL_discount vs QL_no_discount

# Plot de comparación con QL_best_fit vs Human