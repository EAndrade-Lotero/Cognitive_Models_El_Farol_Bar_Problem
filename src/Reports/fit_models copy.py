import json
import pandas as pd
from pathlib import Path

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.agent_utils import ProxyDict
from Classes.cognitive_model_agents import *
from Classes.parameter_recovery import ParameterFit, PPT

# file_name = '2-player-UR.csv'
file_name = '5-player-IU.csv'
# file_name = 'multi-player.csv'

file = Path('..', 'data', 'human', file_name)
folder = Path('..', 'reports', file_name)
folder.mkdir(parents=True, exist_ok=True)
best_fit_file = Path(folder, f'best_fit.json')

print(f'Loading data from {file}...')
data = pd.read_csv(file)

data['num_players'] = 2
data['threshold'] = 0.5
num_agents = data.num_players.unique()[0]
threshold = 0.5
states = list(product([0,1], repeat=num_agents))
count_states = ProxyDict(
	keys=states,
	initial_val=0
)
count_transitions = ProxyDict(
	keys=list(product(states, repeat=2)),
	initial_val=0
)

hyperparameters = {
    'init_points':4,
    'n_iter':32
}

fixed_parameters = {
	'num_agents': num_agents,
	'threshold': threshold,
}

models = {
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
		'fixed_parameters': {
			"count_states": count_states,
			"count_transitions": count_transitions,
			"states":states,
			"designated_agent": True
		},
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

def fit_models():
	best_fit = {
		'model_name': list(),
		'deviance': list(),
		'AIC': list(),
		'fixed_parameters': list(),
		'free_parameters': list()
	}
	for model_name, model in models.items():
		# if model_name not in ['MFP']:
		# 	continue
		print(f'Fitting data to model {model_name}...')
		params_list = PPT.get_fixed_parameters(data)
		for fixed_parameters in params_list:
			params = fixed_parameters.copy()
			best_fit['model_name'].append(model_name)
			best_fit['fixed_parameters'].append(fixed_parameters)
			num_agent_column = PPT.get_num_player_column(data.columns)
			num_ag, thres = list(params.values())
			print(f'Retrieving data with {num_ag} players and threshold {thres}')
			if num_ag > 2:
				thres = round(thres * num_ag, 0)
			df = data.groupby([num_agent_column, 'threshold']).get_group((num_ag, thres))
			if 'fixed_parameters' in model.keys():
				params.update(model["fixed_parameters"])
			print('Creating parameter recovery class...')
			pf = ParameterFit(
				agent_class=model["class"],
				model_name=model_name,
				fixed_parameters=params,
				free_parameters=model["free_parameters"],
				data=df,
				optimizer_name='bayesian'
			)
			print('Running bayesian optimizer...')
			res = pf.get_optimal_parameters(hyperparameters)
			print(res)
			best_fit['deviance'].append(-res["target"])
			best_fit['AIC'].append(2*len(model["free_parameters"]) - 2*res["target"])
			best_fit['free_parameters'].append(res["params"])
	print(best_fit)
	with open(best_fit_file, 'w') as f:
		json.dump(best_fit, f)
	print(f'Model recovery data writen to file {best_fit_file}')

def report_model_recovery():
	best_parameters = dict()
	AICs = dict()
	latex_string = '\n\n' + r'\noindent\textbf{Fixed parameters:}' + '\n\n'
	latex_string += PrintLaTeX.print_parameters(fixed_parameters, are_free=False)
	latex_string += '\n\n' + r'\vspace{\baselineskip}' + '\n\n'
	for model_name, model in models.items():
		if model_name != 'POOOOO':
			continue
		print(f'Fitting data to model {model_name}...')
		params = fixed_parameters.copy()
		if 'fixed_parameters' in model.keys():
			params.update(model["fixed_parameters"])
		print('Creating parameter recovery class...')
		pf = ParameterFit(
			agent_class=model["class"],
			model_name=model_name,
			fixed_parameters=params,
			free_parameters=model["free_parameters"],
			data=data,
			optimizer_name='bayesian'
		)
		print('Running bayesian optimizer...')
		res = pf.get_optimal_parameters(hyperparameters)
		print(res)
		AICs[model_name] = 2*len(model["free_parameters"]) - 2*res["target"]
		best_parameters[model_name] = res["params"]
		latex_string += '\n\n' + r'\noindent\textbf{' + model_name + '}:\n\n' 
		latex_string += PrintLaTeX.print_parameters(res["params"])

	latex_string += '\n\n' + r'\vspace{\baselineskip}' + '\n\n'
	latex_string += '\n\n' + r'\noindent\textbf{AICs}:' + '\n\n' 
	latex_string += PrintLaTeX.print_parameters(AICs, are_free=False)
    # #-------------------------------
    # # Add visual examination of best parameters
    # #-------------------------------
	# latex_string += draw_best_fit_model()
    #-------------------------------
    # Wrap and save
    #-------------------------------
	latex_string = PrintLaTeX.wrap_with_header_and_footer(latex_string)
	latex_file = Path.joinpath(folder, f'report_fit_{file.name}.tex')
	PrintLaTeX.save_to_file(
		latex_string=latex_string,
		latex_file=latex_file
	)

def draw_best_fit_model():
	simulation_parameters = {
		'num_rounds': 60,
		'num_episodes': 80,
		'verbose': 0,
	}
	free_parameters = {
		"inverse_temperature": 1,
		"go_drive": 1,
		"learning_rate": 1,
		"discount_factor": 1
	}
	latex_string = Performer.simple_run(
		agent_class=Q_learning,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters,
		simulation_parameters=simulation_parameters,
		seeds=None,
		# seeds=[6, 38, 87], # chosen for num_players = 2
		image_folder=folder
	)    
	return latex_string