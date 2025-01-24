import json
import pandas as pd
from pathlib import Path

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.agent_utils import ProxyDict
from Classes.cognitive_model_agents import *
from Classes.parameter_recovery import ParameterFit

# Create paths for data and results
data_folder = Path('..', 'data', 'human')
folder = Path('..', 'reports', 'MLE_2P')
folder.mkdir(parents=True, exist_ok=True)
best_fit_file = Path(folder, f'best_fit.json')

# Load data into a dataframe
file_names = [
	'2-player-UR.csv',
	'3-player-IU.csv',
	'4-player-IU.csv',
	'5-player-IU.csv',
	'6-player-IU.csv',
	'7-player-IU.csv',
	'8-player-IU.csv',
	'9-player-IU.csv',
	'11-player-IU.csv',
	'12-player-IU.csv',
]
df_list = list()
for file_name in file_names:
	file = data_folder / Path(file_name)
	print(f'Loading data from {file}...')
	df = pd.read_csv(file)
	df_list.append(df)
data = pd.concat(df_list, ignore_index=True)

# Create optimization hyperparameters
hyperparameters = {
    'init_points':4,
    'n_iter':16
}


def fit_models():
	best_fit = {
		'model_name': list(),
		'deviance': list(),
		'AIC': list(),
		'free_parameters': list()
	}
	# MODELS is a dictionary with the free parameters for 
	# each model and comes from Classes.cognitive_model_agents
	for model_name, model in MODELS.items():
		# if model_name not in ['QAttendance', 'QFairness']:
		# 	continue
		print(f'Fitting data to model {model_name}...')
		best_fit['model_name'].append(model_name)
		print('Creating parameter recovery class...')
		pf = ParameterFit(
			agent_class=model["class"],
			model_name=model_name,
			free_parameters=model["free_parameters"],
			data=data,
			optimizer_name='bayesian'
		)
		print('Running bayesian optimizer...')
		res = pf.get_optimal_parameters(hyperparameters)
		best_fit['deviance'].append(-res["target"])
		best_fit['AIC'].append(2*len(model["free_parameters"]) - 2*res["target"])
		best_fit['free_parameters'].append(res["params"])
	print(best_fit)
	with open(best_fit_file, 'w') as f:
		json.dump(best_fit, f)
	print(f'Model recovery data writen to file {best_fit_file}')


if __name__ == '__main__':
	fit_models()