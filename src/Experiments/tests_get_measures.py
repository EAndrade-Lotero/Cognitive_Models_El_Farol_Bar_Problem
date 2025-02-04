import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from Classes.cognitive_model_agents import Random
from Utils.utils import GetMeasurements
from Utils.interaction import Performer
from Utils.plot_utils import PlotsAndMeasures

human_data_file = Path('..', 'data', 'human', '2-player-UR.csv')
image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'test')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'Random')
data_folder.mkdir(parents=True, exist_ok=True)

random_data = Path.joinpath(data_folder, 'Random.csv')
df_random = pd.read_csv(random_data)
df_random['model'] = 'Mixed Strategies'
df_random['treatment'] = 'No treatment'

df_human = pd.read_csv(human_data_file)
df_human['model'] = 'human'
df_human['num_agents'] = 2
df_human['id_sim'] = df_human['room']
df_human['id_player'] = df_human['player']
df_human['decision'] = df_human['choice']
drop_cols = [
	'room', 'player', 'choice', 'num_players', 'group', 'ac_score'
]
df_human.drop(columns=drop_cols, inplace=True)

def test_measures():
	measures = [
		'attendance', 'efficiency',
		'inequality', 'entropy'
	]
	gm = GetMeasurements(df_human, measures)
	df_measures = gm.get_measurements()
	print(df_measures)

def test_kde_per_player():
	p = PlotsAndMeasures(df_human)
	file = Path.joinpath(image_folder, 'per_player_attendance_vs_efficiency.png')
	p.plot_attendance_vs_efficiency_per_player(file=file)
	file = Path.joinpath(image_folder, 'group_attendance_vs_efficiency.png')
	p.plot_attendance_vs_efficiency(file=file)


def test_kde():
	df = pd.concat([df_random, df_human], ignore_index=True)
	p = PlotsAndMeasures(df)
	file = Path.joinpath(image_folder, 'entropy_vs_efficiency.png')
	p.plot_efficiency_vs_entropy(file=file)
	file = Path.joinpath(image_folder, 'inequality_vs_attendance.png')
	p.plot_inequality_vs_attendance(file=file)
	file = Path.joinpath(image_folder, 'inequality_vs_efficiency.png')
	p.plot_inequality_vs_efficiency(file=file)
	file = Path.joinpath(image_folder, 'attendance_vs_efficiency.png')
	p.plot_attendance_vs_efficiency(file=file)
	file = Path.joinpath(image_folder, 'inequality_vs_entropy.png')
	p.plot_inequality_vs_entropy(file=file)


def test_kde_treatment():
	df = df_human.copy()
	df['model'] = df['treatment']
	p = PlotsAndMeasures(df_human)
	measures = ['attendance', 'efficiency', 'inequality', 'entropy']
	p.plot_measures(
		measures=measures,
		folder=image_folder
	)


def create_random_dataset():
	#-------------------------------
	# Set parameters
	#-------------------------------
	fixed_parameters = {
		'num_agents': 2,
		'threshold': 0.5,
	}
	simulation_parameters = {
		'num_rounds': 50,
		'num_episodes': 100,
		'verbose': 0,
	}
	free_parameters = {
		# "go_prob": 0.26, # Optimal go_prob for four players
		"go_prob": 0.31, # Optimal go_prob for two players
		# "go_prob": 0.5, # Optimal go_prob for four players
	}	
	#-------------------------------
	# Create dataset from random
	#-------------------------------
	Performer.simple_plots(
		agent_class=Random,
		fixed_parameters=fixed_parameters,
		free_parameters=free_parameters,
		simulation_parameters=simulation_parameters,
		data_folder=data_folder
	)