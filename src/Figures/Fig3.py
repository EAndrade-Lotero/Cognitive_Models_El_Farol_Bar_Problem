import numpy as np
import pandas as pd
from pathlib import Path

from Utils.plot_utils import PlotsAndMeasures, BarRenderer
from Utils.interaction import Performer
from Classes.cognitive_model_agents import Random

folder_name = 'Fig3'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)
human_data_file = Path('..', 'data', 'human', '2-player-UR.csv')
data_folder = Path.cwd() / Path('..').resolve() / Path('data', folder_name)
data_folder.mkdir(parents=True, exist_ok=True)


def create_random_dataset() -> None:
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

create_random_dataset()
random_data = Path.joinpath(data_folder, 'Random.csv')
df_random = pd.read_csv(random_data)
df_random['model'] = 'Mixed Strategies'
df_random['treatment'] = 'No treatment'

df_human = pd.read_csv(human_data_file)
df_human['model'] = 'Human'
df_human['num_agents'] = 2
df_human['id_sim'] = df_human['room']
df_human['id_player'] = df_human['player']
df_human['decision'] = df_human['choice']
drop_cols = [
	'room', 'player', 'choice', 'num_players', 'group', 'ac_score'
]
df_human.drop(columns=drop_cols, inplace=True)


def top_panel():
	print('\tPlotting top panels...')
	df_ = pd.read_csv(human_data_file)
	#-------------------------------
	# Plot groups
	#-------------------------------
	groups = [
		'Grupo-0017', 'Grupo-0022', 'Grupo-0004',
		'Grupo-0021', 'Grupo-0008', 'Grupo-0006'
	]
	for group in groups:
		df = df_[df_['room'] == group]
		bar_renderer = BarRenderer(
			data=df,
			images_folder=image_folder
		)
		bar_renderer.render(num_rounds=30)


def center_panels():
	print('\tPlotting center panels...')
	df = df_human.copy()
	df['model'] = df['treatment']
	p = PlotsAndMeasures(df)
	p.width = 2
	p.height = 2.25
	measures = ['attendance', 'efficiency', 'inequality', 'entropy']
	# kwargs = {
	# 	'title_size':18
	# }
	# p.plot_measures(
	# 	measures=measures,
	# 	folder=image_folder
	# )
	p = PlotsAndMeasures(df_human)
	p.width = 2.5
	p.height = 2.75
	file = Path.joinpath(image_folder, 'Fig3_per_player_attendance_vs_efficiency.png')
	p.plot_attendance_vs_efficiency_per_player(file=file)


def bottom_panels():
	print('\tPlotting bottom panels...')
	df = pd.concat([df_random, df_human], ignore_index=True)
	p = PlotsAndMeasures(df)
	p.width = 3
	p.height = 3
	file = Path.joinpath(image_folder, 'Fig3_entropy_vs_efficiency.png')
	p.plot_efficiency_vs_entropy(file=file)
	file = Path.joinpath(image_folder, 'Fig3_inequality_vs_attendance.png')
	p.plot_inequality_vs_attendance(file=file)
	file = Path.joinpath(image_folder, 'Fig3_inequality_vs_efficiency.png')
	p.plot_inequality_vs_efficiency(file=file)
	file = Path.joinpath(image_folder, 'Fig3_attendance_vs_efficiency.png')
	p.plot_attendance_vs_efficiency(file=file)
	file = Path.joinpath(image_folder, 'Fig3_inequality_vs_entropy.png')
	p.plot_inequality_vs_entropy(file=file)




