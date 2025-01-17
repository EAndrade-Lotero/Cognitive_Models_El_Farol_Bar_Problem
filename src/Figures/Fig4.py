import numpy as np
import pandas as pd
from pathlib import Path

from Utils.plot_utils import PlotsAndMeasures, BarRenderer
from Utils.interaction import Performer
from Classes.cognitive_model_agents import Random

folder_name = 'Fig4'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)
human_data_file = Path('..', 'data', 'human', '2-player-UR.csv')
data_folder = Path.cwd() / Path('..').resolve() / Path('data', folder_name)
data_folder.mkdir(parents=True, exist_ok=True)


def top_panel():
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


def bottom_panels():
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
	random_data = Path.joinpath(data_folder, 'Random.csv')
	df_random = pd.read_csv(random_data)
	df_random['model'] = 'Mixed Strategies'
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
	df = pd.concat([df_random, df_human], ignore_index=True)
	p = PlotsAndMeasures(df)
	measures = [
		'attendance',
		'efficiency',
		'inequality',
		'entropy'
	]
	kwargs={
		'T': 30,
	}
	list_images = p.plot_measures(					
		folder=image_folder,
		measures=measures,
		kwargs=kwargs
	)

