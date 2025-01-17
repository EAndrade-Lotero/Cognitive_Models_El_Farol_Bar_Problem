from tqdm import tqdm
from pathlib import Path
from itertools import product

from Classes.bar import Bar
from Classes.agents import AgentMFP_Multi, AgentMFPAgg
from Utils.interaction import Episode, PlotsAndMeasures, Experiment

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'tests_simulations')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'tests_simulations')
data_folder.mkdir(parents=True, exist_ok=True)

# Define simulation parameters
num_agents = 4
threshold = .75
num_rounds = 100
num_episodes = 50

def sweep_MFP():
	belief_strength = 1
	list_num_agentes = [2, 3, 4, 5]
	# list_num_agentes = [4]
	# list_num_agentes = [10]
	for num_agents in tqdm(list_num_agentes, desc='Simulating for each num. agents'):
		# Define agents
		states = list(product([0,1], repeat=num_agents))
		alphas = {(x,y):1/len(states) for x in states for y in states}
		parameters = {
			"num_agents":num_agents,\
			"threshold":threshold,\
			"belief_strength":belief_strength,\
			"alphas":alphas,\
		}
		# Create experiment
		experiment = Experiment(
			agent_class=AgentMFP_Multi,
			num_rounds=num_rounds,
			num_episodes=num_episodes,
			measures=[
				'attendance', 
				'deviation', 
				'efficiency', 
				'inequality', 
				'convergence'
			]
		)
		# Define image folder 
		image_folder_ = image_folder / Path('MFP', f'{num_agents}_agents')
		image_folder_.mkdir(parents=True, exist_ok=True)
		experiment.run_sweep1(
			parameter='threshold',
			values=[0.25, 0.5, 0.75],
			file=image_folder_,
			kwargs={
				'x_label':'Threshold',
				'only_value':True
			}
		)


def sweep_MFPAgg():
	belief_strength = 1
	list_num_agentes = [2, 3, 4, 5, 6, 10, 20]
	# list_num_agentes = [4]
	for num_agentes in tqdm(list_num_agentes, desc='Simulating for each num. agents'):
		# Define agents
		states = list(product([0,1], repeat=2))
		alphas = {(x,y):1/len(states) for x in states for y in states}
		parameters = {
			"num_agents":num_agents,\
			"threshold":threshold,\
			"belief_strength":belief_strength,\
			"alphas":alphas,\
		}
		agents = [AgentMFPAgg(parameters, n) for n in range(num_agents)]
		# Create bar
		bar = Bar(
			num_agents=num_agents, 
			threshold=threshold
		)
		# Create experiment
experiment = Experiment(
	agent_class=agent_class,
			num_rounds=num_rounds,
			num_episodes=num_episodes,
			measures=[
				'attendance', 
				'deviation', 
				'efficiency', 
				'inequality', 
				'convergence'
			]
		)
		# Define image folder 
		image_folder_ = image_folder / Path('MFPAgg', f'{num_agentes}_agents')
		image_folder_.mkdir(parents=True, exist_ok=True)
		experiment.run_sweep1(
			parameter='threshold',
			values=[0.25, 0.5, 0.75],
			file=image_folder_,
			kwargs={
				'x_label':'Threshold',
				'only_value':True
			}
		)


def test_experiment_MPF():
	# Define agents
	belief_strength = 1
	states = list(product([0,1], repeat=num_agents))
	alphas = {(x,y):1/len(states) for x in states for y in states}
	parameters = {
		"num_agents":num_agents,\
		"threshold":threshold,\
		"belief_strength":belief_strength,\
		"alphas":alphas,\
	}
	agents = [AgentMFP_Multi(parameters, n) for n in range(num_agents)]
	# Create bar
	bar = Bar(
		num_agents=num_agents, 
		threshold=threshold
	)
	# Create experiment
	experiment = Experiment(
		environment=bar,
		agents=agents,
		num_rounds=num_rounds,
		num_episodes=num_episodes,
		measures=[
			'attendance', 
			'deviation', 
			'efficiency', 
			'inequality', 
			'convergence'
		]
	)
	experiment.run_sweep1(
		parameter='threshold',
		values=[0.25, 0.5, 0.75],
		file=image_folder,
		kwargs={
			'x_label':'Threshold',
			'only_value':True
		}
	)

