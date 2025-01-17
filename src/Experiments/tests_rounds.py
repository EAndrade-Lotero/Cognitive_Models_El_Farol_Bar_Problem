from tqdm import tqdm
from pathlib import Path
from itertools import product

from Classes.bar import Bar
from Classes.agents import AgentMFP_Multi, AgentMFPAgg
from Utils.interaction import Episode, PlotsAndMeasures

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'tests_rounds')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'tests_rounds')
data_folder.mkdir(parents=True, exist_ok=True)

# Define simulation parameters
num_agents = 2
threshold = .75
num_rounds = 4

def test_MFPAGG():
	# Define agents
	belief_strength = 1
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
	# Create simulation
	episode = Episode(
		environment=bar,\
		agents=agents,\
		model='MFPAgg',\
		num_rounds=num_rounds
	)
	# Run simulation
	df = episode.simulate(verbose=0)
	file = Path.joinpath(image_folder, 'decisions_MFPAGG.png')
	episode.environment.render(file=file)
	p = PlotsAndMeasures(data=df)
	file = Path.joinpath(image_folder, 'attendance_MFPAGG.png')
	p.plot_round_attendance(file=file)
	file = Path.joinpath(image_folder, 'efficiency_MFPAGG.png')
	p.plot_round_efficiency(file=file)
	file = Path.joinpath(image_folder, 'inequality_MFPAGG.png')
	p.plot_round_inequality(file=file)
	file = Path.joinpath(image_folder, 'convergence_MFPAGG.png')
	p.plot_round_convergence(file=file)

def test_MFP():
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
	# Create simulation
	episode = Episode(
		environment=bar,\
		agents=agents,\
		model='MFPAgg',\
		num_rounds=num_rounds
	)
	# Run simulation
	df = episode.simulate(verbose=0)
	file = Path.joinpath(image_folder, 'decisions_MFP.png')
	episode.environment.render(file=file)
	p = PlotsAndMeasures(data=df)
	file = Path.joinpath(image_folder, 'attendance_MFP.png')
	p.plot_round_attendance(file=file)
	file = Path.joinpath(image_folder, 'efficiency_MFP.png')
	p.plot_round_efficiency(file=file)
	file = Path.joinpath(image_folder, 'inequality_MFP.png')
	p.plot_round_inequality(file=file)
	file = Path.joinpath(image_folder, 'convergence_MFP.png')
	p.plot_round_convergence(file=file)
