from Classes.bar import Bar
from Classes.agents import epsilon_greedy
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from random import randint, seed
from itertools import product
from pathlib import Path

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'fairness')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'fairness')
data_folder.mkdir(parents=True, exist_ok=True)

def draw_bar_attendances_3P():
    # Define simulation parameters
    num_agents = 3
    threshold = .7
    num_rounds = 1000
    # Define agents
    epsilon = None
    belief_strength = 1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFP',\
                   num_rounds=num_rounds)
    # Run simulation --- perfect cooperation
    seed(62)
    df = episode.simulate()
    file = Path.joinpath(image_folder, 'drawbar_eq_coop_3P.png')
    episode.environment.render(file=file, num_rounds=10)

def draw_bar_attendances_2P():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 1000
    # Define agents
    epsilon = None
    belief_strength = 1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFP',\
                   num_rounds=num_rounds)
    # Run simulation
    seed(31)
    df = episode.simulate()
    file = Path.joinpath(image_folder, 'drawbar_eq_coop_2P.png')
    episode.environment.render(file=file, num_rounds=10)
    # Run simulation
    seed(6)
    df = episode.simulate()
    file = Path.joinpath(image_folder, 'drawbar_no_eq_coop_2P.png')
    episode.environment.render(file=file, num_rounds=10)

