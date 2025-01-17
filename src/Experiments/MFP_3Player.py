from Classes.bar import Bar
from Classes.agents import epsilon_greedy
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from random import randint, seed
from itertools import product
from pathlib import Path
from tqdm import tqdm

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFP_3Player')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFP_3Player')
data_folder.mkdir(parents=True, exist_ok=True)

def simple_draw_bar_attendances():
    # Define simulation parameters
    num_agents = 3
    threshold = .7
    num_rounds = 20
    # Define agents
    epsilon = 0
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
    df = episode.simulate(verbose=4)
    # file = Path.joinpath(image_folder, 'simple_attendance.png')
    # episode.environment.render(file=file)

def draw_optimal_attendances():
    # Define simulation parameters
    num_agents = 3
    threshold = .7
    num_rounds = 10
    # Define agents
    epsilon = 0
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

    # set seed
    # semillas = [1, 2, 5]
    semillas = range(100)
    for semilla in tqdm(semillas, desc='Running seeds...'):
        # seed(semilla)
        # Run simulation
        df = episode.simulate()
        file = Path.joinpath(image_folder, f'attendance_{semilla}.png')
        # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
        episode.environment.render(file=file)

def compare_cooldown():
    '''
    A parameter sweep to compare performance on several values of epsilon
    '''
    # Define simulation parameters
    num_agents = 3
    threshold = .7
    num_rounds = 100
    num_episodes = 100
    measures = ['eq_coop']
    # Define agents
    epsilon = 0
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
    # Create experiment
    experiment = Experiment(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,\
                   num_episodes=num_episodes,\
                   measures=measures,\
                    parameters=parameters)
    # Run sweep
    parameter = 'epsilon'
    values = [None, 0, 0.05, 0.1]
    file = Path.joinpath(image_folder, 'sweep_epsilon_eq_coop.png')
    experiment.run_sweep1(parameter=parameter, values=values, file=file)
