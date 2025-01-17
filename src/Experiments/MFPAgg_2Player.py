from Classes.bar import Bar
from Classes.agents import AgentMFPAgg
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from random import randint, seed
from itertools import product
from pathlib import Path
from tqdm import tqdm

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFPAgg_2Player')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFPAgg_2Player')
data_folder.mkdir(parents=True, exist_ok=True)

def simple_draw_bar_attendances():
    seed(1)
    # Define simulation parameters
    num_agents = 10
    threshold = .5
    num_rounds = 10000
    # Define agents
    epsilon = 0
    belief_strength = 1
    states = list(product([0,1], repeat=2))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [AgentMFPAgg(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFPAgg',\
                   num_rounds=num_rounds)
    # Run simulation
    df = episode.simulate(verbose=0)
    file = Path.joinpath(image_folder, 'simple_attendance.png')
    episode.environment.render(file=file)

def simple_draw_scores():
    seed(1)
    # Define simulation parameters
    num_agents = 10
    threshold = .5
    num_rounds = 1000
    # Define agents
    epsilon = 0
    belief_strength = 1
    states = list(product([0,1], repeat=2))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [AgentMFPAgg(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFPAgg',\
                   num_rounds=num_rounds)
    # Run simulation
    df = episode.simulate(verbose=0)
    file = Path.joinpath(image_folder, 'simple_scores.png')
    p = PlotsAndMeasures(data=df)
    p.plot_scores(file=file)

def draw_suboptimal_attendances():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 1000
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
    semillas = [4, 6, 14, 31, 52]
    # semillas = range(100)
    for semilla in tqdm(semillas, desc='Running seeds...'):
        seed(semilla)
        # Run simulation
        df = episode.simulate()
        file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
        episode.environment.render(file=file)

def compare_cooldown():
    '''
    A parameter sweep to compare performance on several values of epsilon
    '''
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 1000
    num_episodes = 1000
    measures = ['score']
    # Define agents
    epsilon = 0
    belief_strength = 0.1
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
    values = [None, 0, 0.01, 0.1, 1]
    file = Path.joinpath(image_folder, 'sweep_epsilon.png')
    experiment.run_sweep1(parameter=parameter, values=values, file=file)
