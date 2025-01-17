from Classes.bar import Bar
from Classes.agent_bkup import epsilon_greedy
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from random import randint, seed
from itertools import product
from pathlib import Path
from tqdm import tqdm
import pandas as pd

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFP_2Player')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFP_2Player')
data_folder.mkdir(parents=True, exist_ok=True)

def simple_draw_bar_attendances():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 20
    # Define agents
    epsilon = 0
    belief_strength = 1
    states = list(product([0,1], repeat=num_agents))
    #alphas = {(x,y):1/len(states) for x in states for y in states}
    #alphas = {(x,y):1 if y[0] == 1 else 0 for x in states for y in states}
    alphas = {(x,y):1 if (x[0] == 1 - y[0] and x[1] == 1 - y[1]) else (1 if y[0] == 1 else 0) for x in states for y in states}
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
    df = episode.simulate()
    file = Path.joinpath(image_folder, 'simple_attendance.png')
    episode.environment.render(file=file)

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
    num_rounds = 300
    num_episodes = 1000
    measures = [
        'round_attendance',
        'round_efficiency'
    ]
    # Define agents
    epsilon = 0
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    #alphas = {(x,y):1 if y[0] == 1 else 0 for x in states for y in states}
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
    experiment = Experiment(
        environment=bar,\
        agents=agents,\
        num_rounds=num_rounds,\
        num_episodes=num_episodes,\
        measures=measures,\
        # parameters=parameters
    )
    # Run sweep
    parameter = 'epsilon'
    values = [None, 0, 0.01, 0.1]
    file = Path.joinpath(image_folder, 'sweep_epsilon_bkup.png')
    experiment.run_sweep1(
        parameter=parameter, 
        values=values, 
        folder_plots=image_folder
    )

def analyze_eq_coop():
    '''
    A parameter sweep to compare eq coop score
    '''
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 600
    num_episodes = 10
    measures = ['eq_coop']
    # Define agents no coop
    epsilon = 0
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1 if y[0] == 1 else 0 for x in states for y in states}
    #alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Episode(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,
                   model='no_coop')
    
    df_no_coop = experiment.simulate(num_episodes=num_episodes)

    # Define agents coop
    epsilon = 0
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1 if (x[0] == 1 - y[0] and x[1] == 1 - y[1]) else 0 for x in states for y in states}
    #alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Episode(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,
                   model='coop')
    
    df_coop = experiment.simulate(num_episodes=num_episodes)

    # Define random agents
    epsilon = 1
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    #alphas = {(x,y):1 if y[0] == 1 else 0 for x in states for y in states}
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
    experiment = Episode(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,
                   model='random')
    
    df_random = experiment.simulate(num_episodes=num_episodes)
    
    # Define MFP agents
    epsilon = None
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    #alphas = {(x,y):1 if y[0] == 1 else 0 for x in states for y in states}
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
    experiment = Episode(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,
                   model='MFP')
    
    df_MFP = experiment.simulate(num_episodes=num_episodes)
    
    df = pd.concat([df_no_coop, df_random, df_MFP], ignore_index=True)

    p = PlotsAndMeasures(df)
    file = Path.joinpath(image_folder, 'eq_coop.png')
    p.plot_EQ_scores(file=file, mu=threshold)
