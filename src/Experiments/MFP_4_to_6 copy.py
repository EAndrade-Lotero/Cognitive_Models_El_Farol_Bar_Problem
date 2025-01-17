from tqdm import tqdm
from pathlib import Path
from itertools import product
from random import randint, seed
from prettytable import PrettyTable

from Classes.bar import Bar
from Classes.agents import AgentMFP_Multi, epsilon_greedy
from Utils.interaction import (
    Episode, PlotsAndMeasures, Experiment, Performer
)
from Classes.agent_utils import ProxyDict

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFP_4_to_6')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFP_4_to_6')
data_folder.mkdir(parents=True, exist_ok=True)


def simple_run():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    num_rounds = 500
    epsilon = None
    belief_strength = 1
    # Define agents
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    count_states = ProxyDict(
        keys=states,
        initial_val=0
    )
    count_transitions = ProxyDict(
        keys=list(product(states, repeat=2)),
        initial_val=0
    )
    trans_probs = ProxyDict(
        keys=list(product(states, repeat=2)),
        initial_val=1/len(states)
    )
    parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "epsilon":epsilon,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "states":states,
        "designated_agent":False,
        "belief_strength":belief_strength,
        "alphas":alphas
    }
    agents = [AgentMFP_Multi(parameters, n) for n in range(num_agents)]
    agents[0].designated_agent = True
    # agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(
        environment=bar,\
        agents=agents,\
        model='MFP',\
        num_rounds=num_rounds
    )
    # set seed
    semillas = [36, 49]
    # semillas = range(100)
    for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
        seed(semilla)
        # Run simulation
        df = episode.simulate(verbose=0)
        # file = Path.joinpath(image_folder, f'NP{num_agents}_attendance_{semilla}.png')
        # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
        # episode.environment.render(file=file)
        B = episode.agents[0].trans_probs
        table_B = PrettyTable([''] + [str(s) for s in states])
        for prev_state in states:
            dummies = [round(B((prev_state, state)),2) for state in states]
            table_B.add_row([str(prev_state)] + dummies)
        print('Seed:', semilla)
        print(table_B)
        # p = PlotsAndMeasures(df)
        # file_att = Path.joinpath(image_folder, f'Round_attendance_{semilla}.png')
        # p.plot_round_attendance(file=file_att)

def draw_attendances():
    # Define simulation parameters
    threshold = .5
    num_rounds = 500
    epsilon = None
    belief_strength = 1
    # Define agents
    for num_agents in tqdm([10], desc='Iterating over num agents'):
        states = list(product([0,1], repeat=num_agents))
        alphas = {(x,y):1/len(states) for x in states for y in states}
        # trans_probs = TransitionsFrequencyMatrix(num_agents)
        # count_states = BinaryCounter1D(num_agents)
        # count_transitions = BinaryCounter2D(num_agents)
        count_states = ProxyDict(
            keys=states,
            initial_val=0
        )
        count_transitions = ProxyDict(
            keys=list(product(states, repeat=2)),
            initial_val=0
        )
        trans_probs = ProxyDict(
            keys=list(product(states, repeat=2)),
            initial_val=1/len(states)
        )
        parameters = {
            "belief_strength":belief_strength,\
            "alphas":alphas,\
            "num_agents":num_agents,\
            "threshold":threshold,\
            "epsilon": epsilon,
            "trans_probs": trans_probs,
            "count_states": count_states,
            "count_transitions": count_transitions,
            "states":states,
            "designated_agent":False
        }
        agents = [AgentMFP_Multi(parameters, n) for n in range(num_agents)]
        agents[0].designated_agent = True
        # agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
        # Create bar
        bar = Bar(num_agents=num_agents, threshold=threshold)
        # Create simulation
        episode = Episode(
            environment=bar,\
            agents=agents,\
            model='MFP',\
            num_rounds=num_rounds
        )
        # set seed
        # semillas = [1]
        semillas = range(15)
        for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
            seed(semilla)
            # Run simulation
            df = episode.simulate()
            file = Path.joinpath(image_folder, f'NP{num_agents}_attendance_{semilla}.png')
            # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
            episode.environment.render(file=file)

def sweep_epsilon():
    # Define simulation parameters
    num_agents = 5
    threshold = .5
    num_rounds = 1000
    num_episodes = 1000
    epsilon = 0
    belief_strength = 0.1
    # Define agents
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    # trans_probs = TransitionsFrequencyMatrix(num_agents)
    # count_states = BinaryCounter1D(num_agents)
    # count_transitions = BinaryCounter2D(num_agents)
    count_states = ProxyDict(
        keys=states,
        initial_val=0
    )
    count_transitions = ProxyDict(
        keys=list(product(states, repeat=2)),
        initial_val=0
    )
    trans_probs = ProxyDict(
        keys=list(product(states, repeat=2)),
        initial_val=1/len(states)
    )
    parameters = {
        "belief_strength": belief_strength,\
        "alphas": alphas,\
        "num_agents": num_agents,\
        "threshold": threshold,\
        "epsilon": epsilon,
        "trans_probs": trans_probs,
        "count_states": count_states,
        "count_transitions": count_transitions,
		"states":states,
		"designated_agent":False
    }
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    agents[0].designated_agent = True
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Experiment(
        environment=bar,
        agents=agents,
        num_rounds=num_rounds,
        num_episodes=num_episodes,
        measures=[
            'round_attendance',
            'round_efficiency',
            # 'attendance', 
            # 'deviation', 
            # 'efficiency', 
            # 'inequality', 
            # 'convergence',
        ]
    )
    # Define range of sweep
    # epsilons = [0, 0.001]
    epsilons = [None, 0.001, 0.01, 0.1]
    # Run sweep
    experiment.run_sweep1(
        parameter='epsilon',
        values=epsilons,
        folder_plots=image_folder,
        file_data=data_folder / Path('data.csv'),
        kwargs={
            'x_label':'Epsilon',
            'only_value':True
        }
    )

def sweep_belief_strength():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 1000
    num_episodes = 100
    epsilon = 0
    belief_strength = 1
    # Define agents
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    trans_probs = TransitionsFrequencyMatrix(num_agents)
    count_states = BinaryCounter1D(num_agents)
    count_transitions = BinaryCounter2D(num_agents)
    parameters = {
        "belief_strength": belief_strength,\
        "alphas": alphas,\
        "num_agents": num_agents,\
        "threshold": threshold,\
        "epsilon": epsilon,
        "trans_probs": trans_probs,
        "count_states": count_states,
        "count_transitions": count_transitions,
        "designated_agent": False
    }
    agents = [AgentMFP_Multi(parameters, n) for n in range(num_agents)]
    agents[0].designated_agent = True
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Experiment(
        environment=bar,
        agents=agents,
        num_rounds=num_rounds,
        num_episodes=num_episodes,
        measures=[
            # 'round_attendance',
            # 'round_efficiency',
            # 'attendance', 
            # 'deviation', 
            # 'efficiency', 
            # 'inequality', 
            # 'convergence',
        ]
    )
    # Run sweep
    experiment.run_sweep1(
        parameter='belief_strength',
        values=[0.5, 1, 10, 100],
        folder_plots=image_folder,
        file_data=data_folder / Path('data.csv'),
        kwargs={
            'x_label':'Belief Strength',
            'only_value':True
        }
    )