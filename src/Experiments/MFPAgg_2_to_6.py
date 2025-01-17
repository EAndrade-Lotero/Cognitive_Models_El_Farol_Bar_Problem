from tqdm import tqdm
from pathlib import Path
from itertools import product
from random import randint, seed
from prettytable import PrettyTable

from Classes.bar import Bar
from Classes.agents import AgentMFPAgg
from Utils.interaction import Episode
from Classes.agent_utils import TransitionsFrequencyMatrix

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFPAgg_2_to_6')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFPAgg_2_to_6')
data_folder.mkdir(parents=True, exist_ok=True)

def draw_attendances():
    # Define simulation parameters
    threshold = .7
    num_rounds = 100
    # Define agents
    epsilon = 0
    belief_strength = 1
    for num_agents in tqdm([5], desc='Iterating over num agents'):
        states = list(product([0,1], repeat=2))
        alphas = {(x,y):1/len(states) for x in states for y in states}
        parameters = {
            "belief_strength":belief_strength,\
            "alphas":alphas,\
            "num_agents":num_agents,\
            "threshold":threshold,\
            "epsilon": epsilon
        }
        agents = [AgentMFPAgg(parameters, n) for n in range(num_agents)]
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
        semillas = [3]
        # semillas = range(15)
        for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
            seed(semilla)
            # Run simulation
            df = episode.simulate()
            file = Path.joinpath(image_folder, f'NPAgg{num_agents}_attendance_{semilla}.png')
            # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
            episode.environment.render(file=file)
            for agent in episode.agents:
                A = TransitionsFrequencyMatrix(2)
                A.from_dict(agent.trans_probs)
                table_A = PrettyTable([''] + [str(s) for s in states])
                for prev_state in states:
                    dummies = [round(A((prev_state, state)),2) for state in states]
                    table_A.add_row([str(prev_state)] + dummies)
                print(table_A)                
