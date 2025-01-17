import torch
from tqdm import tqdm
from pathlib import Path
from itertools import product
from random import randint, seed

from Classes.bar import Bar
from Classes.agents import AgentNN
from Classes.agent_utils import MLP
from Utils.interaction import Episode

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFP_NN')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFP_NN')
data_folder.mkdir(parents=True, exist_ok=True)

def draw_attendances():
    # Define simulation parameters
    threshold = .7
    num_rounds = 1000
    # Define agents
    epsilon = None
    alpha = 1e-1
    for num_agents in tqdm([5], desc='Iterating over num agents'):
        trans_probs = MLP(
            sizes=[num_agents, 2 * num_agents, 2 * num_agents, num_agents],
            intermediate_activation_function=torch.nn.Tanh(),
            last_activation_function=torch.nn.Sigmoid(),
            alpha=alpha
        )
        parameters = {
            "belief_strength":0,\
            "alphas":dict(),\
            "num_agents":num_agents,\
            "threshold":threshold,\
            "epsilon": epsilon,
            "count_states": None,
            "count_transitions": None,
            "trans_probs": trans_probs,
            "alpha":alpha
        }
        agents = [AgentNN(parameters, n, False) for n in range(num_agents)]
        agents[0].designated_agent = True
        # agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
        # Create bar
        bar = Bar(num_agents=num_agents, threshold=threshold)
        # Create simulation
        episode = Episode(
            environment=bar,\
            agents=agents,\
            model='MFP-NN',\
            num_rounds=num_rounds
        )
        # set seed
        semillas = [1]
        # semillas = range(15)
        for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
            seed(semilla)
            # Run simulation
            df = episode.simulate(verbose=True)
            file = Path.joinpath(image_folder, f'NP{num_agents}_attendance_{semilla}.png')
            # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
            episode.environment.render(file=file)
