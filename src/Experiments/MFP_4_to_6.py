import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from random import randint, seed
from prettytable import PrettyTable

from Classes.bar import Bar
from Classes.agents import softmax_greedy #AgentMFPMultiSameTransProb #AgentMFP_Multi, epsilon_greedy
from Utils.interaction import (
    Episode, PlotsAndMeasures, Experiment, Performer
)
from Classes.agent_utils import ProxyDict
from Utils.parameter_optimization import GetMeasure, ParameterOptimization

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'MFP_4_to_6')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'MFP_4_to_6')
data_folder.mkdir(parents=True, exist_ok=True)


def very_simple_run():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    belief_strength = 1
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    simulation_parameters = {
        'num_rounds':5,
        'verbose':4
    }
    Performer.examine_simulation(
        agent_class=softmax_greedy,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        semilla=1
    )


def simple_run():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    belief_strength = 1
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    simulation_parameters = {
        'num_rounds':1000,
        'verbose':0
    }
    # Simple run
    Performer.simple_run(
        agent_class=softmax_greedy,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        measures=['render'],
        image_folder=image_folder
    )


def sweep_belief_strength():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    belief_strength = 1
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    simulation_parameters = {
        'num_rounds':500,
        'num_episodes':100,
        'verbose':0
    }
    Performer.sweep(
            agent_class=softmax_greedy,
            fixed_parameters=fixed_parameters,
            free_parameters=free_parameters,
            simulation_parameters=simulation_parameters,
            sweep_parameter='belief_strength',
            values=[4, 8, 16, 32],
            image_folder=image_folder,
            measures=[
				'attendance', 
				'efficiency', 
				'inequality', 
				'entropy'
            ]
        )


def sweep_num_agents():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    belief_strength = 1
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    simulation_parameters = {
        'num_rounds':500,
        'num_episodes':100,
        'verbose':0
    }
    Performer.sweep(
            agent_class=softmax_greedy,
            fixed_parameters=fixed_parameters,
            free_parameters=free_parameters,
            simulation_parameters=simulation_parameters,
            sweep_parameter='num_agents',
            values=[4, 5, 9, 10],
            image_folder=image_folder,
            measures=[
				'attendance', 
				'efficiency', 
				'inequality', 
				'entropy'
            ]
        )


def sweep_inverse_temperature():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    belief_strength = 8
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    simulation_parameters = {
        'num_rounds':500,
        'num_episodes':100,
        'verbose':0
    }
    Performer.sweep(
            agent_class=softmax_greedy,
            fixed_parameters=fixed_parameters,
            free_parameters=free_parameters,
            simulation_parameters=simulation_parameters,
            sweep_parameter='inverse_temperature',
            values=[0, 16, 32, 64],
            image_folder=image_folder,
            measures=[
				'attendance', 
				'efficiency', 
				'inequality', 
				'entropy'
            ]
        )


def find_best_parameters():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    belief_strength = 8
    inverse_temperature = 32
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    free_parameters = {
        "belief_strength":belief_strength,
        "inverse_temperature":inverse_temperature,
    }
    po = ParameterOptimization(
        agent_class=softmax_greedy,
        free_parameters=free_parameters,
        fixed_parameters=fixed_parameters,
        measure_class=GetMeasure,
        measure='efficiency',
        optimizer_name='bayesian'
    )
    result = po.get_optimal_parameters()
    print(result)


def simulate_best_parameters():
    # Define simulation parameters
    num_agents = 4
    threshold = .5
    states, alphas, count_states, count_transitions, trans_probs = create_proxy_dicts(num_agents)
    fixed_parameters = {
        "num_agents":num_agents,
        "threshold":threshold,
        "states":states,
        "alphas":alphas,
        "trans_probs":trans_probs,
        "count_states":count_states,
        "count_transitions":count_transitions,
        "designated_agent":False
    }
    simulation_parameters = {
        'num_rounds':1000,
        'num_episodes':100,
        'verbose':0
    }
    # Best performing parameters
    free_parameters = {
        "belief_strength":1,
        "inverse_temperature":64,
    }
    Performer.simple_plots(
        agent_class=softmax_greedy,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        data_folder=data_folder
    )


def plot_two_variate_kdes():
    data_file = Path.joinpath(data_folder, 'MFP(softmax).csv')
    df = pd.read_csv(data_file, index_col=False)
    df['model'] = 'MFP(softmax)'
    p = PlotsAndMeasures(df)
    p.width = 3
    p.height = 3
    file = Path.joinpath(image_folder, 'entropy_vs_efficiency.png')
    p.plot_efficiency_vs_entropy(file=file)
    file = Path.joinpath(image_folder, 'inequality_vs_attendance.png')
    p.plot_inequality_vs_attendance(file=file)
    file = Path.joinpath(image_folder, 'inequality_vs_efficiency.png')
    p.plot_inequality_vs_efficiency(file=file)
    file = Path.joinpath(image_folder, 'attendance_vs_efficiency.png')
    p.plot_attendance_vs_efficiency(file=file)
    file = Path.joinpath(image_folder, 'inequality_vs_entropy.png')
    p.plot_inequality_vs_entropy(file=file)


def create_proxy_dicts(num_agents):
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
    return states, alphas, count_states, count_transitions, trans_probs