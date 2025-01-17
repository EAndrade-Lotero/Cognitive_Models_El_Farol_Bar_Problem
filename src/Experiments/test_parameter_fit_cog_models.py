import numpy as np
import pandas as pd
from pathlib import Path

from Classes.agents import *
from Classes.cognitive_model_agents import *
from Classes.parameter_recovery import ParameterFit


hyperparameters = {
    'init_points':4,
    'n_iter':32
}

def test_parameter_fit_Random():
    print('\nTesting parameter fit for WSLS model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    fixed_parameters = [
        'num_agents', 
        'threshold'
    ]
    free_parameters = {
        'go_prob':0,
    }
    pf = ParameterFit(
        agent_class=Random,
        model_name='Random',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_WSLS():
    print('\nTesting parameter fit for Random model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    fixed_parameters = {
        'num_agents':2,
        'threshold':0.5,
    }
    free_parameters = {
        'inverse_temperature':10,
        'go_drive':0,
        'wsls_strength':0
    }
    pf = ParameterFit(
        agent_class=WSLS,
        model_name='WSLS',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_PRW():
    print('\nTesting parameter fit for PRW model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    fixed_parameters = {
        'num_agents':2,
        'threshold':0.5,
    }
    free_parameters = {
        'inverse_temperature':10,
        'initial_reward_estimate_go':0,
        'initial_reward_estimate_no_go':0,
        'learning_rate':0.1
    }
    pf = ParameterFit(
        agent_class=PayoffRescorlaWagner,
        model_name='PRW',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_ARW():
    print('\nTesting parameter fit for ARW model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    fixed_parameters = {
        'num_agents':2,
        'threshold':0.5,
    }
    free_parameters = {
        'inverse_temperature':10,
        'initial_luft_estimate':0,
        'learning_rate':0.1
    }
    pf = ParameterFit(
        agent_class=AttendanceRescorlaWagner,
        model_name='ARW',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_Qlearning():
    print('\nTesting parameter fit for Qlearning model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    fixed_parameters = {
        'num_agents':2,
        'threshold':0.5,
    }
    free_parameters = {
        'inverse_temperature':10,
        "go_drive": 0,
        "learning_rate": 0.001,
        "discount_factor": 0.8
    }
    pf = ParameterFit(
        agent_class=Q_learning,
        model_name='Qlearning',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_MFP():
    print('\nTesting parameter fit for MFP model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    num_agents = 2
    threshold = 0.5
    states = list(product([0,1], repeat=num_agents))
    count_states = ProxyDict(
        keys=states,
        initial_val=0
    )
    count_transitions = ProxyDict(
        keys=list(product(states, repeat=2)),
        initial_val=0
    )
    fixed_parameters = {
        'num_agents': num_agents,
        'threshold': threshold,
        "count_states": count_states,
        "count_transitions": count_transitions,
        "states":states,
        "designated_agent":False
    }
    free_parameters = {
        "inverse_temperature":1,
        'belief_strength':1,
        "go_drive":0.5,
    }
    pf = ParameterFit(
        agent_class=MFP,
        model_name='MFP',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_parameter_fit_MFPAgg():
    print('\nTesting parameter fit for MFPAgg model...')
    # file = Path('..', 'data', 'random_model', 'random-05.csv')
    file = Path('..', 'data', 'human', '2-player-UR.csv')
    print(f'Loading data from {file}...')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    num_agents = 2
    threshold = 0.5
    fixed_parameters = {
        'num_agents': num_agents,
        'threshold': threshold,
    }
    free_parameters = {
        "inverse_temperature":1,
        'belief_strength':1,
        "go_drive":0.5,
    }
    pf = ParameterFit(
        agent_class=MFPAgg,
        model_name='MFPAgg',
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)