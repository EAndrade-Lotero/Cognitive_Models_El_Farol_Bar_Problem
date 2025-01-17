import pandas as pd
import numpy as np
from pathlib import Path

from Classes.agents import *
from Classes.parameter_recovery import GetDeviance, ParameterFit


def test_parameter_fit():
    print('\nTesting parameter fit...')
    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-05.csv')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    parameters = {
        'num_agents':2,
        'go_prob':0
    }
    pf = ParameterFit(
        agent_class=Random,
        model_name='Random',
        parameters=parameters,
        data=data,
        optimizer_name='bayesian'
    )
    print('Running bayesian optimizer...')
    hyperparameters = {
        'init_points':4,
        'n_iter':32
    }
    res = pf.get_optimal_parameters(hyperparameters)
    print(res)


def test_dev_random_p(p:float):
    print(f'\nTesting with model p={p}...')
    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-05.csv')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    parameters = {
        'go_prob':p
    }
    dict_parameters = {"parameters":parameters}
    pr = GetDeviance(
        model=Random,
        parameters=dict_parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)


def test_dev_random():

    print('\nTesting with model p=0...')
    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-0.csv')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    parameters = {
        'go_prob':0
    }
    pr = GetDeviance(
        model=Random,
        parameters=parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)
    # print(pr.log_likelihoods)
    # likelihood = np.exp(np.sum([value for key, value in pr.log_likelihoods.items()]))
    # deviance = pr.get_deviance()
    # print('Deviance:', deviance, '--- Likelihood:', likelihood)
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, 9.5367431640625e-07))
    assert(np.isclose(deviance, 27.725887222397812))
    print('First test passed!')

    print('\nTesting with model p=1...')
    print('Creating parameter recovery class...')
    parameters = {
        'go_probability':1
    }
    pr = GetDeviance(
        model=Random,
        parameters=parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)
    # pr.get_likelihood_from_data()
    # print(pr.log_likelihoods)
    # likelihood = np.exp(np.sum([value for key, value in pr.log_likelihoods.items()]))
    # deviance = pr.get_deviance()
    # print('Deviance:', deviance, '--- Likelihood:', likelihood)
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, 0))
    assert(np.isclose(deviance, np.inf))
    print('Second test passed!')    

    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-1.csv')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    parameters = {
        'go_probability':1
    }
    pr = GetDeviance(
        model=Random,
        parameters=parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)
    # pr.get_likelihood_from_data()
    # print(pr.log_likelihoods)
    # likelihood = np.exp(np.sum([value for key, value in pr.log_likelihoods.items()]))
    # deviance = pr.get_deviance()
    # print('Deviance:', deviance, '--- Likelihood:', likelihood)
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, 9.5367431640625e-07))
    assert(np.isclose(deviance, 27.725887222397812))
    print('Third test passed!')

    print('\nTesting with model p=0.5...')
    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-05.csv')
    data = pd.read_csv(file)
    print('=>', data['decision'].mean())
    print('Creating parameter recovery class...')
    parameters = {
        'go_probability':0.5
    }
    pr = GetDeviance(
        model=Random,
        parameters=parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)
    # pr.get_likelihood_from_data()
    # print(pr.log_likelihoods)
    # likelihood = np.exp(np.sum([value for key, value in pr.log_likelihoods.items()]))
    # deviance = pr.get_deviance()
    # print('Deviance:', deviance, '--- Likelihood:', likelihood)
    num_episodes = len(data['id_sim'].unique())
    num_rounds = len(data['round'].unique())
    num_players = len(data['id_player'].unique())
    # assert(np.isclose(likelihood, np.power(0.5, num_episodes * num_rounds * num_players))), f'{likelihood} != {np.power(0.5, num_rounds * num_players)}'
    assert(np.isclose(deviance, -2 * np.log(np.power(0.5, num_episodes * num_rounds * num_players)))), f'{deviance} != {-2 * np.log(np.power(0.5, num_rounds * num_players))}'
    print('Fourth test passed!')

    print('Loading data...')
    file = Path('..', 'data', 'random_model', 'random-1.csv')
    data = pd.read_csv(file)
    print('Creating parameter recovery class...')
    pr = GetDeviance(
        model=Random,
        parameters=parameters,
        data=data
    )
    deviance = pr.get_deviance_from_data()
    print('Deviance:', deviance)
    # pr.get_likelihood_from_data()
    # print(pr.log_likelihoods)
    # likelihood = np.exp(np.sum([value for key, value in pr.log_likelihoods.items()]))
    # deviance = pr.get_deviance()
    # print('Deviance:', deviance, '--- Likelihood:', likelihood)
    num_episodes = len(data['id_sim'].unique())
    num_rounds = len(data['round'].unique())
    num_players = len(data['id_player'].unique())
    # assert(np.isclose(likelihood, np.power(0.5, num_episodes * num_rounds * num_players))), f'{likelihood} != {np.power(0.5, num_rounds * num_players)}'
    assert(np.isclose(deviance, -2 * np.log(np.power(0.5, num_episodes * num_rounds * num_players)))), f'{deviance} != {-2 * np.log(np.power(0.5, num_rounds * num_players))}'
    print('Fifth test passed!')

    # data = pd.read_csv('./data/random-0.csv')
    # agent_numbers = data.id_player.unique().tolist()
    # likeli = GetEpisodeLikelihood(agent_class=Random, model_name="Random", data=data)
    # parameters = {id:{'go_probability':0} for id in agent_numbers}
    # likelihood = likeli.get_likelihood(parameters)
    # print('Likelihood:', likelihood, '--- Deviance from likelihood:', -2 * np.log(likelihood))    
    # deviance = likeli.get_deviance(parameters)
    # print('Deviance:', deviance, '--- Likelihood from deviance:', np.exp(deviance / -2))
    # assert(np.isclose(likelihood, np.exp(deviance / -2)))
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, 1))
    # assert(np.isclose(deviance, 0))
    # print('First test passed!')

    # print('\nTesting with p=1...')
    # data = pd.read_csv('./data/random-1.csv')
    # likeli = GetEpisodeLikelihood(agent_class=Random, model_name="Random", data=data)
    # parameters = {id:{'go_probability':1} for id in agent_numbers}
    # likelihood = likeli.get_likelihood(parameters)
    # print('Likelihood:', likelihood, '--- Deviance from likelihood:', -2 * np.log(likelihood))    
    # deviance = likeli.get_deviance(parameters)
    # print('Deviance:', deviance, '--- Likelihood from deviance:', np.exp(deviance / -2))
    # assert(np.isclose(likelihood, np.exp(deviance / -2)))
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, 1))
    # assert(np.isclose(deviance, 0))
    # print('Second test passed!')

    # print('\nTesting with p=0.5...')
    # data = pd.read_csv('./data/random-05.csv')
    # num_rounds = max(data['round'].unique())
    # num_players = len(data['id_player'].unique())
    # likeli = GetEpisodeLikelihood(agent_class=Random, model_name="Random", data=data)
    # parameters = {id:{'go_probability':0.5} for id in agent_numbers}
    # likelihood = likeli.get_likelihood(parameters)
    # print('Likelihood:', likelihood, '--- Deviance from likelihood:', -2 * np.log(likelihood))    
    # deviance = likeli.get_deviance(parameters)
    # print('Deviance:', deviance, '--- Likelihood from deviance:', np.exp(deviance / -2))
    # assert(np.isclose(likelihood, np.exp(deviance / -2)))
    # assert(np.isclose(deviance, -2 * np.log(likelihood)))
    # assert(np.isclose(likelihood, np.power(0.5, num_rounds * num_players))), f'{likelihood} != {np.power(0.5, num_rounds * num_players)}'
    # assert(np.isclose(deviance, -2 * np.log(np.power(0.5, num_rounds * num_players)))), f'{deviance} != {-2 * np.log(np.power(0.5, num_rounds * num_players))}'
    # print('Third test passed!')