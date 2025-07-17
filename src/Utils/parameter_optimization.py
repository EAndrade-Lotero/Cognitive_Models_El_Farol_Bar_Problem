import pandas as pd
from pathlib import Path
from bayes_opt import BayesianOptimization
from typing import (
    List, Dict, Tuple,
    Union, Optional, Callable
)

from Classes.bar import Bar
from Classes.cognitive_model_agents import *
from Utils.interaction import Episode
from Utils.utils import PPT


class GetEfficiency:
    '''
    Class to obtain a measure from a model given some 
    free parameters.

    Input:
        - agent_class, an agent class
        - fixed_parameters, a dictionary with the model's fixed parameters
    '''
    def __init__(
                self, 
                agent_class: CogMod, 
                fixed_parameters: Dict[str, any],
                simulation_parameters: Dict[str, any],
                T: Optional[int]=20
            ) -> None:
        self.agent_class = agent_class
        self.fixed_parameters = fixed_parameters
        self.num_rounds = simulation_parameters['num_rounds']
        self.num_episodes = simulation_parameters['num_episodes']
        self.T = T
        # For debugging
        self.debug = False

    def efficiency(self, **free_parameters) -> float:
        # Generate simulated data
        if self.debug:
            print('Parameters for simulation:')
            print(free_parameters)
        data = self.generate_simulated_data(free_parameters)
        # Get only last T rounds from data
        num_rounds = max(data["round"].unique())
        data = pd.DataFrame(data[data["round"] >= num_rounds - self.T])
        if self.debug:
            print('')
            print('-'*50)
            # Get average score per simulation
            for sim, grp in data.groupby('id_sim'):
                m = grp['score'].mean()
                print(f'Sim:{sim} => mean score:{m}')
        # Get average score
        return data.score.mean()

    def generate_simulated_data(
                self, 
                free_parameters: Dict[str, any]
            ) -> None:
        '''Generate a dataset with simulated data with the 
        given free parameters'''
        bar, agents = self.initialize(free_parameters)
        episode = Episode(
            environment=bar,
            agents=agents,
            model=self.agent_class.name(),
            num_rounds=self.num_rounds
        )
        df = episode.simulate(num_episodes=self.num_episodes)
        return df

    def initialize(
                self,
                free_parameters: Dict[str, any]
            ) -> Tuple[Bar, List[CogMod]]:
        # Create bar
        bar = Bar(
            num_agents=self.fixed_parameters['num_agents'],
            threshold=self.fixed_parameters['threshold']
        )

        # # Check if focal regions are used
        # if 'FRA' in self.agent_class.name():
        #     # Create set of focal regions
        #     sfr = SetFocalRegions(
        #         num_agents=self.fixed_parameters['num_agents'],
        #         threshold=self.fixed_parameters['threshold'],
        #         len_history=free_parameters['len_history'], 
        #         max_regions=free_parameters['max_regions']
        #     )
        #     sfr.generate_focal_regions()
        #     self.fixed_parameters['sfr'] = sfr

        # Create agents
        agents = [
            self.agent_class(
                free_parameters=free_parameters, 
                fixed_parameters=self.fixed_parameters, 
                n=n
            ) for n in range(self.fixed_parameters['num_agents'])
        ]
        return bar, agents



class ParameterOptimization :
    '''
    Class for finding the parameters that maximize 
    a given measure of a given agent.

    Input:
        - agent_class, an agent class
        - model_name, the name of the model
        - fixed_parameters (list), a list with the name 
                of the fixed parameters used by the class
        - free_parameters (dict), a dictionary with the value 
                of the free parameters used by the class
        - measure_class, a class that takes the model_class and returns
                a float representing the measure to evaluate the free parameters
        - optimizer, str with the name of the optimizer. Two options available:
            * bayesian
            * skitlearn
    '''

    def __init__(
                self, 
                agent_class: CogMod, 
                fixed_parameters: Dict[str, any],
                simulation_parameters: Dict[str, any],
                hyperparameters: Optional[Union[Dict[str,any],None]]=None
            ) -> None:
        # --------------------------
        # Checking bounds are given
        # --------------------------
        assert hasattr(agent_class, 'bounds')
        # --------------------------
        # Bookkeeping
        # --------------------------
        self.fixed_parameters = fixed_parameters
        self.simulation_parameters = simulation_parameters
        self.agent_class = agent_class
        self.verbose = False
        # --------------------------
        # Initialize optimizer
        # --------------------------
        if hyperparameters is None:
                self.hyperparameters = {
                    'init_points':4,
                    'n_iter':16
                }
        else:
            self.hyperparameters = hyperparameters
        self.create_bayesian_optimizer()

    def get_optimal_parameters(self) -> Tuple[float]:
        '''
        Returns the parameters that minimize dev(parameters)
        '''
        self.optimizer.maximize(**self.hyperparameters)
        return self.optimizer.max

    def create_bayesian_optimizer(self) -> None:
        # Initialize function to get deviance from model        
        pr = GetEfficiency(
            agent_class=self.agent_class,
            fixed_parameters=self.fixed_parameters,
            simulation_parameters=self.simulation_parameters
        )
        # Bounded region of parameter space
        pbounds = self.agent_class.bounds(self.fixed_parameters)
        if self.verbose:
            print('pbounds:', pbounds)
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=pr.efficiency,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True,
            verbose=False
        )
        self.optimizer = optimizer

