'''
Classes for parameter recovery
'''

import json
import pprint
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from types import MethodType
from typing import (
	List, Dict, Tuple,
	Union, Optional, Callable
)
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize, Bounds

import warnings
warnings.filterwarnings("ignore")

from Utils.utils import PPT
from Config.config import PATHS
from Classes.agents import Agent
from Classes.cognitive_model_agents import *


data_folder = PATHS['data_likelihoods']


class GetEpisodeLikelihood :
    '''
    Class for obtaining the likelihood of a given model given some data.
    It assumes that only one episode is contained in the data.
    It detects the following from data:
        * number of players
        * bar's threshold

    Input:
        - model, an agent class
        - parameters, a dictionary with the model parameters
        - data, a pandas dataframe with the columns 
            * id_sim
            * threshold
            * round
            * id_player 
            * decision
            * attendance
            * score
    '''

    def __init__(
                self, 
                model: Agent, 
                fixed_parameters: Dict[str, any],
                free_parameters: Dict[str, any],
                data: pd.DataFrame,
                agents: Optional[Union[Dict[int, Agent], None]]=None
            ) -> None:
        assert(hasattr(model, 'go_probability'))
        # Data bookkeeping
        self.data = data
        # Check that data contains only one episode
        group_column = PPT.get_group_column(self.data.columns)
        episodes = self.data[group_column].unique().tolist()
        num_episodes = len(episodes)
        assert(num_episodes == 1), f'Invalid data: {num_episodes} episodes where found but only one (1) is accepted!'
        self.id_sim = episodes[0]
        # Obtain player ids from data
        player_column = PPT.get_player_column(self.data.columns)
        self.agent_numbers = data[player_column].unique().tolist()
        # Obtain number of players:
        self.num_agents = len(self.agent_numbers)
        # Obtain bar's threshold
        assert('threshold' in data.columns), f"Invalid data: It should contain a column with the bar's threshold! (columns: {data.columns})"
        thresholds = data['threshold'].unique().tolist()
        assert(len(thresholds) == 1), f"Invalid data: There should be just one threshold per episode, found {len(thresholds)}!" 
        self.episode_threshold = thresholds[0]
        # Model bookkeeping
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        self.model = model
        self.agents = agents
        if agents is not None:
            msg = f'Incorrect agent types:\n\t{[type(agent) for agent in agents.values()]}\n'
            assert(np.all([isinstance(agent, model) for agent in agents.values()])), msg + f'Agents should be of type {model}'
        # Initialize log likelihoods
        self.log_likelihoods = {agent_number:0 for agent_number in self.agent_numbers}
        # For debugging
        self.debug = False

    def get_episode_likelihoods(self) -> None:
        '''
        Obtains the log likelihoods of the model given the data.

        Output:
            - likelihood, float
        '''
        episode_data = self.data
        # Check that data contains only one episode
        if self.debug:
            print('Check episode integrity...')
        group_column = PPT.get_group_column(self.data.columns)
        num_episodes = len(episode_data[group_column].unique())
        assert(num_episodes == 1), f'Invalid data: {num_episodes} episodes where found but only one (1) is accepted!'
        # Create or update agents from model and parameters
        self.initialize_agents()
        # Check integrity of rounds
        if self.debug:
            print('Check integrity of rounds...')
        rounds = set(self.data["round"])
        rounds = list(rounds)
        rounds.sort()
        T = max(rounds)
        # assert(set(rounds) == set(range(T + 1)) ), f"Rounds incomplete! Only found {rounds}\nCheck missing values!"
        # assert(0 in rounds)
        if self.debug:
            print('Iterating over rounds...')
        for round in rounds:
            if self.debug:
                print(f'\tCalculating likelihood in round {round}')
            round_data = pd.DataFrame(episode_data.groupby('round').get_group(round))
            self.add_round_likelihoods(round_data)

    def add_round_likelihoods(self, round_data:pd.DataFrame) -> None:
        '''Find the log_likelihoods for the round'''
        # Get round attendance
        obs_state = self.get_round_attendance(round_data)
        for index, row in round_data.iterrows():
            # Bookkeeping
            decision_column = PPT.get_decision_column(self.data.columns)
            decision = int(row[decision_column])
            assert(isinstance(decision, int) or isinstance(decision, np.int16)), f'Error: action of type {type(decision)}. Type int was expected.'
            round = row['round']
            score = row['score']
            agent_column = PPT.get_player_column(self.data.columns)
            agent_number = row[agent_column]
            # Get agent's go probability
            agent = self.agents[agent_number]
            assert(agent_number == self.agent_numbers[agent.number]), f'{agent_number} != {self.agent_numbers[agent.number]}'
            go_prob = agent.go_probability()
            assert(not np.isnan(go_prob)), f'Parameters:\n{agent.free_parameters}\nAction preferences: {agent.softmax(agent.determine_action_preferences(agent.prev_state_))}'
            # Obtain likelihood for agent's decision
            if round == 0:
                likelihood = 0.5
            else:
                likelihood = go_prob if decision == 1 else 1 - go_prob
            likelihood = np.clip(likelihood, 1e-3, 1 - 1e-3)
            assert(not np.isinf(np.log(likelihood))), f"Player's decision: {decision}\n{agent}"
            self.log_likelihoods[self.agent_numbers[agent.number]] += np.log(likelihood)
            # Update agent
            agent.update(score, obs_state)
            # Show progress
            if self.debug:
                print(f'\t\tCalculating likelihood for agent {agent_number}')
                action = 'go' if decision == 1 else 'no go'
                print(f"\t\tAction taken was {action} with likelihood {likelihood}")

    def get_deviance(self) -> float:
        '''
        Obtains the deviance of the model given the data.

        Output:
            - output_dict with
                * deviance, float
                * num_agents, int
                * threshold, float
        '''
        # Get log likelihood from data
        self.get_episode_likelihoods()
        # Calculate deviance
        deviance = -2 * np.sum([self.log_likelihoods[id] for id in self.agent_numbers])
        dec_col = PPT.get_decision_column(self.data.columns)
        assert(not np.isinf(deviance)), [self.log_likelihoods[id] for id in self.agent_numbers]
        assert(not np.isnan(deviance)), [self.log_likelihoods[id] for id in self.agent_numbers]
        # Create output dictionary
        output_dict = {
            'deviance':deviance,
            'num_agents':self.num_agents,
            'threshold':self.episode_threshold
        }
        return output_dict

    def initialize_agents(self):
        '''
        Initialize the agents with the given parameters.
        '''
        # # Check if focal regions are used
        # if 'FRA' in self.model.name():
        #     # Create set of focal regions
        #     sfr = SetFocalRegions(
        #         num_agents=self.fixed_parameters['num_agents'],
        #         threshold=self.fixed_parameters['threshold'],
        #         len_history=self.free_parameters['len_history'], 
        #         max_regions=self.free_parameters['max_regions']
        #     )
        #     sfr.generate_focal_regions()
        #     self.fixed_parameters['sfr'] = sfr
        # Checks if agents exist
        if self.agents is not None:
            for n, agent in self.agents.items():
                agent.ingest_parameters(
                    fixed_parameters=self.fixed_parameters,
                    free_parameters=self.free_parameters
                )
                agent.reset()
            if self.debug:
                print('Agents initialized for new episode!')
        else:
            self.agents = {
                n:self.model(
                    fixed_parameters=self.fixed_parameters, 
                    free_parameters=self.free_parameters,
                    n=i
                ) for i, n in enumerate(self.agent_numbers)
            }
            if self.debug:
                print('Fresh agents created!')     

    def save_likelihoods(self):
        list_agents = [key for key, value in self.log_likelihoods.items()]
        list_likelihoods = [value for key, value in self.log_likelihoods.items()]
        df = pd.DataFrame({
            'id_player': list_agents, 
            'log_likelihood': list_likelihoods
        })
        df['id_sim'] = self.id_sim
        file = Path.joinpath(data_folder, f'{str(self.model)}_likelihoods.csv')
        df.to_csv(file)
        print(f'Data saved to {file}')

    def get_round_attendance(self, round_data:pd.DataFrame) -> Dict[int, int]:
        decision_column = PPT.get_decision_column(self.data.columns)
        #----------------------------
        # list version
        #----------------------------
        decisions = np.zeros(self.num_agents, dtype=np.int16)
        for index, row in round_data.iterrows():
            agent_column = PPT.get_player_column(self.data.columns)
            agent_number = row[agent_column]
            index = self.agent_numbers.index(agent_number)
            decisions[index] = row[decision_column]
        return tuple(decisions)
        #----------------------------
        # dict version (deprecated)
        #----------------------------
        # dict_decisions = dict()
        # for index, row in round_data.iterrows():
        #     agent_column = PPT.get_player_column(self.data.columns)
        #     agent_number = row[agent_column]
            # dict_decisions[agent_number] = row[decision_column]
        # return dict_decisions


class GetDeviance:
    '''
    Class to obtain deviance for a model given some data from multiple trials.

    Input:
        - model, an agent class
        - fixed_parameters, a dictionary with the model's fixed parameters
        - free_parameters, a dictionary with the model's free parameters
        - data, a pandas dataframe with the columns 
            * id_sim
            * threshold
            * round
            * id_player 
            * decision
            * attendance
            * score
    '''
    def __init__(
                self, 
                model: Agent, 
                free_parameters: Dict[str, any],
                data: pd.DataFrame,
                with_treatment: Optional[bool]=False
            ) -> None:
        assert(hasattr(model, 'go_probability'))
        # Model bookkeeping
        self.model = model
        self.free_parameters = free_parameters
        self.agents = None
        self.with_treatment = with_treatment
        # Data bookkeeping
        self.data = data
        self.process_log = {
            'id':list(),
            'num_agents':list(),
            'threshold':list(),
            'deviance':list()
        }
        # For debugging
        self.debug = False

    def get_deviance_from_group(
                self, 
                free_parameters: Dict[str, any],
                save: Optional[bool]=False
            ) -> None:
        # Get episodes id
        group_column = PPT.get_group_column(self.data.columns)
        groups = self.data[group_column].unique().tolist()
        if self.debug:
            print('Iterating over groups...')
        # Bookkeeping
        deviances = list()
        for group in groups:
            if self.debug:
                print('\n' + '-'*50)
                print(f'Calculating likelihood in group {group}')
            group_data = self.data.groupby(group_column).get_group(group)
            # Get the list of fixed parameters
            list_fixed_parameters = PPT.get_fixed_parameters(group_data)
            # Iterate over fixed parameters
            initial = True
            for fixed_parameters in list_fixed_parameters:
                # If first episode for this group, initialize agents
                if initial:
                    player_column = PPT.get_player_column(self.data.columns)
                    agents = {
                        n:self.model(
                            fixed_parameters=fixed_parameters, 
                            free_parameters=free_parameters,
                            n=i
                        ) for i, n in enumerate(group_data[player_column].unique().tolist())
                    }
                    initial = False
                # Filter data for this group and fixed parameters
                num_players = int(fixed_parameters["num_agents"])
                threshold = fixed_parameters["threshold"]
                episode_data = group_data.groupby([PPT.get_num_player_column(self.data.columns), "threshold"]).get_group(tuple([num_players, threshold])).reset_index()
                deviance_calculator = GetEpisodeLikelihood(
                    model=self.model,
                    fixed_parameters=fixed_parameters,
                    free_parameters=free_parameters,
                    data=episode_data,
                    agents=agents
                )
                output_dict = deviance_calculator.get_deviance()
                self.process_log['id'].append(group)
                self.process_log['num_agents'].append(output_dict['num_agents'])
                self.process_log['threshold'].append(output_dict['threshold'])
                self.process_log['deviance'].append(output_dict['deviance'])
                deviances.append(output_dict['deviance'])
                if save:
                    # Save to file
                    if self.debug: print('Saving...')
                    deviance_calculator.save_likelihoods()
                    if self.debug: print('Done!')
        if self.debug:
            pprint.pp(self.process_log)
        assert(np.all([not np.isnan(x) for x in deviances])), deviances
        deviance = np.sum(deviances)
        return deviance

    def get_deviance_from_data(
                self, 
                free_parameters: Dict[str, any],
                save: Optional[bool]=False
            ) -> None:
        '''Get the parameters' likelihood given the data.'''
        if self.with_treatment:
            deviance = self.get_deviance_from_group(
                free_parameters=free_parameters,
                save=save
            )
        else:
            # Get the list of fixed parameters
            list_deviances = list()
            list_fixed_parameters = PPT.get_fixed_parameters(self.data)
            # Iterate over fixed parameters
            for fixed_parameters in list_fixed_parameters:
                num_ag = fixed_parameters["num_agents"]
                threshold = fixed_parameters["threshold"]
                if self.debug:
                    print(f'Finding deviance for {int(num_ag)} players and threshold {threshold}...')
                num_agent_column = PPT.get_num_player_column(self.data.columns)
                try:
                    df = self.data.groupby([num_agent_column, "threshold"]).get_group(tuple([num_ag, threshold])).reset_index()
                except Exception as e:
                    print(tuple([num_ag, threshold]))
                    for key, grp in self.data.groupby([num_agent_column, "threshold"]):
                        print('=>', key)
                    raise Exception(e)
                deviance = self.get_deviance_given_parameters(
                    free_parameters=free_parameters,
                    fixed_parameters=fixed_parameters, 
                    data=df
                )
                list_deviances.append(deviance)
                deviance = sum(list_deviances)
        return deviance

    def get_deviance_given_parameters(
                self, 
                free_parameters: Dict[str, any],
                fixed_parameters: Dict[str, any],
                data: pd.DataFrame,
                save: Optional[bool]=False
            ) -> float:
        '''Get the likelihood for each episode in turn.'''
        # Get episodes id
        group_column = PPT.get_group_column(self.data.columns)
        episodes = data[group_column].unique().tolist()
        if self.debug:
            print('Iterating over episodes...')
        # Bookkeeping
        deviances = list()
        for episode in episodes:
            self.process_log['id'].append(episode)
            if self.debug:
                print('\n' + '-'*50)
                print(f'Calculating likelihood in episode {episode}')
            episode_data = data.groupby(group_column).get_group(episode)
            deviance_calculator = GetEpisodeLikelihood(
                model=self.model,
                fixed_parameters=fixed_parameters,
                free_parameters=free_parameters,
                data=episode_data
            )
            output_dict = deviance_calculator.get_deviance()
            self.process_log['num_agents'].append(output_dict['num_agents'])
            self.process_log['threshold'].append(output_dict['threshold'])
            self.process_log['deviance'].append(output_dict['deviance'])
            deviances.append(output_dict['deviance'])
            if save:
                # Save to file
                if self.debug:
                    print('Saving...')
                deviance_calculator.save_likelihoods()
                if self.debug:
                    print('Done!')
        if self.debug:
            pprint.pp(self.process_log)
        assert(np.all([not np.isnan(x) for x in deviances])), deviances
        deviance = np.sum(deviances)
        return deviance

    def create_deviance_function(self) -> Callable:
        def deviance_function(self, **kwargs):
            # Set default values from free_parameters
            for key, default in self.free_parameters.items():
                kwargs.setdefault(key, default)
            
            # Compute and return negative deviance
            deviance = self.get_deviance_from_data(kwargs)
            return -deviance

        # Bind the function as a method to the instance
        self.black_box_function = MethodType(deviance_function, self)


class ParameterFit :
    '''
    Class for fitting the parameters of a model to a given data.

    Input:
        - agent_class, an agent class
        - model_name, the name of the model
        - data, a pandas dataframe with the columns id_sim, round, id_player, decision
        - optimizer, str with the name of the optimizer. Two options available:
            * bayesian
            * skitlearn
    '''

    def __init__(
                self, 
                agent_class: Agent, 
                model_name: str, 
                data: pd.DataFrame, 
                optimizer_type: Union[str, None],
                with_treatment: Optional[bool]=False
            ) -> None:
        # Bookkeeping
        self.data = data
        self.agent_class = agent_class
        self.model_name = model_name
        self.with_treatment = with_treatment
        self.optimizer_type = optimizer_type
        self.debug = False

    def get_optimal_parameters(
                self,
                hyperparameters:Dict[str, int],
            ) -> Tuple[float]:
        '''
        Returns the parameters that minimize dev(parameters)
        '''
        results = dict()
        list_fixed_parameters = PPT.get_fixed_parameters(self.data)
        # Iterate over fixed parameters
        for fixed_parameters in list_fixed_parameters:
            # Get data matching fixed parameters
            num_ag = int(fixed_parameters["num_agents"])
            threshold = fixed_parameters["threshold"]
            num_agent_column = PPT.get_num_player_column(self.data.columns)
            if self.debug:
                print(f'Finding deviance for {num_ag} players and threshold {threshold}...')
            try:
                df = self.data.groupby([num_agent_column, "threshold"]).get_group(tuple([num_ag, threshold])).reset_index()
            except Exception as e:
                print(tuple([num_ag, threshold]))
                for key, grp in self.data.groupby([num_agent_column, "threshold"]):
                    print('=>', key)
                raise Exception(e)
            
            # Create list of free parameters from model
            try:
                free_parameters, pbounds = self.get_pbounds(fixed_paramters=fixed_parameters)
            except Exception as e:
                print(f'Error getting bounds from {self.agent_class}:\n{e}')
                raise e

            # Create optimizer
            if self.optimizer_type == 'bayesian':
                optimizer = self.create_bayesian_optimizer(
                    data=df,
                    free_parameters=free_parameters, 
                    pbounds=pbounds
                )
                # Find optimal parameters
                optimizer.maximize(**hyperparameters)
            elif self.optimizer_type == 'scipy':
                # Find optimal parameters
                result = self.create_scipy_optimizer(
                    data=df,
                    free_parameters=free_parameters, 
                    pbounds=pbounds
                )
            else:
                raise NotImplementedError(f'Optimizer {self.optimizer_type} not implemented!')
            

            # Save results
            results['model'] = self.agent_class.__name__
            results['fixed_parameters'] = fixed_parameters

            if self.optimizer_type == 'bayesian':
                results['free_parameters'] = optimizer.max['params']
                results['deviance'] = optimizer.max['target']
            elif self.optimizer_type == 'scipy':
                results['free_parameters'] = {parameter:result.x[i] for i, parameter in enumerate(free_parameters.keys())}
                results['deviance'] = -result.fun
            k = len(results['free_parameters'])
            dev = results['deviance']
            results['AIC'] = 2*k - 2*dev

            if self.debug:
                print(f'Optimal parameters for {num_ag} players and threshold {threshold}:\n{results["free_parameters"]}')
                print(f'Deviance: {results["deviance"]}')
                print(f'AIC: {results["AIC"]}')
                print('-'*50)

        return results
    
    def create_bayesian_optimizer(
                self, 
                data: pd.DataFrame,
                free_parameters: Dict[str, any],
                pbounds: Dict[str, Tuple[float]],
            ) -> BayesianOptimization:
        '''
        Create a Bayesian optimizer for the model given the data.'''

        # Get the list of free parameters
        # Initialize function to get deviance from model        
        pr = GetDeviance(
            model=self.agent_class,
            data=data,
            free_parameters=free_parameters,
            with_treatment=self.with_treatment
        )
        pr.create_deviance_function()
        # Bounded region of parameter space
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=pr.black_box_function,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=False ###########################
        )
        return optimizer

    def create_scipy_optimizer(
                self, 
                data: pd.DataFrame,
                free_parameters: Dict[str, any],
                pbounds: Dict[str, Tuple[float]],
            ) -> Callable:
        '''
        Create a Scipy optimizer for the model given the data.
        '''
        def black_box_function(x) -> float:
            input_dict = {parameter:x[i] for i, parameter in enumerate(free_parameters.keys())}
            return -pr.black_box_function(**input_dict)
        
        # Get the list of free parameters
        # Initialize function to get deviance from model        
        pr = GetDeviance(
            model=self.agent_class,
            data=data,
            free_parameters=free_parameters,
            with_treatment=self.with_treatment
        )
        pr.create_deviance_function()
        # Define bounds
        lower_bounds = []
        upper_bounds = []
        for lims in pbounds.values():
            lower_bounds.append(lims[0])
            upper_bounds.append(lims[1])
        x0 = self.random_init(pbounds)
        # Define optimizer
        optimizer = minimize(
            black_box_function,
            x0=x0,
            bounds=Bounds(lower_bounds, upper_bounds),
            method='L-BFGS-B'
        )
        return optimizer

    def random_init(self, pbounds: Dict[str, any]) -> np.array:
        '''Returns a random initial point for the optimizer'''
        return np.array([
            np.random.uniform(lims[0], lims[1]) 
                for parameter, lims in pbounds.items()
        ])

    def get_pbounds(self, fixed_paramters: Dict[str, any]) -> Dict[str, Tuple[float]]:
        '''Returns the bounds of the parameters'''
        pbounds = self.agent_class.bounds(fixed_paramters)
        assert (pbounds is not None), f'No bounds for {self.agent_class.name()}'
        free_parameters = {parameter:np.nan for parameter in pbounds.keys()}
        return free_parameters, pbounds

    def get_saved_bounds(
                self, 
                parameter:str,
            ) -> Tuple[float]:
        '''Returns the known bounds for given parameter'''
        #------------------------------------------
        # Random
        #------------------------------------------
        if parameter == 'go_prob':
            return {'go_prob':(0, 1)}
        #------------------------------------------
        #Other cognitive models
        #------------------------------------------
        elif parameter == 'inverse_temperature':
            return {'inverse_temperature':(4, 32)}
        #------------------------------------------
        # WSLS
        #------------------------------------------
        if parameter == 'go_drive':
            return {'go_drive':(0, 1)}
        if parameter == 'wsls_strength':
            return {'wsls_strength':(0, 2)}
        #------------------------------------------
        # Error-driven
        #------------------------------------------
        if parameter == 'learning_rate':
            return {'learning_rate':(0, 1)}
        if parameter == 'bias':
            return {'bias':(0, 1)}
        #------------------------------------------
        # MFP
        #------------------------------------------
        if parameter == 'belief_strength':
            return {'belief_strength':(1, 100)}
        else:
            raise Exception(f'Parameter {parameter} not known!')
    
    @staticmethod
    def run(
                data: pd.DataFrame, 
                model_list: List[CogMod], 
                best_fit_path:Path,
                optimizer_type: Optional[Union[str, None]]='bayesian',
                hyperparameters: Optional[Dict[str, int]]=None,
                new_file: Optional[bool]=False
            ) -> None:
        # Create optimization hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'init_points':8,
                'n_iter':16
            }

        if new_file:
            open_method = 'w'
        else:
            open_method = 'a'

        with open(best_fit_path, open_method) as f:
            for model in tqdm(model_list, desc='Fitting models...'):
                print(f'Fitting data to model {model.name()}...')
                best_fit = {'model_name': model.name()}

                print('Creating parameter recovery class...')
                pf = ParameterFit(
                    agent_class=model,
                    model_name=model.__name__,
                    data=data,
                    optimizer_type=optimizer_type
                )

                print('Running optimizer...')
                res = pf.get_optimal_parameters(hyperparameters)
                best_fit.update(res)

                # Write one JSON object per line
                f.write(json.dumps(best_fit) + '\n')        