'''
Classes for parameter recovery
'''

import pprint
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from bayes_opt import BayesianOptimization
from typing import (
	List, Dict, Tuple,
	Union, Optional, Callable
)

import warnings
warnings.filterwarnings("ignore")

from Classes.agents import Agent
from Utils.utils import PPT


data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'likelihoods')
data_folder.mkdir(parents=True, exist_ok=True)


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
                data: pd.DataFrame
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
        self.model = model
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        self.agents = None
        # Initialize log likelihoods
        self.log_likelihoods = {agent_number:0 for agent_number in self.agent_numbers}
        # For debugging
        self.debug = False

    def get_episode_log_likelihoods(self) -> None:
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
        # Create agents from model and parameters
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
        self.get_episode_log_likelihoods()
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
        # Checks if agents exist
        if self.agents is not None:
            for n, agent in self.agents.items():
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
                data: pd.DataFrame
            ) -> None:
        assert(hasattr(model, 'go_probability'))
        # Model bookkeeping
        self.model = model
        self.free_parameters = free_parameters
        self.agents = None
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

    def get_deviance_from_data(
                self, 
                free_parameters: Dict[str, any],
                save: Optional[bool]=False
            ) -> None:
        '''Get the parameters' likelihood given the data.'''
        list_deviances = list()
        list_fixed_parameters = self.get_params_list_from_data()
        for fixed_parameters in list_fixed_parameters:
            num_ag = int(fixed_parameters["num_agents"])
            threshold = fixed_parameters["threshold"]
            if self.debug:
                print(f'Finding deviance for {num_ag} players and threshold {threshold}...')
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
        return sum(list_deviances)

    def get_params_list_from_data(self) -> float:
        '''Returns a list with tuples, each containing
        fixed parameters and the corresponding group from the data'''
        list_fixed_parameters = list()
        params_list = PPT.get_fixed_parameters(self.data)
        # if self.model == MFP:
        #     for params in params_list:
        #             states = list(product([0,1], repeat=int(params["num_agents"])))
        #             count_states = ProxyDict(
        #                 keys=states,
        #                 initial_val=0
        #             )
        #             count_transitions = ProxyDict(
        #                 keys=list(product(states, repeat=2)),
        #                 initial_val=0
        #             )
        #             params["states"] = states
        #             params["count_states"] = count_states
        #             params["count_transitions"] = count_transitions
        #             params["designated_agent"] = False
        return params_list

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
        arguments = ', '.join([f'{key}=None' for key in self.free_parameters.keys()])
        function_definition = f'''
from types import MethodType
        
def def_funct(self, {arguments}):
    parameters = locals()
    # Return deviance
    deviance = self.get_deviance_from_data(parameters)
    return -deviance

self.black_box_function = MethodType(def_funct, self)
'''
        # print(function_definition)
        exec(function_definition)


class ParameterFit :
    '''
    Class for fitting the parameters of a model to a given data.

    Input:
        - agent_class, an agent class
        - model_name, the name of the model
        - fixed_parameters (list), a list with the name 
                of the fixed parameters used by the class
        - free_parameters (dict), a dictionary with the value 
                of the free parameters used by the class
        - data, a pandas dataframe with the columns id_sim, round, id_player, decision
        - optimizer, str with the name of the optimizer. Two options available:
            * bayesian
            * skitlearn
    '''

    def __init__(
                self, 
                agent_class: Agent, 
                model_name: str, 
				free_parameters: Dict[str, any],
                data: pd.DataFrame, 
                optimizer_name: Union[BayesianOptimization, None]
            ) -> None:
        # Bookkeeping
        self.data = data
        self.free_parameters = free_parameters
        self.agent_class = agent_class
        self.model_name = model_name
        # Initialize optimizer
        self.optimizer_name = optimizer_name
        if optimizer_name == 'bayesian':
            self.optimizer = self.create_bayesian_optimizer()
        else:
            raise Exception('Oooops')

    def get_optimal_parameters(
                self,
                hyperparameters:Dict[str, int],
            ) -> Tuple[float]:
        '''
        Returns the parameters that minimize dev(parameters)
        '''
        self.optimizer.maximize(**hyperparameters)
        return self.optimizer.max

    def create_bayesian_optimizer(self):
        # Initialize function to get deviance from model        
        pr = GetDeviance(
            model=self.agent_class,
            free_parameters=self.free_parameters,
            data=self.data
        )
        pr.create_deviance_function()
        # Bounded region of parameter space
        pbounds = dict()
        for parameter in self.free_parameters:
            extend_dict = self.get_saved_bounds(parameter)
            if extend_dict is not None:
                pbounds.update(extend_dict)
        print('pbounds:', pbounds)
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=pr.black_box_function,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True
        )
        return optimizer

    def get_saved_bounds(self, parameter:str) -> Tuple[float]:
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


# class GetEpisodeLikelihood :
#     '''
#     Class for obtaining the likelihood of a given model given some data.
#     It assumes that only one episode is contained in the data.
#     It detects the following from data:
#         * number of players
#         * bar's threshold

#     Input:
#         - agent_class, an agent class
#         - model_name, the name of the model
#         - data, a pandas dataframe with the columns id_sim, round, id_player, decision
#     '''

#     def __init__(self, agent_class, model_name:str, data:pd.DataFrame):
#         self.agent_class = agent_class
#         self.model_name = model_name
#         self.data = data
#         # Check that data contains only one episode
#         num_episodes = len(self.data['id_sim'].unique())
#         assert(num_episodes == 1), f'Invalid data: {num_episodes} episodes where found but only one (1) is accepted!'
#         # Obtain player ids from data
#         self.agent_numbers = data.id_player.unique().tolist()
#         # Obtain number of players:
#         self.num_agents = len(self.agent_numbers)
#         # Obtain bar's threshold
#         assert('threshold' in data.columns), f"Invalid data: It should contain a column with the bar's threshold!"
#         thresholds = data['threshold'].unique().tolist()
#         assert(len(thresholds) == 1), f"Invalid data: There should be just one threshold per episode, found {len(thresholds)}!" 
#         self.episode_threshold = thresholds[0]

#     def get_likelihood(self, parameters:dict) -> float:
#         '''
#         Obtains the likelihood of the parameters given the data.

#         Input:
#             - parameters (dict), a dictionary with the value of the parameters used by the class

#         Output:
#             - likelihood, float
#         '''
#         # Initialize agentes with given parameters
#         self.initialize_agents(parameters)
#         # Processes all rounds
#         num_rounds = max(self.data['round'].unique()) + 1
#         for _ in range(num_rounds):
#             self.process_round()
#         # -----------------------------------
#         # # Initializes variable
#         # likelihood = 1
#         # # Iterates over agents
#         # for id in self.agent_numbers:
#         #     print(id, self.likelihoods[id])
#         #     # Gets the likelihood per agent
#         #     for i in range(1, len(self.likelihoods[id])): # Do not count first round
#         #         p = self.likelihoods[id][i]
#         #         likelihood *= p
#         # -----------------------------------
#         #  Vectorized version of previous iteration
#         # -----------------------------------
#         likelihood = np.product([np.product(self.likelihoods[id][1:]) for id in self.agent_numbers])
#         return likelihood

#     def get_deviance(self, parameters:dict) -> float:
#         '''
#         Obtains the deviance of the parameters given the data.

#         Input:
#             - parameters (dict), a dictionary with the value of the parameters used by the class

#         Output:
#             - deviance, float
#         '''
#         # Initialize agentes with given parameters
#         self.initialize_agents(parameters)
#         # Processes all rounds
#         num_rounds = max(self.data['round'].unique()) + 1
#         for _ in range(num_rounds):
#             self.process_round()
#         # -----------------------------------
#         # # Initializes variable
#         # deviance = 0
#         # # Iterates over agents
#         # for id in self.agent_numbers:
#         #     # Gets the deviance per agent
#         #     for i in range(1, len(self.likelihoods[id])): # Do not count first round
#         #         p = self.likelihoods[id][i]
#         #         deviance += np.log(p)
#         # -----------------------------------
#         #  Vectorized version of previous iteration
#         # -----------------------------------
#         deviance = np.sum([np.sum(np.log(self.likelihoods[id][1:])) for id in self.agent_numbers])
#         return -2 * deviance

#     def initialize_agents(self, parameters:dict):
#         '''
#         Initialize the agents with the given parameters

#         Input:
#             - parameters (dict), a dictionary with the value of the parameters 
#                                  used to initialize each agent.
#         '''
#         self.parameters = parameters
#         # Create list of agents with given parameters
#         self.agents = [self.agent_class(id, self.parameters[id]) for id in self.agent_numbers]        
#         # Initialize round counter
#         self.round = 0
#         # Initialize per agent data likelihoods 
#         self.likelihoods = {id:[] for id in self.agent_numbers}

#     def process_round(self):
#         '''
#         Process one round of the game in order to obtain the round's likelihoods.
#         It performs the following steps:
#             * Obtains the attendances from the data on the round stored in cache.
#             * Gets the probability of each action for each agent. 
#             * Updates the likelihood of the action performed by each agent.
#             * Updates the agents with the data on the round stored in cache.
#         '''
#         # Get attendances
#         attendances = self.data.groupby('round').get_group(self.round)["decision"].tolist()
#         # Obtain the action probabilities for each agent
#         action_probabilities = self.get_action_probabilities()
#         # Update likelihoods
#         for i, id in enumerate(self.agent_numbers):
#             action = attendances[i]
#             probability = action_probabilities[i]
#             self.likelihoods[id].append(probability[action])
#         # Update agents
#         for agent in self.agents:
#             agent.update(score=None, obs_state_=tuple(attendances))
#         # Update round counter
#         self.round += 1

#     def get_action_probabilities(self):
#         '''
#         Obtains the probabilities of the actions for each agent
#         given the data on the round stored in cache.

#         Output:
#             - action_probabilities, list with a list with the probability of each
#                                     action for each agent
#         '''
#         # Initialize list that includes the action probabilities of all agents
#         action_probabilities = []
#         if self.round == 0:
#             action_probabilities = [[0.5, 0.5] for _ in range(len(self.agents))]
#         else:
#             for agent in self.agents:
#                 go_probability = agent.go_probability()
#                 action_probabilities.append([1 - go_probability, go_probability])
#         return action_probabilities



# class PPT :

#     @staticmethod
#     def get_group_column(columns: List[str]) -> str:
#         if 'id_sim' in columns:
#             return 'id_sim'
#         elif 'room' in columns:
#             return 'room'
#         elif 'group' in columns:
#             return 'group'
#         else:
#             raise Exception(f'Error: No column data found. Should be one of "id_sim", "room", or "group".\nColumns found: {columns}')

#     @staticmethod
#     def get_player_column(columns: List[str]) -> str:
#         if 'id_player' in columns:
#             return 'id_player'
#         elif 'player' in columns:
#             return 'player'
#         else:
#             raise Exception(f'Error: No player data found. Should be one of "id_player" or "player".\nColumns found: {columns}')

#     @staticmethod
#     def get_num_player_column(columns: List[str]) -> str:
#         if 'num_players' in columns:
#             return 'num_players'
#         elif 'num_agents' in columns:
#             return 'num_agents'
#         else:
#             raise Exception(f'Error: No number of players column found. Should be one of "num_players" or "num_agents".\nColumns found: {columns}')

#     @staticmethod
#     def get_decision_column(columns: List[str]) -> str:
#         if 'decision' in columns:
#             return 'decision'
#         elif 'choice' in columns:
#             return 'choice'
#         else:
#             raise Exception(f'Error: No decision data found. Should be one of "decision" or "choice".\nColumns found: {columns}')

#     @staticmethod
#     def get_fixed_parameters(data: pd.DataFrame) -> List[int]:
#         assert('threshold' in data.columns)
#         num_players_col = PPT.get_num_player_column(data.columns)
#         pairs = data[[num_players_col, 'threshold']].dropna().values.tolist()
#         pairs = [tuple(x) for x in pairs]
#         pairs = list(set(pairs))
#         list_fixed = list()
#         for num_p, threshold in pairs:
#             if num_p > 2:
#                 thres = threshold / num_p
#             else:
#                 thres = threshold
#             fixed_params = {
#                 'num_agents': int(num_p),
#                 'threshold': round(float(thres), 2)
#             }
#             list_fixed.append(fixed_params)
#         return list_fixed

