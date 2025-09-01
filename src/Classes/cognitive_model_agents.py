'''
Classes with cognitive model agents' rules
'''
import numpy as np
from math import comb
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from random import randint, uniform
from typing import Optional, Union, Dict, List, Tuple, Any

from Classes.agent_utils import ProxyDict, TransitionsFrequencyMatrix
from Classes.focal_regions import SetFocalRegions


class CogMod() :
    '''
    Basic class for cognitive agents
    '''

    def __init__(
                self, 
                free_parameters: Optional[Dict[str, Any]]={}, 
                fixed_parameters: Optional[Dict[str, Any]]={}, 
                n: Optional[int]=1,
                fix_overflow: Optional[bool]=True
            ) -> None:
        #----------------------
        # Initialize lists
        #----------------------
        self.decisions = []
        self.scores = []
        self.prev_state_ = None
        self.debug = False
        self.number = n
        #----------------------
        # Parameter bookkeeping
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)
        #----------------------
        # Dealing with softmax overflow
        #----------------------
        self.fix_overflow = fix_overflow

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            go_prob = self.go_probability()
            probabilities = [1 - go_prob, go_prob]
            decision = np.random.choice(
                a=[0,1],
                size=1,
                p=probabilities
            )[0]
            if isinstance(decision, np.int64):
                decision = int(decision)
            return decision
        else:
            # no previous data, so make random decision
            return randint(0, 1)
        
    def go_probability(self) -> float:
        '''
        Agent returns the probability of going to the bar
        according to its model.
        Output:
            - p, float representing the probability that the
                agent decides to go to the bar.
        '''
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            # determine action preferences given previous state
            preferences = self.determine_action_preferences()
            probabilities = self.softmax(preferences)
            if self.debug:
                print('Action probabilities:')
                print(f'no go:{probabilities[0]} ---- go:{probabilities[1]}')
            return probabilities[1]
        else:
            # no previous data
            return 0.5
        
    def payoff(
                self, 
                action: int, 
                state: List[int]
            ) -> int:
        '''
        Determines the payoff of an action given the bar's attendance.
        Input:
            - action, go = 1 or no_go = 0
            - state, list of decisions of all agents
        Output:
            - List with payoff for the action
        '''
        attendance = sum([x == 1 or x == '1' for x in state])
        if action == 0 or action == '0':
            return 0
        elif attendance <= self.threshold * self.num_agents:
            return 1
        else:
            return -1
        
    def softmax(
                self,
                preferences: List[float]
            ) -> List[float]:
        '''
        Determines the softmax of the vector of preferences.
        Input:
            - preferences, list of preferences for actions
        Output:
            - List with softmax preferences
        '''
        # Broadcast inverse temperature and exponential
        # print('\tPreferences:', preferences)
        # numerator = np.exp(self.inverse_temperature * np.array(preferences))
        numerator = np.exp(self.inverse_temperature * np.array(preferences) - np.max(preferences)) # <= subtracted max for numerical stability
        num_inf = [np.isinf(x) for x in numerator]
        if sum(num_inf) == 1:
            softmax_values = [1 if np.isinf(x) else 0 for x in numerator]
            return softmax_values
        elif sum(num_inf) > 1:
            if self.fix_overflow:
                if self.debug:
                    print(f'Overflow warning: {num_inf}')
                numerator = np.array([1 if np.isinf(x) else 0 for x in numerator])
            else:
                raise Exception(f'Error: softmax gives rise to numerical overflow (numerator: {numerator})!')
        # print('\tNumerator:', numerator)
        # find sum of values
        denominator = sum(numerator)
        assert(not np.isinf(denominator))
        if np.isclose(denominator, 0):
            if self.fix_overflow:
                if self.debug:
                    print('Underflow warning:')
                softmax_values = [1 / len(numerator) for _ in numerator]
                return softmax_values
            else:
                raise Exception(f'Error: softmax gives rise to numerical overflow (denominator: {denominator})!')
        # print('\tDenominator:', denominator)
        # Return softmax using broadcast
        softmax_values = numerator / denominator
        assert(np.all([not np.isnan(n) for n in softmax_values])), f'numerator:{numerator} --- denominator: {denominator} --- preferences:{preferences}'
        return softmax_values

    def determine_action_preferences(self) -> List[float]:
        # To be defined by subclass
        pass

    def update(self, score:int, obs_state:List[int]) -> None:
        '''
        Agent updates its model.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        '''
        # Update records
        self.scores.append(score)
        action = obs_state[self.number]
        try:
            action = int(action)
        except Exception as e:
            print(f'Error: action of type {type(action)}. Type int was expected. (previous actions: {self.decisions})')
            raise Exception(e)
        self.decisions.append(action)
        self.prev_state_ = tuple(obs_state)

    def reset(self) -> None:
        '''
        Restarts the agent's data for a new trial.
        '''
        self.decisions = []
        self.scores = []
        self.prev_state_ = None

    def restart(self) -> None:
        # To be defined by subclass
        pass

    def print_agent(self, ronda:int=None) -> str:
            '''
            Returns a string with the state of the agent on a given round.
            Input:
                - ronda, integer with the number of the round.
            Output:
                - string with a representation of the agent at given round.
            '''
            if ronda is None:
                try:
                    ronda = len(self.decisions) - 1
                except:
                    ronda = 0
            try:
                decision = self.decisions[ronda]
            except:
                decision = "nan"
            try:
                score = self.scores[ronda]
            except:
                score = "nan"
            print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}")

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        '''
        Ingests parameters from the model.
        Input:
            - fixed_parameters, dictionary with fixed parameters
            - free_parameters, dictionary with free parameters
        '''
        self.fixed_parameters = fixed_parameters
        self.free_parameters = free_parameters
        self.threshold = fixed_parameters["threshold"]
        self.num_agents = int(fixed_parameters["num_agents"])
        self.inverse_temperature = free_parameters["inverse_temperature"]

    def __str__(self) -> str:
            '''
            Returns a string with the state of the agent on a given round.
            Input:
                - ronda, integer with the number of the round.
            Output:
                - string with a representation of the agent at given round.
            '''
            try:
                ronda = len(self.decisions) - 1
            except:
                ronda = 0
            try:
                decision = self.decisions[ronda]
            except:
                decision = "nan"
            try:
                score = self.scores[ronda]
            except:
                score = "nan"
            return f"No.agent:{self.number}\nDecision:{decision}, Score:{score}"

    @staticmethod
    def name():
        return 'CogMod'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        return {
            'inverse_temperature': (1, 64),
        }


class Random(CogMod) :
    '''
    Implements a random rule of go/no go with probability given by go_prob.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.go_prob = free_parameters["go_prob"]

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        go_prob = self.go_probability()
        return 1 if uniform(0, 1) < go_prob else 0
    
    def go_probability(self):
        '''
        Agent returns the probability of going to the bar
        according to its model.
        Output:
            - p, float representing the probability that the
                agent goes to the bar.
        '''
        return self.go_prob

    def __str__(self) -> str:
            '''
            Returns a string with the state of the agent on a given round.
            Input:
                - ronda, integer with the number of the round.
            Output:
                - string with a representation of the agent at given round.
            '''
            try:
                ronda = len(self.decisions) - 1
            except:
                ronda = 0
            try:
                decision = self.decisions[ronda]
            except:
                decision = "nan"
            try:
                score = self.scores[ronda]
            except:
                score = "nan"
            return f"No.agent:{self.number}, go_prob:{self.go_prob}\nDecision:{decision}, Score:{score}"

    @staticmethod
    def name():
        return 'Random'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = CogMod.bounds(fixed_parameters)
        bounds.update({
            'go_prob': (0, 1)
        })
        return bounds
    
    @staticmethod
    def create_random_params(num_agents:int) -> Dict[str, float]:
        free_parameters = {
            "inverse_temperature": np.random.uniform(4, 32),
            'go_prob': np.random.uniform(0, 1)
        }
        return free_parameters


class PriorsM1(CogMod) :
    '''
    Implements a random rule of go/no go with probability given by go_prob.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #--------------------------------------------
        # Create states
        #--------------------------------------------
        self.states = [0]
        self.shape = (1,)
        self.number = n
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.go_prob = np.zeros(self.shape)
        try:
            my_free_parameters = self.get_my_free_parameters(free_parameters)
        except Exception as e:
            print(f'Error: something wrong with the free parameters.\n{free_parameters=}')				
            raise Exception(e)
        for state in self.states:
            try:
                self.go_prob[state] = my_free_parameters[f"go_prob_{state}"]
            except Exception as e:
                print(f'Error with state {state}')
                print(f'{self.shape=}')
                print(f'{self.states=}')
                print(f'{self.go_prob.shape=}')
                print(f'{self.go_prob=}')
                print(f'Check also possible error with free parameters:\n{my_free_parameters=}')
                print(f'Here are also the fixed parameters:\n{fixed_parameters}')
                raise Exception(e)			

    def get_my_free_parameters(self, free_parameters:Dict[str, Any]) -> Dict[str, Any]:
        parameters = dict()
        for parameter, value in free_parameters.items():
            if '-' in parameter:
                number, state = parameter.split('-')
                if int(number) == self.number:
                    parameters[state] = value
            else:
                parameters[parameter] = value
        return parameters

    def go_probability(self) -> np.ndarray:
        '''
        Agent returns the probability of going to the bar
        according to its model.
        Output:
            - p, float representing the probability that the
                agent goes to the bar.
        '''
        if self.prev_state_ is None:
            return 0.5
        state = self._get_information_state(self.prev_state_)
        return self.go_prob[state]

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        return 0

    def determine_action_preferences(self) -> List[float]:
        # To be defined by subclass
        if self.prev_state_ is not None:
            go_prob = self.go_probability()
            probabilities = [1 - go_prob, go_prob]
            return probabilities
        else:
            return [0, 0]

    @staticmethod
    def name():
        return 'Priors-M1'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = CogMod.bounds(fixed_parameters)
        num_agents = fixed_parameters["num_agents"]
        for agent_n in range(num_agents):
            bounds.update({
                f"{agent_n}-go_prob_0": (0, 1)
            })
        return bounds

    @staticmethod
    def create_random_params(num_agents:int) -> Dict[str, float]:
        free_parameters = {"inverse_temperature": np.random.uniform(4, 32)}
        for agent_n in range(num_agents):
            free_parameters[f"{agent_n}-go_prob_0"] = np.random.uniform(0, 1)
        return free_parameters


class PriorsM2(PriorsM1) :
    '''
    Implements a random rule of go/no go with probability given by go_prob.
    This models conditions the probability on the previous action and aggregate state.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        num_agents = fixed_parameters['num_agents']
        free_parameters_ = PriorsM1.create_random_params(num_agents)
        super().__init__(
            free_parameters=free_parameters_, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create states
        #--------------------------------------------		
        self.states = list(product([0, 1], np.arange(num_agents + 1)))
        self.shape = (2, num_agents + 1)
        #--------------------------------------------
        # Bookkeeping for model parameters
        #--------------------------------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        if isinstance(state, dict):
            state_ = list(state.values())
        else:
            state_ = state
        action = state_[self.number]
        attendance_others = np.sum(state_) - action
        # Return index
        return action, attendance_others
    
    @staticmethod
    def name():
        return 'Priors-M2'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        num_agents = fixed_parameters["num_agents"]
        bounds = CogMod.bounds(fixed_parameters)
        states = list(product([0, 1], np.arange(num_agents + 1)))
        for agent_n in range(num_agents):
            bounds.update({f"{agent_n}-go_prob_{state}":(0, 1) for state in states})
        return bounds	

    @staticmethod
    def create_random_params(num_agents:int) -> Dict[str, float]:
        states = list(product([0, 1], np.arange(num_agents + 1)))
        free_parameters = {"inverse_temperature": np.random.uniform(4, 32)}
        for agent_n in range(num_agents):
            for state in states:
                free_parameters[f"{agent_n}-go_prob_{state}"] = np.random.uniform(0, 1)
        return free_parameters


class PriorsM3(PriorsM1) :
    '''
    Implements a random rule of go/no go with probability given by go_prob.
    This models conditions the probability on the previous actions vector, the full-state.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        num_agents = fixed_parameters['num_agents']
        free_parameters_ = PriorsM1.create_random_params(num_agents)
        super().__init__(
            free_parameters=free_parameters_, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create states
        #--------------------------------------------		
        self.shape = (2 ** self.num_agents, )
        self.states = np.arange(self.shape[0])
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        if isinstance(state, dict):
            state_ = list(state.values())
        else:
            state_ = state
        # Convert the state into a binary sequence
        binary_sequence = "".join(str(x) for x in state_)
        # Interpret the sequence as a decimal integer
        index =  int(binary_sequence, 2)
        # Return index
        return index
    
    @staticmethod
    def name():
        return 'Priors-M3'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        num_agents = fixed_parameters["num_agents"]
        states = np.arange(2 ** num_agents)
        bounds = CogMod.bounds(fixed_parameters)
        for agent_n in range(num_agents):
            for state in states:
                bounds.update({f"{agent_n}-go_prob_{state}":(0, 1) for state in states})
        return bounds

    @staticmethod
    def create_random_params(num_agents:int) -> Dict[str, float]:
        states = np.arange(2 ** num_agents)
        free_parameters = {"inverse_temperature": np.random.uniform(4, 32)}
        for agent_n in range(num_agents):
            for state in states:
                free_parameters[f"{agent_n}-go_prob_{state}"] = np.random.uniform(0, 1)
        return free_parameters


class WSLSM1(CogMod) :
    '''
    Defines the model of Win-Stay, Lose-Shift.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Initializing memory
        #----------------------
        self.states = [0]
        self.backup_av_payoff = [0]
        self.av_payoff = deepcopy(self.backup_av_payoff)
        self.restart()
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)		
        self.wsls_strength = free_parameters["wsls_strength"]
        self.heuristic_strength = free_parameters["heuristic_strength"]

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is None:
            return [0, 0]
        # Get previous action
        previous_state = self._get_information_state(self.prev_state_)
        # Use model to determine preferences
        average_payoff = self.av_payoff[previous_state]
        if average_payoff == 0:
            go_preference = 0
            no_go_preference = 0
        elif average_payoff > 0:
            go_preference = self.wsls_strength
            no_go_preference = 0
        else:
            no_go_preference = self.wsls_strength
            go_preference = 0
        if self.debug:
            print(f'Previous state: {previous_state}')
            print(f'Average payoff in previous state: {average_payoff}')
            print(f'go_preference = {go_preference}')
        # Return preferences
        preferences = [no_go_preference, go_preference]
        return preferences

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        return 0

    def reset(self):
        super().reset()
        self.av_payoff = deepcopy(self.backup_av_payoff)

    def restart(self) -> None:
        '''
        Restarts the agent memory.
        '''
        self.reset()
        self.count_states = ProxyDict(keys=self.states, initial_val=0)

    def update(self, score:int, obs_state:Tuple[int]):
        '''
        Agent updates its model.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        Input:
        '''
        # Update records
        self.scores.append(score)
        self.decisions.append(obs_state[self.number])
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            prev_state = self._get_information_state(self.prev_state_)
            # Increment count of states
            self.count_states.increment(prev_state)
            # Get action
            action = obs_state[self.number]
            # Update payoff average in an online fashion
            G = self.payoff(action, obs_state)
            self.av_payoff[prev_state] = self.heuristic_strength * G + (1 - self.heuristic_strength) * self.av_payoff[prev_state]
            # self.av_payoff[prev_state] += (1 / self.count_states(prev_state)) * (G - self.av_payoff[prev_state])
        if self.debug:
            print(f'I see the previous state: {prev_state}')
            print('I recall the following average payoff:')
            print(self.av_payoff[prev_state])
        # Update previous state
        self.prev_state_ = obs_state

    @staticmethod
    def name():
        return 'WSLS-M1'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = CogMod.bounds(fixed_parameters)
        bounds.update({
            'wsls_strength': (0, 10),
            'heuristic_strength': (0, 1)
        })
        return bounds


class WSLSM2(WSLSM1):
    '''
    Defines the model of Win-Stay, Lose-Shift.
    This model conditions G on the previous action and aggregate state.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create states
        #--------------------------------------------		
        self.states = list(product([0, 1], np.arange(self.num_agents + 1)))
        self.shape = (2, self.num_agents + 1)
        #----------------------
        # Initializing memory
        #----------------------
        self.backup_av_payoff = np.zeros(self.shape)
        self.av_payoff = deepcopy(self.backup_av_payoff)
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)
        self.restart()

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        if isinstance(state, dict):
            state_ = list(state.values())
        else:
            state_ = state
        action = state_[self.number]
        attendance_others = np.sum(state_) - action
        # Return index
        return action, attendance_others
    
    @staticmethod
    def name():
        return 'WSLS-M2'


class WSLSM3(WSLSM1):
    '''
    Defines the model of Win-Stay, Lose-Shift.
    This model conditions G on the previous action and aggregate state.
    '''
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create states
        #--------------------------------------------		
        self.states = list(product([0, 1], repeat=self.num_agents))
        self.shape = (2 ** self.num_agents, )
        #----------------------
        # Initializing memory
        #----------------------
        self.backup_av_payoff = ProxyDict(keys=self.states, initial_val=0)
        self.av_payoff = deepcopy(self.backup_av_payoff)
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)
        self.restart()

    def _get_information_state(self, state: List[int]) -> int:
        '''
        Determines the state observed by the agent according to its memory level.
        '''
        if isinstance(state, list):
            state_ = tuple(state)
        else:
            state_ = state
        assert isinstance(state_, tuple), f'Error: state should be a tuple or list. Got {state} of type {type(state)}'
        return tuple(state_)

    @staticmethod
    def name():
        return 'WSLS-M3'


class PayoffM1(CogMod) :
    '''
    Defines the error-driven learning rule based on payoffs.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for go preference
        #----------------------
        self.backup_Q = np.zeros(2)
        self.Q = deepcopy(self.backup_Q)
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.learning_rate = free_parameters["learning_rate"]

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is None:
            return [0, 0]
        else:
            return self.Q

    def update(
                self, 
                score: int, 
                obs_state: Tuple[int]
            ) -> None:
        '''
        Agent updates its model.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        '''
        action = obs_state[self.number]
        if self.prev_state_ is not None:
            # Agent learns
            self.learn(obs_state)
        # Update records
        self.scores.append(score)
        self.decisions.append(action)
        self.prev_state_ = obs_state

    def learn(
                self,
                obs_state: Tuple[int],
            ) -> None:
        '''
        Agent updates their action preferences
        Input:
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        '''
        # Get action
        action = obs_state[self.number]
        # Observe G
        G = self._get_G(obs_state)
        # Determine error prediction
        delta = G - self.Q[action]
        # Update Q table
        if self.debug:
            print('Learning rule:')
            print(f'Q[{action}] <- {self.Q[action]} + {self.learning_rate} * ({G} - {self.Q[action]})')
        self.Q[action] += self.learning_rate * delta
        if self.debug:
            print(f'Q[{action}] = {self.Q[action]}')

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        G = self.payoff(action, obs_state)
        if self.debug:
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    def restart(self) -> None:
        super().reset()
        self.Q = deepcopy(self.backup_Q)

    @staticmethod
    def name():
        return 'Payoff-M1'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = CogMod.bounds(fixed_parameters)
        bounds.update({
            'learning_rate': (0, 1),
        })
        return bounds
    

class PayoffM2(PayoffM1) :
    '''
    Defines the error-driven learning rule based on payoffs.
    This model conditions G on the previous action and aggregate state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for go preference
        #----------------------
        self.backup_Q = np.zeros((2, self.num_agents, 2))
        self.Q = deepcopy(self.backup_Q)

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is None:
            return [0, 0]
        else:
            prev_action, attendance = self._get_index(self.prev_state_)
            return self.Q[prev_action, attendance, :]

    def learn(
                self,
                obs_state: Tuple[int],
            ) -> None:
        '''
        Agent updates their action preferences
        Input:
            - obs_state, list of decisions on previous round
        '''
        # Get previous state
        previous_state = self.prev_state_
        # Get action
        action = obs_state[self.number]
        # Observe G 
        G = self._get_G(obs_state)
        # Determine error prediction
        prev_action, attendance = self._get_index(previous_state)
        delta = G - self.Q[prev_action, attendance, action]
        # Update Q table
        if self.debug:
            print('Learning rule:')
            print(f'Q[({prev_action}, {attendance}), {action}] <- {self.Q[prev_action, attendance, action]} + {self.learning_rate} * ({G} - {self.Q[prev_action, attendance, action]})')
        self.Q[prev_action, attendance, action] += self.learning_rate * delta
        if self.debug:
            print(f'Q[({prev_action}, {attendance}), {action}] = {self.Q[prev_action, attendance, action]}')

    def _get_index(self, state: List[int]) -> int:
        '''
        Determines the index of a state in a Q table
        Input:
            - state, list of decisions
        Output:
            - tuple, integers corresponding to the action and attendance
        '''
        if isinstance(state, dict):
            state_ = list(state.values())
        else:
            state_ = state
        action = state_[self.number]
        attendance_others = np.sum(state_) - action
        # Return index
        return action, attendance_others
    
    @staticmethod
    def name():
        return 'Payoff-M2'


class PayoffM3(PayoffM1) :
    '''
    Defines the error-driven learning rule based on payoffs.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for go preference
        #----------------------
        self.backup_Q = np.zeros((2 ** self.num_agents, 2))
        self.Q = deepcopy(self.backup_Q)

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is None:
            return [0, 0]
        else:
            index_previous_state = self._get_index(self.prev_state_)
            return self.Q[index_previous_state, :]

    def learn(
                self,
                obs_state: Tuple[int],
            ) -> None:
        '''
        Agent updates their action preferences
        Input:
            - action, go = 1 or no_go = 0
            - previous_state, list of decisions on previous round
            - new_state, list of decisions obtained after decisions
        '''
        # Get previous state
        previous_state = self.prev_state_
        # Get action
        action = obs_state[self.number]
        # Observe G 
        G = self._get_G(obs_state)
        # Determine error prediction
        index_previous_state = self._get_index(previous_state)
        delta = G - self.Q[index_previous_state, action]
        # Update Q table
        if self.debug:
            print('Learning rule:')
            print(f'Q[{previous_state},{action}] <- {self.Q[index_previous_state, action]} + {self.learning_rate} * ({G} - {self.Q[index_previous_state, action]})')
        self.Q[index_previous_state, action] += self.learning_rate * delta
        if self.debug:
            print(f'Q[{previous_state},{action}] = {self.Q[index_previous_state, action]}')

    def _get_index(self, state: List[int]) -> int:
        '''
        Determines the index of a state in a Q table
        Input:
            - state, list of decisions
        Output:
            - index, integer with the index of the Q table corresponding to the state
        '''
        if isinstance(state, dict):
            state_ = list(state.values())
        else:
            state_ = state
        # Convert the state into a binary sequence
        binary_sequence = "".join(str(x) for x in state_)
        # Interpret the sequence as a decimal integer
        index =  int(binary_sequence, 2)
        # Return index
        return index
    
    @staticmethod
    def name():
        return 'Payoff-M3'


class AttendanceM1(PayoffM1) :
    '''
    Defines the error-driven learning rule based on 
    weighted combination of average go and payoff.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.bias = free_parameters['bias']

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_go = np.mean(self.decisions + [action])
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_go + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average go: {average_go}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Attendance-M1'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = PayoffM1.bounds(fixed_parameters)
        bounds.update({
            'bias': (0, 1)
        })
        return bounds


class AttendanceM2(PayoffM2) :
    '''
    Defines the error-driven learning rule based on 
    weighted combination of average go and payoff.
    This model conditions G on the previous action 
    and aggregate state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.bias = free_parameters['bias']

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_go = np.mean(self.decisions + [action])
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_go + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average go: {average_go}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Attendance-M2'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = PayoffM2.bounds(fixed_parameters)
        bounds.update({
            'bias': (0, 1)
        })
        return bounds
    

class AttendanceM3(PayoffM3) :
    '''
    Defines the error-driven learning rule based on weighted 
    combination of average go and payoff.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.bias = free_parameters['bias']

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_go = np.mean(self.decisions + [action])
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_go + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average go: {average_go}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Attendance-M3'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = PayoffM3.bounds(fixed_parameters)
        bounds.update({
            'bias': (0, 1)
        })
        return bounds


class AvailableSpaceM1(AttendanceM1) :
    '''
    Defines the error-driven learning rule based on 
    available space in the bar.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get attendance other players
        attendance_others = np.sum(obs_state) - action
        G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
        if self.debug:
            print(f'Attendance other players: {attendance_others}')
            if action == 0:
                print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
            elif action == 1:
                print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'AvailableSpace-M1'


class AvailableSpaceM2(AttendanceM2) :
    '''
    Defines the error-driven learning rule based on 
    available space in the bar.
    This model conditions G on the previous action 
    and aggregate state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get attendance other players
        attendance_others = np.sum(obs_state) - action
        G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
        if self.debug:
            print(f'Attendance other players: {attendance_others}')
            if action == 0:
                print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
            elif action == 1:
                print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'AvailableSpace-M2'


class AvailableSpaceM3(AttendanceM3) :
    '''
    Defines the error-driven learning rule based on 
    available space in the bar.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get attendance other players
        attendance_others = np.sum(obs_state) - action
        G = ((attendance_others + 0.5) - int(self.threshold * self.num_agents)) * (1 - 2 * action)
        if self.debug:
            print(f'Attendance other players: {attendance_others}')
            if action == 0:
                print(f'G = {attendance_others + 0.5} - {self.threshold} * {self.num_agents}')
            elif action == 1:
                print(f'G = {self.threshold} * {self.num_agents} - {attendance_others + 0.5}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'AvailableSpace-M3'


class FairnessM1(AttendanceM1) :
    '''
    Defines the error-driven learning rule based 
    on weighted combination of fair amount of go
    and payoff.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_fairness = np.mean(self.decisions + [action]) - self.threshold
        average_fairness = average_fairness * (1 - 2 * action)
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_fairness + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average fairness: {average_fairness}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Fairness-M1'


class FairnessM2(AttendanceM2) :
    '''
    Defines the error-driven learning rule based 
    on weighted combination of fair amount of go
    and payoff.
    This model conditions G on the previous action 
    and aggregate state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_fairness = np.mean(self.decisions + [action]) - self.threshold
        average_fairness = average_fairness * (1 - 2 * action)
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_fairness + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average fairness: {average_fairness}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Fairness-M2'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = PayoffM2.bounds(fixed_parameters)
        bounds.update({
            'bias': (0, 0.01)
        })
        return bounds
    

class FairnessM3(AttendanceM3) :
    '''
    Defines the error-driven learning rule based 
    on weighted combination of fair amount of go
    and payoff.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )

    def _get_G(self, obs_state: Tuple[int]) -> float:
        action = obs_state[self.number]
        # Get go frequency
        average_fairness = np.mean(self.decisions + [action]) - self.threshold
        average_fairness = average_fairness * (1 - 2 * action)
        # Get payoff
        payoff = self.payoff(action, obs_state)
        G = self.bias * average_fairness + (1 - self.bias) * payoff
        if self.debug:
            print(f'Average fairness: {average_fairness}')
            print(f'Payoff: {payoff}')
            print(f'G observed for action {action} in state {self.prev_state_} is: {G}')
        return G

    @staticmethod
    def name():
        return 'Fairness-M3'


class MFPM1(CogMod) :
    '''
    Implements an agent using the Markov Fictitious Play learning rule 
    for multiple players.
    This is the unconditioned model.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        #--------------------------------------------
        # Create counters
        #--------------------------------------------
        self.states = [0]
        self.restart()
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        self.belief_strength = free_parameters['belief_strength']
        assert(self.belief_strength > 0)

    def determine_action_preferences(self) -> List[float]:
        '''
        Agent determines their preferences to go to the bar or not.
        Input:
            - state, list of decisions of all agents
        Output:
            - List with no go preference followed by go preference
        '''
        if self.prev_state_ is not None:
            eus = [self.exp_util(action) for action in [0,1]]
        else:
            eus = [0, 0]
        if self.debug:
            print('Expected utilities:')
            print(f'no go:{eus[0]} ---- go:{eus[1]}')
        return eus
    
    def exp_util(self, action):
        '''
        Evaluates the expected utility of an action.
        Input:
            - prev_state, a tuple with the state of the previous round, 
                        where each argument is 0 or 1.
            - action, which is a possible decision 0 or 1.
        Output:
            - The expected utility (float).
        '''
        if action == 0:
            return 0
        else:
            prev_sate = self.get_prev_state()
            numerator = self.count_bar_with_capacity(prev_sate) + self.belief_strength
            denominator = self.count_states(prev_sate) + 2 * self.belief_strength # <= The '2' comes from two options to normalize over (capacity and no capacity)
            prob_capacity = numerator / denominator
            prob_crowded = 1 - prob_capacity
            eu = prob_capacity - prob_crowded
            if self.debug:
                print(f'{prob_capacity=} --- {prob_crowded=}')
        return eu

    def update(self, score:int, obs_state:Tuple[int]):
        '''
        Agent updates its model using the observed frequencies.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                        where each argument is 0 or 1.
        Input:
        '''
        # Update records
        self.scores.append(score)
        self.decisions.append(obs_state[self.number])
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            prev_state = self.get_prev_state()
            # Increment count of states
            self.count_states.increment(prev_state)
            # Find other player's attendance
            action = obs_state[self.number]
            other_players_attendance = sum(obs_state) - action
            if other_players_attendance < int(self.threshold * self.num_agents):
                # Increment count of bar with capacity given previous state
                self.count_bar_with_capacity.increment(prev_state)
        if self.debug:
            print(f'I see the previous state: {prev_state}')
            print('I recall the following frequencies of states:')
            print(self.count_states)
            print('I recall the following frequencies of bar with capacity:')
            print(self.count_bar_with_capacity)
        # Update previous state
        self.prev_state_ = obs_state

    def _get_error_message(
                self, 
                new_prob: float, 
                transition: Dict[any, any], 
                prev_state: Tuple[int]
            ) -> str:
        error_message = f'Error: Improper probability value {new_prob}.\n'
        error_message += f'Transition:{transition}\n'
        error_message += f'Transition counts:{self.count_transitions(transition)}\n'
        error_message += f'Prev. state counts:{self.count_states(tuple(prev_state))}'
        return error_message	

    def reset(self) :
        '''
        Restarts the agent's data for a new trial.
        '''
        super().reset()
        self.prev_state_ = None

    def restart(self) -> None:
        '''
        Restarts the agent memory.
        '''
        self.reset()
        self.count_states = ProxyDict(keys=self.states, initial_val=0)
        self.count_bar_with_capacity = ProxyDict(
            keys=self.states,
            initial_val=0
        )

    def print_agent(self, ronda:int=None) -> str:
        '''
        Returns a string with the state of the agent on a given round.
        Input:
            - ronda, integer with the number of the round.
        Output:
            - string with a representation of the agent at given round.
        '''
        if ronda is None:
            try:
                ronda = len(self.decisions) - 1
            except:
                ronda = 0
        try:
            decision = self.decisions[ronda]
        except:
            decision = "nan"
        try:
            score = self.scores[ronda]
        except:
            score = "nan"
        print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}")
        tp = TransitionsFrequencyMatrix(num_agents=self.num_agents)
        tp.from_proxydict(self.trans_probs)
        print(tp)

    def get_prev_state(self):
        return 0

    @staticmethod
    def name():
        return 'MFP-M1'

    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        bounds = CogMod.bounds(fixed_parameters)
        bounds.update({
            'belief_strength': (1, 100)
        })
        return bounds


class MFPM2(MFPM1) :
    '''
    Implements an agent using the Markov Fictitious Play learning rule 
    for multiple players.
    This model conditions G on the previous action and aggregate state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        num_agents = fixed_parameters['num_agents']
        self.states =  list(product([0, 1], np.arange(num_agents + 1)))
        self.restart()

    def get_prev_state(self):
        assert(self.prev_state_ is not None)
        action = self.prev_state_[self.number]
        others_attendance = sum(self.prev_state_) - action 
        prev_state = (action, others_attendance)
        return prev_state

    @staticmethod
    def name():
        return 'MFP-M2'
    

class MFPM3(MFPM1) :
    '''
    Implements an agent using the Markov Fictitious Play learning rule 
    for multiple players.
    This model conditions G on the previous actions vector, the full-state.
    '''

    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(
            free_parameters=free_parameters, 
            fixed_parameters=fixed_parameters, 
            n=n
        )
        self.states = list(product([0, 1], repeat=self.num_agents))
        self.restart()

    def get_prev_state(self):
        assert(len(self.prev_state_) == self.num_agents)
        if isinstance(self.prev_state_, list):
            prev_state = tuple(self.prev_state_)
        else:
            prev_state = self.prev_state_
        return prev_state

    @staticmethod
    def name():
        return 'MFP-M3'


class FocalRegionAgent(CogMod):
    '''
    Agent that uses focal regions to determine next action.
    '''
    def __init__(
                self,
                free_parameters: Optional[Dict[str, Any]] = {},
                fixed_parameters: Optional[Dict[str, Any]] = {},
                n: Optional[int] = 1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(free_parameters, fixed_parameters, n)
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        # Create set of focal regions
        sfr = SetFocalRegions(
            num_agents=self.fixed_parameters['num_agents'],
            threshold=self.fixed_parameters['threshold'],
            len_history=self.free_parameters['len_history'], 
        )
        self.len_history = free_parameters['len_history']
        if 'max_regions' in free_parameters.keys():
            self.max_regions = free_parameters['max_regions']
            sfr.max_regions = free_parameters['max_regions']
        if 'c' in free_parameters.keys():
            self.c = free_parameters['c']
            sfr.c = free_parameters['c']
        if 'steepness' in free_parameters.keys():
            self.steepness = free_parameters['steepness']
            sfr.steepness = free_parameters['steepness']
        sfr.generate_focal_regions()
        self.sfr = sfr

    def determine_action_preferences(self) -> List[float]:
        # Get preferences based on Jaccard distance
        preferences = self.sfr.get_action_preferences(self.number)
        if self.debug:
            print(f'Preferences: {preferences}')
        return preferences

    def update(self, score:int, obs_state:List[int]) -> None:
        self.sfr.add_history(obs_state)
        super().update(score, obs_state)

    @staticmethod
    def name():
        return 'FRA'
    
    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        return {
            'inverse_temperature': (1, 64),
            'len_history': (1, 4),
            'c': (0.5, 1)
            # 'max_regions': (1, 10),
        }


class FRAplus(AttendanceM2):
    def __init__(
                self, 
                free_parameters:Optional[Dict[str, Any]]={}, 
                fixed_parameters:Optional[Dict[str, Any]]={}, 
                n:Optional[int]=1
            ) -> None:
        #----------------------
        # Initialize superclass
        #----------------------
        super().__init__(free_parameters, fixed_parameters, n)
        #----------------------
        # Bookkeeping for model parameters
        #----------------------
        self.ingest_parameters(fixed_parameters, free_parameters)

    def ingest_parameters(
                self, 
                fixed_parameters:Dict[str, Any], 
                free_parameters:Dict[str, Any]
            ) -> None:
        super().ingest_parameters(fixed_parameters, free_parameters)
        # Create set of focal regions
        sfr = SetFocalRegions(
            num_agents=self.fixed_parameters['num_agents'],
            threshold=self.fixed_parameters['threshold'],
            len_history=self.free_parameters['len_history'], 
        )
        self.len_history = free_parameters['len_history']
        if 'max_regions' in free_parameters.keys():
            self.max_regions = free_parameters['max_regions']
            sfr.max_regions = free_parameters['max_regions']
        if 'c' in free_parameters.keys():
            self.c = free_parameters['c']
            sfr.c = free_parameters['c']
        if 'steepness' in free_parameters.keys():
            self.steepness = free_parameters['steepness']
            sfr.steepness = free_parameters['steepness']
        sfr.generate_focal_regions()
        self.sfr = sfr
        self.delta = free_parameters['delta']

    def determine_action_preferences(self) -> List[float]:
        Q_preferences = super().determine_action_preferences()
        # Get preferences based on Jaccard similarity
        FRA_preferences = self.sfr.get_action_preferences(self.number)
        # Add up preferences
        preferences = self.delta * FRA_preferences + (1 - self.delta) * Q_preferences
        if self.debug:
            print(f'Preferences: {preferences}')
        return preferences

    def update(self, score:int, obs_state:List[int]) -> None:
        self.sfr.add_history(obs_state)
        super().update(score, obs_state)

    @staticmethod
    def name():
        return 'FRA+Payoff+Attendance'
    
    @staticmethod
    def bounds(fixed_parameters: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
        return {
            'inverse_temperature': (1, 64),
            'bias': (0, 1),
            'learning_rate': (0, 1),
            'len_history': (1, 4),
            'c': (0.5, 1),
            # 'max_regions': (1, 10),
            'delta': (0, 0.2),
        }




MODELS = [
    # PriorsM1, PriorsM2, PriorsM3,
    Random,
    WSLSM1, WSLSM2, WSLSM3,
    PayoffM1, PayoffM2, PayoffM3,
    AttendanceM1, AttendanceM2, AttendanceM3,
    AvailableSpaceM1, AvailableSpaceM2, AvailableSpaceM3,
    FairnessM1, FairnessM2, FairnessM3,
    MFPM1, MFPM2, MFPM3, 
    FocalRegionAgent, FRAplus
]

M1_MODELS = [model for model in MODELS if model.name().split('-')[-1] == 'M1']
M2_MODELS = [model for model in MODELS if model.name().split('-')[-1] == 'M2']
M3_MODELS = [model for model in MODELS if model.name().split('-')[-1] == 'M3']

MFPS = [MFPM1, MFPM2, MFPM3]

PRIOR_MODELS = [PriorsM1, PriorsM2, PriorsM3]

