'''
Classes with agents' rules
'''
from random import randint, uniform, choice
from copy import deepcopy
import numpy as np
from itertools import product, combinations
from Classes.networks import MLP, ExperienceDataset
from torch.utils.data import DataLoader

DICT_STATES = {
    (0,0):0,
    (0,1):1,
    (1,0):2,
    (1,1):3
}


class Agent :
    '''
    Defines the basic methods for each agent.
    '''

    def __init__(self, parameters:dict={}, n:int=1):
        self.parameters = parameters
        self.decisions = []
        self.scores = []
        self.number = n

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # To be defined by subclass
        pass

    def update(self, score:int, obs_state_:tuple):
        '''
        Agent updates its model.
        Input:
            - score, a number 0 or 1.
            - obs_state_, a tuple with the sate of current round,
                         where each argument is 0 or 1.
        '''
        # To be defined by subclass
        pass

    def go_probability(self):
        '''
        Agent returns the probability of going to the bar
        according to its model.

        Output:
            - p, float representing the probability that the
                 agent goes to the bar.
        '''
        # To be defined by subclass
        pass

    def reset(self):
        '''
        Restarts the agent's data for a new trial.
        '''
        self.decisions = []
        self.scores = []

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


class AgentFP(Agent) :
    '''
    Implements an agent using the simple empirical distribution of attendances as per Fictitious Play.
     It uses the following parameters:
        * alphas
        * num_agents
        * threshold
        * belief_strength
    It also requires an id_number n
   '''
    def __init__(self, parameters:dict, n:int):
        super().__init__(parameters, n)
        assert(parameters["alphas"] is not None)
        self.alphas = deepcopy(parameters["alphas"])
        assert(parameters["num_agents"] is not None)
        self.num_agents = parameters["num_agents"]
        assert(parameters["threshold"] is not None)
        self.threshold = parameters["threshold"]
        assert(parameters["belief_strength"] is not None)
        self.belief_strength = parameters["belief_strength"]
        assert(parameters["epsilon"] is not None)
        self.epsilon = parameters["epsilon"]
        self.frequencies = [alpha for alpha in self.alphas]
        self.t = 0

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # Explore with probability epsilon.
        if uniform(0, 1) < self.epsilon:
            return randint(0, 1)
        else:
            # No exploration. Return greedy action.
            return self.greedy_action()
        
    def update(self, score:int, obs_state:tuple):
        '''
        Updates the observed frequencies of attendance of their partners.
        '''
        # Update records
        self.scores.append(score)
        self.decisions.append(obs_state[self.number])
        self.t += 1
        # Iterate over agents to update frequency
        t = self.t
        for i in range(self.num_agents):
            belief_strength = self.belief_strength
            prev_frequency = self.frequencies[i]
            a = obs_state[i] # Agent i's decision
            # Calculate online update of frequencies
            new_frequency = ((t - 1 + belief_strength) * prev_frequency +  a) / (t + belief_strength)
            self.frequencies[i] = new_frequency

    def greedy_action(self) -> int:
        '''
        Returns the action with higher expected utility.
        Break ties uniformly.
        Input:
            - prev_state, a tuple with the state of the previous round, 
                          where each argument is 0 or 1.
        Output:
            - a decision 0 or 1
        '''
        eus = [self.exp_util(action) for action in [0,1]]
        max_eu = max(eus)
        max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
        return choice(max_actions)
        
    def exp_util(self, action):
        '''
        Evaluates the expected utility of an action.
        Input:
            - action, which is a possible decision 0 or 1.
        Output:
            - The expected utility of action (float).
        '''
        if action == 0:
            # expected utility of no go
            return 0
        capacity = int(self.threshold * self.num_agents)
        # Calculate probabilities
        prob_ok = self.get_prob_upto_n(capacity - 1)
        prob_not_ok = self.get_prob_over_n(capacity - 1)
        # Calculate rewards
        utility_go_ok = 1
        utility_go_not_ok = -1
        # Calculate expected utility of go
        return utility_go_ok * prob_ok + utility_go_not_ok * prob_not_ok

    def get_prob_An(self, n:int) -> float:
        '''
        Determines the probability that exactly n agents go to the bar.
        Such agents do not include self.
        Input:
            - n, the number of attending agents
        Output:
            - prob_n, the probability of n attendants
        '''
        assert(0 <= n <= self.num_agents - 1)
        agent_list = [i for i in range(self.num_agents) if i != self.number]
        agent_go_tuples = list(combinations(agent_list, n))
        prob_n = 0
        for c in agent_go_tuples:
            probability_go = np.prod([self.frequencies[i] for i in c])
            probability_no_go = np.prod([1 - self.frequencies[i] for i in range(self.num_agents) if i not in c])
            prob_n += (probability_go * probability_no_go)
        return prob_n

    def get_prob_upto_n(self, n:int) -> float:
        '''
        Returns the probability that the attendance is less than or equal to n.
        Such agents do not include self.
        Input:
            - n, the number of attending agents
        Output:
            - prob, the probability of there are up to n attendants
        '''
        return sum([self.get_prob_An(i) for i in range(n+1)])

    def get_prob_over_n(self, n:int) -> float:
        '''
        Returns the probability that the attendance is less than or equal to n.
        Input:
            - n, the number of attending agents
        Output:
            - prob, the probability of there are up to n attendants
        '''
        return sum([self.get_prob_An(i) for i in range(n+1, self.num_agents)])


class AgentMFP(Agent) :
    '''
    Implements an agent using the Markov Fictitious Play learning rule.
    It uses the following parameters:
        * rate
        * alphas
    It also requires an id_number n
    '''

    def __init__(self, parameters:dict, n:int) :
        super().__init__(parameters)
        assert(parameters["alphas"] is not None)
        self.alphas = deepcopy(parameters["alphas"])
        self.prev_state_ = None
        self.states = [(a,b) for a in range(2) for b in range(2)]
        self.count_states = {state:0 for state in self.states}
        self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
        self.trans_probs = deepcopy(parameters["alphas"])
        assert(parameters["belief_strength"] is not None)
        self.belief_strength = parameters["belief_strength"]
        self.payoff = np.matrix([[0, 0], [1, -1]])
        self.number = n

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            return self.greedy_action(self.prev_state_)
        else:
            # no previous data, so make random decision
            return randint(0, 1)

    def update(self, score:int, obs_state:tuple):
        '''
        Agent updates its model using the Markov Fictitious Play rule.
        Input:
            - score, a number 0 or 1.
            - obs_state, a tuple with the sate of current round,
                         where each argument is 0 or 1.
        Input:
        '''
        # Update records
        self.scores.append(score)
        self.decisions.append(obs_state[self.number])
        # Agent recalls previous state?
        if self.prev_state_ is not None:
            prev_state = self.prev_state_
            # Update transtion counts
            observed_transition = (prev_state, obs_state)
            self.count_transitions[observed_transition] += 1
            # Loop over states and update transition probabilities
            for new_state in self.states:
                transition = (prev_state, new_state)
                numerator = self.count_transitions[transition] + self.belief_strength * self.alphas[transition]
                denominator = self.count_states[prev_state] + self.belief_strength
                new_prob = numerator / denominator
                assert(new_prob <= 1), f'\nTransition:{transition}\nTransition counts:{self.count_transitions[transition]}\nState counts:{self.count_states[prev_state]}'
                self.trans_probs[transition] = new_prob
        # Update state counts
        self.count_states[obs_state] += 1
        # Update previous state
        self.prev_state_ = obs_state

    def reset(self) :
        '''
        Restarts the agent's data for a new trial.
        '''
        super().reset()
        self.prev_state_ = None
        self.count_states = {state:0 for state in self.states}
        self.count_transitions = {(prev_s,new_s):0 for prev_s in self.states for new_s in self.states}
        self.trans_probs = deepcopy(self.alphas)

    def greedy_action(self, prev_state:tuple) -> int:
        '''
        Returns the action with higher expected utility.
        Break ties uniformly.
        Input:
            - prev_state, a tuple with the state of the previous round, 
                          where each argument is 0 or 1.
        Output:
            - a decision 0 or 1
        '''
        eus = [self.exp_util(prev_state, action) for action in [0,1]]
        max_eu = max(eus)
        max_actions = [i for i in range(len(eus)) if eus[i] == max_eu]
        return choice(max_actions)

    def exp_util(self, prev_state:tuple, action:int) -> float:
        '''
        Evaluates the expected utility of an action.
        Input:
            - prev_state, a tuple with the state of the previous round, 
                          where each argument is 0 or 1.
            - action, which is a possible decision 0 or 1.
        Output:
            - The expected utility (float).
        '''
        eu = 0
        state = [np.nan] * 2
        state[self.number] = action
        for partner in [0,1]:
            state[1 - self.number] = partner
            v = self.payoff[action, partner]
            p = self.trans_probs[(prev_state, tuple(state))]
            eu += v*p
        return eu

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
        states = [(a,b) for a in [0,1] for b in [0,1]]
        probs = '        ' + ' '.join([str(s) for s in states])
        print(probs)
        for prev_state in states:
            probs += '\n' + str(prev_state)
            for state in states:
                dummy = str(round(self.trans_probs[(prev_state, state)],2))
                if len(dummy) < 4:
                    dummy += '0'*(4-len(dummy))
                probs += '   ' + dummy
        print(f"No.agent:{self.number}, Decision:{decision}, Score:{score}, Lambda:{self.belief_strength}\ntrans_probs:\n{probs}")

    def go_probability(self):
        '''
        Agent returns the probability of going to the bar
        according to its model.

        Output:
            - p, float representing the probability that the
                 agent goes to the bar.
        '''
        # If greedy action is go (a=1), then probability of going is 1.
        # If greedy action is no go (a=0), then probability of going is 0.
        return self.greedy_action()



class epsilon_greedy(Agent):
    def __init__(self, parameters: dict, n: int):
        super().__init__(parameters, n)
        self.epsilon = parameters["epsilon"]
        # If self.epsilon is None, then cooling down protocol applies.

    def make_decision(self) -> int:
        '''
        Agent decides whether to go to the bar or not.
        Output:
            - A decision 0 or 1
        '''
        # Agent recalls previous state?
        if self.prev_state_ is None:
            return randint(0, 1)
        else:
            if self.epsilon is None:
                # Get the round number
                round = len(self.decisions)
                # Check value of epsilong accoring to cooling down protocol.
                if round < 100:
                    epsilon = 0.2
                elif round < 200:
                    epsilon = 0.1
                elif round < 300:
                    epsilon = 0.05
                elif round < 400:
                    epsilon = 0.01
                elif round < 500:
                    epsilon = 0.005
                else:
                    epsilon = 0
            else:
                # No cooling protocol. Use stored epsilon.
                epsilon = self.epsilon
            # Explore with probability epsilon.
            if uniform(0, 1) < epsilon:
                return randint(0, 1)
            else:
                # No exploration. Return greedy action.
                return self.greedy_action(self.prev_state_)

    def go_probability(self):
        '''
        Agent returns the probability of going to the bar
        according to its model.

        Output:
            - p, float representing the probability that the
                 agent goes to the bar.
        '''
        # Check if agent recalls previous round
        if self.prev_state_ is not None:
            # Obtain expected utility of go and no go
            eus = [self.exp_util(self.prev_state, action) for action in [0,1]]
            if eus[1] > eus[0]:
                # Return 1 - epsilon if go has higher expected utility
                return 1 - self.epsilon
            elif eus[1] < eus[0]:
                # Return epsilon if go has lower expected utility
                return self.epsilon
            else:
                # Return 0.5 for breaking ties randomly
                return 0.5
        else:
            # Agent does not recall previous round, so choice is random
            return 0.5
