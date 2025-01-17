import numpy as np
from itertools import product

from Classes.agent_utils import ProxyDict
from Classes.cognitive_model_agents import (
    WSLS, 
    PayoffRescorlaWagner,
    AttendanceRescorlaWagner,
    Q_learning,
    MFP,
    MFPAgg
)


DASH_LINE = '-'*50


def test_wsls():
    fixed_parameters = {
        "threshold": 0.5,
        "num_agents": 2,
    }
    free_parameters = {
        "inverse_temperature": 2,
        "go_drive": 1,
        "wsls_strength": 1
    }
    agent = WSLS(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=1
    )
    action = 1
    other_player_actions = [1, 1, 0, 0]
    agent.decisions.append(action)
    print("Agent's drive to go:", agent.go_drive)
    print("Agent's wsls strength:", agent.wsls_strength)
    for other_player_action in other_player_actions:
        print(DASH_LINE)
        state = [action, other_player_action]
        agent.prev_state_ = state
        print('state k:', state)
        print('Payoff a_k:', agent.payoff(action, state))
        preferences = agent.determine_action_preferences(state)
        print('Action preferences:', preferences)
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)


def test_payoff_rescorla_wagner():
    fixed_parameters = {
        "threshold": 0.5,
        "num_agents": 2,
    }
    free_parameters = {
        "inverse_temperature": 5,
        "initial_reward_estimate_go": 0,
        "initial_reward_estimate_no_go": 0,
        "learning_rate": 0.1
    }
    agent = PayoffRescorlaWagner(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print('Action preferences:', preferences)
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        state = [action, 1]
        print('State arrived:', state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE)
    agent = PayoffRescorlaWagner(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    action = 1
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print('Action preferences:', preferences)
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        state = [action, 0]
        print('State arrived:', state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state


def test_attendance_rescorla_wagner():
    fixed_parameters = {
        "threshold": 0.5,
        "num_agents": 2,
    }
    free_parameters = {
        "inverse_temperature": 5,
        "initial_luft_estimate": 1,
        "learning_rate": 0.1
    }
    agent = AttendanceRescorlaWagner(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print('Action preferences:', preferences)
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        state = [action, 1]
        print('State arrived:', state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE)
    free_parameters = {
        "inverse_temperature": 5,
        "initial_luft_estimate": 0,
        "learning_rate": 0.1
    }
    agent = AttendanceRescorlaWagner(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    action = 1
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print('Action preferences:', preferences)
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        state = [action, 0]
        print('State arrived:', state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state


def test_q_learning():
    num_agents = 2
    fixed_parameters = {
        "threshold": 0.5,
        "num_agents": num_agents,
    }
    free_parameters = {
        "inverse_temperature": 5,
        "go_drive":0.6,
        "learning_rate": 0.1,
        "discount_factor": 1,
    }
    agent = Q_learning(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 1]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE)   
    agent = Q_learning(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    action = 1
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = state
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 0]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = state


def test_MFP():
    num_agents = 2
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
        "threshold": 0.5,
        "num_agents": num_agents,
        "states": states,
        "count_states": count_states,
        "count_transitions": count_transitions,
        "designated_agent": True
    }
    free_parameters = {
        "inverse_temperature": 5,
        "go_drive":1,
        "belief_strength": 1,
    }
    agent = MFP(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    agent.count_states.increment(tuple(state))
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 1]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = tuple(state)
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE)   
    agent = MFP(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    action = 1
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    agent.count_states.increment(tuple(state))
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 0]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = tuple(state)


def test_MFP_Agg():
    num_agents = 2
    fixed_parameters = {
        "threshold": 0.5,
        "num_agents": num_agents,
    }
    free_parameters = {
        "inverse_temperature": 5,
        "belief_strength": 1,
        "go_drive":1,
    }
    agent = MFPAgg(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    agent.count_states[tuple(state)] += 1
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 1]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = tuple(state)
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE)   
    agent = MFPAgg(
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        n=0
    )
    agent.debug = True
    action = 1
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    agent.count_states[tuple(state)] += 1
    for _ in range(10):
        preferences = agent.determine_action_preferences(state)
        print(f'Action preferences in state {state}: {preferences}')
        probabilities = agent.softmax(preferences)
        print('Action probabilities:', probabilities)
        action = agent.make_decision()
        print('Chosen action:', action)
        agent.decisions.append(action)
        new_state = [action, 0]
        state = new_state
        print('State arrived:', new_state)
        payoff = agent.payoff(action, state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, state)
        agent.prev_state_ = tuple(state)
