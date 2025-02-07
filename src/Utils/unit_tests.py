DASH_LINE = '-'*60

from Classes.cognitive_model_agents import CogMod

def test_bar_is_full(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    state = [action, 1]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    for i in range(num_rounds):
        print(f'---------- Round {i} ----------')
        preferences = agent.determine_action_preferences()
        print(f'Action preferences in state {state}: {preferences}')
        action = agent.make_decision()
        print('Chosen action:', action)
        new_state = [action, 1]
        print('State arrived:', new_state)
        payoff = agent.payoff(action, new_state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, new_state)
        state = new_state 

def test_bar_has_capacity(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    state = [action, 0]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    for i in range(num_rounds):
        print(f'---------- Round {i} ----------')
        preferences = agent.determine_action_preferences()
        print(f'Action preferences in state {state}: {preferences}')
        action = agent.make_decision()
        print('Chosen action:', action)
        new_state = [action, 0]
        print('State arrived:', new_state)
        payoff = agent.payoff(action, new_state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, new_state)
        state = new_state  

def test_alternation(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test other player alternates')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    other_player_action = 0
    state = [action, other_player_action]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    for i in range(num_rounds):
        print(f'---------- Round {i} ----------')
        preferences = agent.determine_action_preferences()
        print(f'Action preferences in state {state}: {preferences}')
        action = agent.make_decision()
        print('Chosen action:', action)
        other_player_action = 1 - other_player_action
        new_state = [action, other_player_action]
        print('State arrived:', new_state)
        payoff = agent.payoff(action, new_state)
        print(f'Payoff action {action}: {payoff}')
        agent.update(payoff, new_state)
        state = new_state