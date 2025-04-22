DASH_LINE = '-'*60

from Classes.cognitive_model_agents import CogMod

def test_bar_is_full(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar is full')
    print(DASH_LINE)    
    action = 1
    other_player_actions = [1] * num_rounds
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def test_bar_has_capacity(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test bar has capacity')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    other_player_actions = [0] * num_rounds
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def test_alternation(agent:CogMod, num_rounds=10) -> None:
    print('')
    print(DASH_LINE)
    print('Test other player alternates')
    print(DASH_LINE) 
    agent.debug = True
    action = 0
    other_player_actions = [0, 1] * (num_rounds // 2)
    state = [action, other_player_actions[0]]
    print('Initial state:', state)
    agent.decisions.append(action)
    agent.prev_state_ = tuple(state)
    rollout(agent, other_player_actions, num_rounds)

def rollout(agent:CogMod, other_player_actions:int, num_rounds:int) -> None:
    state = agent.prev_state_
    for i in range(num_rounds):
        print(f'---------- Round {i} ----------')
        try:
            preferences = agent.determine_action_preferences()
        except Exception as e:
            print('')
            print(DASH_LINE)
            print("Error finding agent's preferences")
            print(e)
            print_agent(agent)
            raise Exception(e)
        print(f'Action preferences in state {state}: {preferences}')
        try:
            action = agent.make_decision()
        except Exception as e:
            print('')
            print(DASH_LINE)
            print("Error determining next action")
            print(e)
            print_agent(agent)
            raise Exception(e)
        print('Chosen action:', action)
        other_player_action = other_player_actions[i]
        new_state = [action, other_player_action]
        print('State arrived:', new_state)
        payoff = agent.payoff(action, new_state)
        print(f'Payoff action {action}: {payoff}')
        try:
            agent.update(payoff, new_state)
        except Exception as e:
            print('')
            print(DASH_LINE)
            print("Error updating agent")
            print(e)
            print_agent(agent)
            raise Exception(e)
        state = new_state

def print_agent(agent:CogMod) -> str:
    print(str(agent))
    print(f'{agent.free_parameters=}')
    if hasattr(agent, 'av_payoff'):
        print(f'{agent.av_payoff=}')
    if hasattr(agent, 'count_states'):
        print("Agent's count_states:")
        print(agent.count_states)
