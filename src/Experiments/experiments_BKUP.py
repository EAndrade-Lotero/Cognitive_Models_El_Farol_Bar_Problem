from Classes.bar import Bar
from Classes.agents import Random, AgentMFP, AgentMFP_Multi, epsilon_greedy
from Utils.interaction import Episode, PlotsAndMeasures, Experiment
from random import randint
from itertools import product

def experiment():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 500
    num_episodes = 100
    verbose = False
    # Define agents
    epsilon = 0.01
    belief_strength = 1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFP',\
                   num_rounds=num_rounds)
    # Run simulation
    df = episode.simulate(num_episodes=num_episodes, verbose=verbose)
    # Save results to cvs
    df.to_csv(f'./data/mfp_{num_agents}agents_epsilon.csv', index=False)
    # Plot results
    p = PlotsAndMeasures(df)
    p.plot_scores(file=f'./images/scores_mfp_{num_agents}agents_epsilon.png')    
    p.plot_decisions(file=f'./images/decisions_mfp_{num_agents}agents_epsilon.png')    
    p.plot_hist_scores(mu=threshold, file=f'./images/dist_mfp_{num_agents}agents_epsilon.png')    
    p.plot_EQ(mu=threshold, file=f'./images/eq_mfp_{num_agents}agents_epsilon.png')



def visualize():
    # Define simulation parameters
    num_agents = 2
    threshold = .6
    num_rounds = 30
    num_episodes = 1
    verbose = True
    # Define agents
    epsilon = 0
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='MFP',\
                   num_rounds=num_rounds)
    # Run simulation
    episode.renderize()


def sweep1(parameter, values):
    # Define simulation parameters
    num_agents = 2
    threshold = .6
    num_rounds = 1000
    num_episodes = 100
    measures = ['score']
    verbose = True
    # Define agents
    epsilon = 0.01
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Experiment(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,\
                   num_episodes=num_episodes,\
                   measures=measures,\
                    parameters=parameters)
    # Run sweep
    experiment.run_sweep1(parameter=parameter, values=values)


def sweep2(parameter1, values1, parameter2, values2):
    # Define simulation parameters
    num_agents = 2
    threshold = .6
    num_rounds = 100
    num_episodes = 100
    measures = ['score']
    verbose = True
    # Define agents
    epsilon = 0
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Experiment(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,\
                   num_episodes=num_episodes,\
                   measures=measures)
    # Run sweep
    experiment.run_sweep2(parameter1=parameter1,\
                          values1=values1,\
                          parameter2=parameter2,\
                          values2=values2)


def simulate_mfp_multi_epsilons():
    # Define simulation parameters
    num_agents = 2
    threshold = .6
    num_rounds = 200
    num_trials = 50

    # Store average scores for each epsilon value
    avg_scores = []
    epsilon_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]

    for epsilon in epsilon_values:
        # Define agents
        belief_strength = 0.1
        states = list(product([0, 1], repeat=num_agents))
        alphas = {(x, y): 1 / len(states) for x in states for y in states}
        parameters = {"belief_strength": belief_strength,
                      "alphas": alphas,
                      "num_agents": num_agents,
                      "threshold": threshold,
                      "epsilon": epsilon}
        agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]

        # Create simulation
        bar = Bar(num_agents=num_agents, threshold=threshold)
        s = Episode(environment=bar, agents=agents, model='MFP', num_rounds=num_rounds)

        # Run simulation
        df = s.simulate(num_trials=num_trials, verbose=False)

        # Calculate average score for the current epsilon value
        avg_score = df['score'].mean()
        avg_scores.append(avg_score)

    p = PlotsAndMeasures(df)
    p.plot_scores_epsilon(epsilon_values, avg_scores, file=f'./data/scores_mfp_{num_agents}agents_epsilons.png')


def compare_cooldown():
    # Define simulation parameters
    num_agents = 2
    threshold = .6
    num_rounds = 500
    num_episodes = 100
    measures = ['score']
    verbose = True
    # Define agents
    epsilon = 0.01
    belief_strength = 0.1
    states = list(product([0,1], repeat=num_agents))
    alphas = {(x,y):1/len(states) for x in states for y in states}
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [epsilon_greedy(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create experiment
    experiment = Experiment(environment=bar,\
                   agents=agents,\
                   num_rounds=num_rounds,\
                   num_episodes=num_episodes,\
                   measures=measures,\
                    parameters=parameters)
    # Run sweep
    parameter = 'epsilon'
    values = [0.01, None]
    experiment.run_sweep1(parameter=parameter, values=values)
