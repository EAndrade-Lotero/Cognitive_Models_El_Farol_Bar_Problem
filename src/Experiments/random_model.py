from pathlib import Path
import pandas as pd

from Classes.bar import Bar
from Classes.agents import Random
from Utils.interaction import Episode

image_folder = Path('..', 'images', 'random_model')
image_folder.mkdir(exist_ok=True)
data_folder = Path('..', 'data', 'random_model')
data_folder.mkdir(exist_ok=True)


def create_tests_data():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 10
    num_episodes = 10
    verbose = False
    #----------------------------------------
    # Create test with p=0
    #----------------------------------------
    # Define agents
    go_prob = 0
    parameters = {"go_prob":go_prob}
    agents = [Random(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(
        num_agents=num_agents, 
        threshold=threshold
    )
    # Create simulation
    episode = Episode(
        environment=bar,\
        agents=agents,\
        model=f'Random({go_prob})',\
        num_rounds=num_rounds
    )
    # Run simulation
    df = episode.simulate(
        num_episodes=num_episodes, 
        verbose=verbose
    )
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random-0.csv')
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')
    #----------------------------------------
    # Create test with p=1
    #----------------------------------------
    # Define agents
    go_prob = 1
    parameters = {"go_prob":go_prob}
    agents = [Random(parameters, n) for n in range(num_agents)]
    # Create simulation
    episode = Episode(
        environment=bar,\
        agents=agents,\
        model=f'Random({go_prob})',\
        num_rounds=num_rounds
    )
    # Run simulation
    df = episode.simulate(
        num_episodes=num_episodes, 
        verbose=verbose
    )
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random-1.csv')
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')
    #----------------------------------------
    # Create test with p=0.5
    #----------------------------------------
    # Define agents
    go_prob = 0.5
    parameters = {"go_prob":go_prob}
    agents = [Random(parameters, n) for n in range(num_agents)]
    # Create simulation
    episode = Episode(
        environment=bar,\
        agents=agents,\
        model=f'Random({go_prob})',\
        num_rounds=num_rounds
    )
    # Run simulation
    df = episode.simulate(
        num_episodes=num_episodes, 
        verbose=verbose
    )
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random-05.csv')
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')


def random_simple_experiment():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 10
    num_episodes = 23
    verbose = False
    # Define agents
    go_prob = 0.5
    parameters = {"go_prob":go_prob}
    agents = [Random(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(
        num_agents=num_agents, 
        threshold=threshold
    )
    # Create simulation
    episode = Episode(
        environment=bar,\
        agents=agents,\
        model=f'Random({go_prob})',\
        num_rounds=num_rounds
    )
    # Run simulation
    df = episode.simulate(
        num_episodes=num_episodes, 
        verbose=verbose
    )
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random.csv')
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')
    # # Plot results
    # p = PlotsAndMeasures(df)
    # file = Path.joinpath(image_folder, 'random_eq_100Pl.png')
    # p.plot_EQ(mu=threshold, file=file)
    # print(f'Image saved to {file}')

def random_two_player_maxlikelyhood():
    # Define simulation parameters
    num_rounds = 10
    num_episodes = 10
    verbose = False
    # Get empirical data
    file_path = Path('..', 'data', 'human', 'random-two-player-maxlikely.csv')
    df = pd.read_csv(file_path, index_col=0)
    list_dfs = list()
    for index, row in df.iterrows():
        print(row)
        num_agents = row['num_players']
        threshold = row['threshold']
        go_probabilities = [float(x) for x in eval(row['go_probs'])]
        agents = [Random({"go_prob":go_probabilities[n]}, n) for n in range(num_agents)]
        # Create bar
        bar = Bar(
            num_agents=num_agents, 
            threshold=threshold
        )
        # Create simulation
        episode = Episode(
            environment=bar,\
            agents=agents,\
            model=f'mu:{threshold}; N:{num_agents} -- MaxLik',\
            num_rounds=num_rounds
        )
        # Run simulation
        df = episode.simulate(
            num_episodes=num_episodes, 
            verbose=verbose
        )
        df['room'] = row['room']
        list_dfs.append(df)
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random-sim-two-players-maxlikely.csv')
    df = pd.concat(list_dfs, ignore_index=True)
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')

def random_multi_player_experiment():
    # Define simulation parameters
    num_agents_per_threshold = {
        3: [0.33, 0.67],
        4: [0.25, 0.5, 0.75],
        5: [0.8, 0.6, 0.4],
        6: [0.33, 0.5, 0.67],
        7: [0.43, 0.71, 1.0],
        8: [0.88, 0.62, 0.38],
        9: [0.33, 0.56, 0.78],
        11: [0.27, 0.45, 0.64],
        12: [0.25, 0.5, 0.75]
    }
    num_rounds = 30
    num_episodes = 10
    verbose = False
    list_dfs = list()
    for num_agents, thresholds in num_agents_per_threshold.items():
        for threshold in thresholds:
            # Define agents
            go_prob = threshold
            parameters = {"go_prob":go_prob}
            agents = [Random(parameters, n) for n in range(num_agents)]
            # Create bar
            bar = Bar(
                num_agents=num_agents, 
                threshold=threshold
            )
            # Create simulation
            episode = Episode(
                environment=bar,\
                agents=agents,\
                model=f'mu:{threshold}; N:{num_agents}',\
                num_rounds=num_rounds
            )
            # Run simulation
            df = episode.simulate(
                num_episodes=num_episodes, 
                verbose=verbose
            )
            df['threshold'] = threshold
            df['num_players'] = num_agents
            df['treatment'] = 'simulation'
            df['trial'] = 1
            list_dfs.append(df)
    df = pd.concat(list_dfs, ignore_index=True)
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random_multi.csv')
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')

def random_multi_player_maxlikelyhood():
    # Define simulation parameters
    num_rounds = 10
    num_episodes = 10
    verbose = False
    # Get empirical data
    file_path = Path('..', 'data', 'human', 'random-multi-player-maxlikely.csv')
    df = pd.read_csv(file_path, index_col=0)
    list_dfs = list()
    for index, row in df.iterrows():
        print(row)
        num_agents = row['num_players']
        threshold = row['threshold']
        go_probabilities = [float(x) for x in eval(row['go_probs'])]
        agents = [Random({"go_prob":go_probabilities[n]}, n) for n in range(num_agents)]
        # Create bar
        bar = Bar(
            num_agents=num_agents, 
            threshold=threshold
        )
        # Create simulation
        episode = Episode(
            environment=bar,\
            agents=agents,\
            model=f'mu:{threshold}; N:{num_agents} -- MaxLik',\
            num_rounds=num_rounds
        )
        # Run simulation
        df = episode.simulate(
            num_episodes=num_episodes, 
            verbose=verbose
        )
        df['room'] = row['room']
        df['threshold'] = threshold
        df['num_players'] = num_agents
        df['treatment'] = 'simulation'
        df['trial'] = 1
        list_dfs.append(df)
    # # Save results to cvs
    file = Path.joinpath(data_folder, 'random-sim-multi-player-maxlikely.csv')
    df = pd.concat(list_dfs, ignore_index=True)
    df.to_csv(file, index=False)
    print(f'Data saved to {file}')