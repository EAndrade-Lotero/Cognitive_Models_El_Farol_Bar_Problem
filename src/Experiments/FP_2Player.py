from Classes.bar import Bar
from Classes.agents import AgentFP
from Utils.interaction import Episode, Experiment
from random import seed
from pathlib import Path

image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'FP_2Player')
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', 'FP_2Player')
data_folder.mkdir(parents=True, exist_ok=True)


def draw_frequencies():
    # Define simulation parameters
    num_agents = 2
    threshold = .7
    num_rounds = 4
    # Define agents
#    alphas = [.25]*num_agents
    alphas = [.25, 0.75]
    belief_strength = 1
    epsilon = 0
    parameters = {"alphas":alphas, 
                  "num_agents":num_agents, 
                  "threshold":threshold,
                  "belief_strength":belief_strength,
                  "epsilon":epsilon
                  }
    agents = [AgentFP(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='FP',\
                   num_rounds=num_rounds)
    # Run simulation 
    episode.run(verbose=True)


def simple_draw_bar_attendances():
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 500
    # Define agents
    alphas = [.25] * num_agents
    belief_strength = 1
    epsilon = 0.1
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [AgentFP(parameters, n) for n in range(num_agents)]
    # Create bar
    bar = Bar(num_agents=num_agents, threshold=threshold)
    # Create simulation
    episode = Episode(environment=bar,\
                   agents=agents,\
                   model='FP',\
                   num_rounds=num_rounds)
    # Run simulation
    semillas = [17]
    # semillas = range(100)
    for semilla in tqdm(semillas, desc='Running seeds...', leave=False):
        seed(semilla)
        # Run simulation
        df = episode.simulate(verbose=0)
        file = Path.joinpath(image_folder, f'FP2_attendance_{semilla}.png')
        # file = Path.joinpath(image_folder, f'suboptimal_attendance_{semilla}.png')
        episode.environment.render(file=file)

def sweep_epsilon():
    '''
    A parameter sweep to compare performance on several values of epsilon
    '''
    # Define simulation parameters
    num_agents = 2
    threshold = .5
    num_rounds = 1000
    num_episodes = 1000
    measures = ['score', 'eq_coop']
    # Define agents
    alphas = [.25] * num_agents
    belief_strength = 1
    epsilon = 0.1
    parameters = {"belief_strength":belief_strength,\
                  "alphas":alphas,\
                  "num_agents":num_agents,\
                  "threshold":threshold,\
                  "epsilon": epsilon}
    agents = [AgentFP(parameters, n) for n in range(num_agents)]
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
    values = [0, 0.01, 0.1, 1]
    file = Path.joinpath(image_folder, 'sweep_epsilon_')
    experiment.run_sweep1(parameter=parameter, values=values, file=file)