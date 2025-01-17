from Utils.interaction import Episode
from Utils.parameter_optimization import GetMeasure
from Classes.cognitive_model_agents import Random
from Utils.utils import ConditionalEntropy

def find_bug_random():
	fixed_parameters = {
		'num_agents': 8,
		'threshold': 0.5,
	}
	free_parameters = {
		"go_prob": 0.5,
	}
	gm = GetMeasure(
		agent_class=Random,
		free_parameters=free_parameters,
		fixed_parameters=fixed_parameters
	)
	gm.num_rounds = 1000
	gm.num_episodes = 1
	df = gm.generate_simulated_data(free_parameters)
	# print(df.columns)
	# print(df.head())
	# print(df.attendance)

	ce = ConditionalEntropy(df)
	tm = ce.get_group_states(df)
	# print(tm)
	entropy = ce.calculate_entropy(tm)
	print(f'Entropy = {entropy}')