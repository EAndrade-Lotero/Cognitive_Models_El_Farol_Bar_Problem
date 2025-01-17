from Utils.parameter_optimization import GetMeasure, ParameterOptimization
from Classes.cognitive_model_agents import *

fixed_parameters = {
	'num_agents': 2,
	'threshold': 0.5,
}
free_parameters = {
	"go_prob": 0.1,
}

def test_get_measure():
	gm = GetMeasure(
		agent_class=Random,
		free_parameters=free_parameters,
		fixed_parameters=fixed_parameters
	)
	gm.create_loss_function('efficiency')
	measure_value = gm.black_box_function(**free_parameters)
	print(measure_value)


def test_parameter_opt_random():
	po = ParameterOptimization(
		agent_class=Random,
		free_parameters=free_parameters,
		fixed_parameters=fixed_parameters,
		measure_class=GetMeasure,
		measure='efficiency',
		optimizer_name='bayesian'
	)
	result = po.get_optimal_parameters()
	print(result)

def test_parameter_opt_PRW():
	free_parameters = {
		"inverse_temperature": 16,
		"initial_reward_estimate_go": 0,
		"initial_reward_estimate_no_go": 0,
		"learning_rate": 0.1
	}
	po = ParameterOptimization(
		agent_class=PayoffRescorlaWagner,
		free_parameters=free_parameters,
		fixed_parameters=fixed_parameters,
		measure_class=GetMeasure,
		measure='efficiency',
		optimizer_name='bayesian'
	)
	result = po.get_optimal_parameters()
	print(result)
