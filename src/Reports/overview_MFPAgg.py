
from pathlib import Path
from itertools import product
from typing import Dict, List, Union, Optional

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.agent_utils import ProxyDict
from Classes.cognitive_model_agents import MFPAgg

folder_name = 'MFP_Agg'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', folder_name)
data_folder.mkdir(parents=True, exist_ok=True)

#-------------------------------
# Define simulation parameters
#-------------------------------
num_agents = 8
threshold = 0.5
inverse_temperature = 64

fixed_parameters = {
    'num_agents': num_agents,
    'threshold': threshold,
}
simulation_parameters = {
    'num_rounds': 100,
    'num_episodes': 100,
    'verbose': 0,
}


def examinations():
    #-------------------------------
    # Examine specific simulation
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "belief_strength":1,\
        "go_drive":1
    }
    Performer.examine_simulation(
        agent_class=MFPAgg,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        semilla=0
    )

   


def full_report():
    #-------------------------------
    # Add section
    #-------------------------------
    latex_string = '\n' + r'\section{Aggregated Markov Fictitious Play}' + '\n\n' 
    #-------------------------------
    # Add fixed parameters
    #-------------------------------
    latex_string += PrintLaTeX.print_parameters(
        parameters=fixed_parameters,
        are_free=False,
    )
    #-------------------------------
    # Add sweep go_drive
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "belief_strength":1,
        "go_drive":0.4
    }
    latex_string += Performer.sweep(
        agent_class=MFPAgg,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='go_drive',
        values=[0, 0.2, 0.4, 0.8],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add sweep belief_strength
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=MFPAgg,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='belief_strength',
        values=[1, 2, 4, 8],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add sweep inverse_temperature
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=MFPAgg,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='inverse_temperature',
        values=[2, 4, 8, 16, 32, 64],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add visual examination of best parameters
    #-------------------------------
    latex_string += Performer.simple_run(
        agent_class=MFPAgg,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=None,
        # seeds=[11, 73, 13], # chosen for num_players = 2
        image_folder=image_folder
    )
    #-------------------------------
    # Wrap and save
    #-------------------------------
    latex_string = PrintLaTeX.wrap_with_header_and_footer(latex_string)
    latex_file = Path.joinpath(image_folder, 'report.tex')
    PrintLaTeX.save_to_file(
        latex_string=latex_string,
        latex_file=latex_file
    )