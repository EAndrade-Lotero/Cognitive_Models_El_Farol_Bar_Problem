
from pathlib import Path
from typing import Dict, List, Union, Optional

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.cognitive_model_agents import Random

folder_name = 'Random'
image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
image_folder.mkdir(parents=True, exist_ok=True)
data_folder = Path.cwd() / Path('..').resolve() / Path('data', folder_name)
data_folder.mkdir(parents=True, exist_ok=True)

#-------------------------------
# Define simulation parameters
#-------------------------------
num_agents = 8
threshold = 0.5

fixed_parameters = {
    'num_agents': num_agents,
    'threshold': threshold,
}
simulation_parameters = {
    'num_rounds': 100,
    'num_episodes': 100,
    'verbose': 0,
}

def examine():
    free_parameters = {
        "go_prob": 1,
    }
    Performer.examine_simulation(
        agent_class=Random,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        semilla=0
    )


def full_report():
    #-------------------------------
    # Add section
    #-------------------------------
    latex_string = '\n' + r'\section{Random}' + '\n\n' 
    #-------------------------------
    # Add fixed parameters
    #-------------------------------
    latex_string += PrintLaTeX.print_parameters(
        parameters=fixed_parameters,
        are_free=False,
    )
    #-------------------------------
    # Add sweep go_prob
    #-------------------------------
    free_parameters = {
        "go_prob": 0,
    }
    latex_string += Performer.sweep(
        agent_class=Random,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='go_prob',
        values=[0, 0.1, 0.3, 0.5],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add visual examination of best drive to go
    #-------------------------------
    free_parameters = {
        "go_prob": 0.3,
    }
    latex_string += Performer.simple_run(
        agent_class=Random,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=None,
        # seeds=[0, 1, 2], # chosen for num_players = 2
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