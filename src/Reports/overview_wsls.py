
from pathlib import Path
from typing import Dict, List, Union, Optional

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.cognitive_model_agents import WSLS

folder_name = 'WSLS'
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

def full_report():
    #-------------------------------
    # Add section
    #-------------------------------
    latex_string = '\n' + r'\section{WSLS}' + '\n\n' 
    #-------------------------------
    # Add fixed parameters
    #-------------------------------
    latex_string += PrintLaTeX.print_parameters(
        parameters=fixed_parameters,
        are_free=False,
    )
    #-------------------------------
    # Add sweep drive to go
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0.25,
        "wsls_strength": 1
    }
    latex_string += Performer.sweep(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='go_drive',
        values=[0, 0.25, 0.5, 0.75],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add visual examination of best drive to go
    #-------------------------------
    latex_string += Performer.simple_run(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=None,
        # seeds=[0, 4], # chosen for num_players = 2
        # seeds=[73, 89, 10, 37], # chosen for num_players = 8
        image_folder=image_folder
    )
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0.5,
        "wsls_strength": 1
    }
    latex_string += Performer.simple_run(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=[15, 40],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add sweep drive to go
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0.25,
        "wsls_strength": 1
    }
    latex_string += Performer.sweep(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='wsls_strength',
        values=[0, 0.5, 1, 2],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add sweep inverse temperature
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='inverse_temperature',
        values=[4, 8, 16, 32, 64],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add visual examination of best drive to go
    #-------------------------------
    free_parameters = {
        "inverse_temperature": 8,
        "go_drive": 0.25,
        "wsls_strength": 1
    }
    latex_string += Performer.simple_run(
        agent_class=WSLS,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=None,
        # seeds=[7, 74], # chosen for num_players = 2
        # seeds=[], # chosen for num_players = 8
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