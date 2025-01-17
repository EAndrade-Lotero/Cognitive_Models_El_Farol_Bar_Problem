
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

from Utils.interaction import Performer
from Utils.LaTeX_utils import PrintLaTeX
from Classes.cognitive_model_agents import CogMod

def initialize_folders(agent_class:CogMod):
    #-------------------------------
    # Define folders
    #-------------------------------
    folder_name = agent_class.name
    image_folder = Path.cwd() / Path('..').resolve() / Path('images', folder_name)
    image_folder.mkdir(parents=True, exist_ok=True)
    data_folder = Path.cwd() / Path('..').resolve() / Path('data', folder_name)
    data_folder.mkdir(parents=True, exist_ok=True)

def full_report(
            agent_class:CogMod,
            free_parameters:Dict[str,any], 
            fixed_parameters:Dict[str,any], 
            simulation_parameters:Dict[str,any]
        ) -> None:
    #-------------------------------
    # Initialize folders
    #-------------------------------
    initialize_folders(agent_class)
    #-------------------------------
    # Add section
    #-------------------------------
    latex_string = '\n' + rf'\section{agent_class.name}' + '\n\n' 
    #-------------------------------
    # Add fixed parameters
    #-------------------------------
    latex_string += PrintLaTeX.print_parameters(
        parameters=fixed_parameters,
        are_free=False,
    )
    #-------------------------------
    # Add sweep learning_rate
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0,
        "learning_rate": 0.01,
        "discount_factor": 0.8
    }
    latex_string += Performer.sweep(
        agent_class=QAttendance,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='learning_rate',
        values=[0.001, 0.01, 0.05, 0.1],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'

    #-------------------------------
    # Add sweep inverse_temperature
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=Q_learning,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='inverse_temperature',
        values=[2, 4, 8, 16, 32, 64],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'
    #-------------------------------
    # Add sweep discount_factor
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=QAttendance,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='discount_factor',
        values=[0, 0.2, 0.4, 0.8],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'    
    #-------------------------------
    # Add sweep go_drive
    #-------------------------------
    latex_string += Performer.sweep(
        agent_class=QAttendance,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='go_drive',
        values=[0, 0.2, 0.4, 0.8],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'   
    #-------------------------------
    # Add visual examination of best parameters
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0.8,
        "learning_rate": 0.05,
        "discount_factor": 0.8
    }
    latex_string += Performer.simple_run(
        agent_class=QAttendance,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        seeds=None,
        # seeds=[6, 38, 87], # chosen for num_players = 2
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


def explore_discount():
    #-------------------------------
    # Add section
    #-------------------------------
    latex_string = '\n' + r'\section{Q-learning}' + '\n\n' 
    #-------------------------------
    # Add fixed parameters
    #-------------------------------
    latex_string += PrintLaTeX.print_parameters(
        parameters=fixed_parameters,
        are_free=False,
    )
    #-------------------------------
    # Add sweep learning_rate
    #-------------------------------
    free_parameters = {
        "inverse_temperature": inverse_temperature,
        "go_drive": 0,
        "learning_rate": 0.1,
        "discount_factor": 0.8
    }
    latex_string += Performer.sweep(
        agent_class=Q_learning,
        fixed_parameters=fixed_parameters,
        free_parameters=free_parameters,
        simulation_parameters=simulation_parameters,
        sweep_parameter='discount_factor',
        values=[0, 0.2, 0.4, 0.8],
        image_folder=image_folder
    )
    latex_string += '\n' + r'\newpage' + '\n'    
    #-------------------------------
    # Wrap and save
    #-------------------------------
    latex_string = PrintLaTeX.wrap_with_header_and_footer(latex_string)
    latex_file = Path.joinpath(image_folder, 'report_.tex')
    PrintLaTeX.save_to_file(
        latex_string=latex_string,
        latex_file=latex_file
    )
