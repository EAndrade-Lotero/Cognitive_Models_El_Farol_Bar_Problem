'''
Class with the El Farol bar environment
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import Union, List, Tuple

class Bar :
    '''
    Class for playing El Farol bar problem. Keeps tally of the number of players
    attending the bar each round and returns a list of scores depending on whether
    the bar overcrouds or not.
    '''
    
    def __init__(self, num_agents, threshold) :
        self.num_agents = num_agents
        self.threshold = threshold
        self.history = []

    def step(self, decisions:List[int], update:bool=True) -> Tuple[int, List[int]] :
        '''
        Computes the scores on the basis of the attendance.
        Input:
            - decisions, list with each player's decision (1=GO, 0=NO GO)
        Output:
            - attendance, number of players attending the bar
            - scores, list with a score for each player according to the following payoff matrix:
                1, if agent goes and attendance <= threshold*num_agents
                -1, if agent goes and attendance > threshold*num_agents
                0, if agent does not go
        '''
        assert(all([a in [0,1] for a in decisions]))
        attendance = sum(decisions)
        if update:
            self.history.append(decisions)
        scores = []
        for a in decisions:
            if a == 1:
                if attendance <= self.threshold * self.num_agents:
                    scores.append(1)
                else:
                    scores.append(-1)
            else:
                scores.append(0)
        return attendance, scores

    def reset(self):
        '''
        Goes back to initial state.
        '''
        self.history = []

    def render(
                self, 
                ax:Union[plt.axis, None]=None,
                file:Union[Path, None]=None, 
                num_rounds:int=15
            ) -> plt.axis:
        '''
        Renders the history of attendances.
        '''
        # Use only last num_rounds rounds
        history = self.history[-num_rounds:]
        len_padding = num_rounds - len(history)
        if len_padding > 0:
            history = [[2 for _ in range(self.num_agents)] for i in range(len_padding)] + history
        # Convert the history into format player, round
        decisions = [[h[i] for h in history] for i in range(self.num_agents)]
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(0.5*num_rounds, self.num_agents)
            )
        # Determine step sizes
        step_x = 1/num_rounds
        step_y = 1/self.num_agents
        # Determine color
        go_color='blue'
        no_go_color='lightgray'
        # Draw rectangles (go_color if player goes, gray if player doesnt go)
        tangulos = []
        for r in range(num_rounds):
            for p in range(self.num_agents):
                if decisions[p][r] == 1:
                    color = go_color
                elif decisions[p][r] == 0:
                    color = no_go_color
                else:
                    color = 'none'
                # Draw filled rectangle
                tangulos.append(
                    patches.Rectangle(
                        (r*step_x,p*step_y),step_x,step_y,
                        facecolor=color
                    )
                )
        for r in range(len_padding, num_rounds + 1):
            # Draw border
            tangulos.append(
                patches.Rectangle(
                    (r*step_x,0),0,1,
                    edgecolor='black',
                    facecolor=no_go_color,
                    linewidth=1
                )
            )
        for p in range(self.num_agents + 1):
            # Draw border
            tangulos.append(
                patches.Rectangle(
                    (len_padding*step_x,p*step_y),1,0,
                    edgecolor='black',
                    facecolor=no_go_color,
                    linewidth=1
                )
            )
        for t in tangulos:
            ax.add_patch(t)
        ax.axis('off')
        if file is not None:
            plt.savefig(file, dpi=300)
        return ax
        
    def to_pandas(self) -> pd.DataFrame:
        '''
        Creates a pandas dataframe with the information from the current objects.
        Output:
            - pandas dataframe with the following six variables:
            
            Variables:
                * id_sim: a unique identifier for the simulation
                * threshold: the bar's threshold
                * round: the round number
                * attendance: the round's attendance
                * id_player: the player's number
                * decision: the player's decision
                * score: the player's score
                * model: the model's name
                * convergence: the maximum difference between 
                            two previous approximations of 
                            probability estimates
        '''
        history = np.array(self.history).T
        data = {}
        data["id_sim"] = list()
        data["round"] = list()
        data["attendance"] = list()
        data["id_player"] = list()
        data["decision"] = list()
        data["score"] = list()
        for r in range(history.shape[1]):
            attendance, scores = self.step(history[:, r], update=False)
            for i in range(self.num_agents):
                data["id_sim"].append(1)
                data["round"].append(r)
                data["id_player"].append(i)
                data["decision"].append(history[i, r])
                data["attendance"].append(attendance)
                data["score"].append(scores[i])
        df = pd.DataFrame.from_dict(data)		
        df["model"] = None
        df["threshold"] = self.threshold
        df["num_agents"] = self.num_agents
        return df