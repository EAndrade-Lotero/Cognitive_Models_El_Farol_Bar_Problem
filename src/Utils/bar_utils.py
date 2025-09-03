import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import (
    Union, 
    Optional, 
 )

from Utils.utils import PPT


class BarRenderer :

    def __init__(
                self, 
                data:pd.DataFrame,
                image_file: Optional[Union[Path, None]]=None
            ) -> None:
        self.data = data
        self.history = self.get_history()
        self.thresholds = list()
        group_column = PPT.get_group_column(data.columns)
        self.room = data[group_column].unique()[0]
        num_players_column = PPT.get_num_player_column(data.columns)
        self.num_players = data[num_players_column].unique()[0]
        self.image_file = image_file
        # Determine color
        self.go_color='blue'
        self.no_go_color='lightgray'
        self.dpi = 300

    def __str__(self) -> str:
        return f'room:{self.room} --- num_players:{self.num_players} --- thresholds:{self.thresholds}'

    def render(
                self,
                ax: Optional[Union[plt.axis, None]]=None, 
                title: Optional[Union[str, None]]=None,
                num_rounds: Optional[int]=30
            ) -> plt.axis:
        if self.image_file is not None:
            file = PathUtils.add_file_name(
                path=self.image_file, 
                file_name=f'room{self.room}',
                extension='png'
            )
        self.render_threshold(
            ax=ax,
            title=title, 
            num_rounds=num_rounds
        )

    def get_history(self):
        history = list()
        for round, grp in self.data.groupby('round'):
            history.append(grp.decision.tolist())
        return history

    def render_threshold(
                self, 
                ax: Optional[Union[plt.axis, None]]=None,
                title: Optional[Union[str, None]]=None,
                num_rounds: Optional[int]=30
            ) -> None:
        '''
        Renders the history of attendances.
        '''
        # Use only last num_rounds rounds
        history = self.history[-num_rounds:]
        len_padding = num_rounds - len(history)
        if len_padding > 0:
            history = [[2 for _ in range(self.num_players)] for i in range(len_padding)] + history
        # Convert the history into format player, round
        decisions = [[h[i] for h in history] for i in range(self.num_players)]
        # Create plot
        if ax is None:
            fig, axes = plt.subplots(figsize=(0.5*num_rounds,0.5*self.num_players))
        else:
            axes = ax
        # Determine step sizes
        step_x = 1/num_rounds
        step_y = 1/self.num_players
        # Draw rectangles (go_color if player goes, gray if player doesnt go)
        tangulos = []
        for r in range(num_rounds):
            for p in range(self.num_players):
                if decisions[p][r] == 1:
                    color = self.go_color
                elif decisions[p][r] == 0:
                    color = self.no_go_color
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
                    facecolor=self.no_go_color,
                    linewidth=1
                )
            )
        for p in range(self.num_players + 1):
            # Draw border
            tangulos.append(
                patches.Rectangle(
                    (len_padding*step_x,p*step_y),1,0,
                    edgecolor='black',
                    facecolor=self.no_go_color,
                    linewidth=1
                )
            )
        for t in tangulos:
            axes.add_patch(t)
        axes.axis('off')
        if title is not None:
            axes.set_title(title)
        if self.image_file is not None:
            plt.savefig(self.image_file, dpi=self.dpi)
            print(f'Bar attendance saved to file {self.image_file}')
        else:
            plt.plot()
        return ax
        