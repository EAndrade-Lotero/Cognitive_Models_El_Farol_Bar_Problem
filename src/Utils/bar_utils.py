import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import (
    Union,
    Optional,
    Dict,
    Any,
)

from Utils.utils import PPT


class BarRenderer:

    def __init__(
                self,
                data: pd.DataFrame,
                image_file: Optional[Union[Path, None]] = None,
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
        self.go_color = 'blue'
        self.no_go_color = 'lightgray'
        self.edgecolor = '0.35'
        self.edge_linewidth = 0.5
        self.under_capacity_color = '#2A9D8F'
        self.over_capacity_color = '#E76F51'
        # Tile height / width (< 1 → wider than tall, left-to-right emphasis)
        self.cell_height_over_width = 0.55
        self.capacity_bar_height = 0.18
        self.capacity_bar_gap = 0.06
        self.dpi = 300

    def __str__(self) -> str:
        return f'room:{self.room} --- num_players:{self.num_players} --- thresholds:{self.thresholds}'

    def render(
                self,
                ax: Optional[Union[plt.axis, None]] = None,
                title: Optional[Union[str, None]] = None,
                num_rounds: Optional[int] = 30,
                title_kwargs: Optional[Dict[str, Any]] = None,
                capacity: Optional[float] = None,
                show_capacity_bar: bool = True,
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
            num_rounds=num_rounds,
            title_kwargs=title_kwargs,
            capacity=capacity,
            show_capacity_bar=show_capacity_bar,
        )

    def get_history(self):
        history = list()
        for round, grp in self.data.groupby('round'):
            history.append(grp.decision.tolist())
        return history

    def _resolve_capacity(self, capacity: Optional[float]) -> Optional[float]:
        if capacity is not None:
            return float(capacity)
        if 'threshold' not in self.data.columns:
            return None
        threshold = float(self.data['threshold'].iloc[0])
        # Fractional thresholds map to absolute capacity (int); keep absolute as-is.
        if 0 < threshold <= 1:
            return float(int(threshold * self.num_players))
        return threshold

    def render_threshold(
                self,
                ax: Optional[Union[plt.axis, None]] = None,
                title: Optional[Union[str, None]] = None,
                num_rounds: Optional[int] = 30,
                title_kwargs: Optional[Dict[str, Any]] = None,
                capacity: Optional[float] = None,
                show_capacity_bar: bool = True,
            ) -> None:
        '''
        Renders the history of attendances.

        Tiles use unit width and height ``cell_height_over_width`` (< 1) so each
        cell is wider than tall. Optional capacity bars below each round are
        green when attendance ≤ capacity and red otherwise.
        '''
        # Use only last num_rounds rounds
        history = self.history[-num_rounds:]
        len_padding = num_rounds - len(history)
        if len_padding > 0:
            history = [[2 for _ in range(self.num_players)] for i in range(len_padding)] + history
        # Convert the history into format player, round
        decisions = [[h[i] for h in history] for i in range(self.num_players)]
        n_players = int(self.num_players)
        cell_h = float(self.cell_height_over_width)
        bar_h = float(self.capacity_bar_height) if show_capacity_bar else 0.0
        bar_gap = float(self.capacity_bar_gap) if show_capacity_bar else 0.0
        board_top = n_players * cell_h
        board_bottom = 0.0
        # Create plot
        if ax is None:
            board_h = board_top + (bar_gap + bar_h if show_capacity_bar else 0.0)
            fig, axes = plt.subplots(
                figsize=(0.28 * num_rounds, max(0.9, 0.28 * board_h / cell_h))
            )
        else:
            axes = ax

        capacity_val = self._resolve_capacity(capacity)
        tangulos = []
        # Draw player tiles: width = 1, height = cell_h (< 1 → wider than tall)
        for r in range(num_rounds):
            for p in range(n_players):
                if decisions[p][r] == 1:
                    color = self.go_color
                elif decisions[p][r] == 0:
                    color = self.no_go_color
                else:
                    color = 'none'
                y = (n_players - 1 - p) * cell_h
                tangulos.append(
                    patches.Rectangle(
                        (r, y), 1.0, cell_h,
                        facecolor=color,
                        edgecolor=self.edgecolor,
                        linewidth=self.edge_linewidth,
                    )
                )

        # Capacity indicators under each round (skip padded lead-in rounds)
        if show_capacity_bar and capacity_val is not None:
            bar_y = board_bottom - bar_gap - bar_h
            for r in range(len_padding, num_rounds):
                attendance = sum(1 for p in range(n_players) if decisions[p][r] == 1)
                bar_color = (
                    self.under_capacity_color
                    if attendance <= capacity_val
                    else self.over_capacity_color
                )
                tangulos.append(
                    patches.Rectangle(
                        (r + 0.08, bar_y), 0.84, bar_h,
                        facecolor=bar_color,
                        edgecolor='none',
                    )
                )

        for t in tangulos:
            axes.add_patch(t)

        x0, x1 = 0.0, float(num_rounds)
        y0 = board_bottom - (bar_gap + bar_h if show_capacity_bar else 0.0)
        y1 = board_top
        axes.set_xlim(x0, x1)
        axes.set_ylim(y0, y1)
        axes.set_aspect('equal', adjustable='box')
        axes.axis('off')
        if title is not None:
            kwargs = {'fontsize': 12, 'pad': 6}
            if title_kwargs:
                kwargs.update(title_kwargs)
            axes.set_title(title, **kwargs)
        if self.image_file is not None:
            plt.savefig(self.image_file, dpi=self.dpi)
            print(f'Bar attendance saved to file {self.image_file}')
        elif ax is None:
            plt.plot()
        return ax
