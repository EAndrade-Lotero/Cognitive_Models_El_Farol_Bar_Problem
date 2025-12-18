import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import Union
from itertools import permutations, combinations
from typing import List, Optional, Dict, Tuple

from Config.config import PATHS
from Utils.cherrypick_simulations import CherryPickEquilibria

class FocalRegion:
    '''
    Determines next action based on the focal region.
    '''

    def __init__(self, 
                focal_region: np.ndarray,
                category: str,
                c: Optional[float] = 0.9,
                steepness: Optional[float] = 10,
            ) -> None:
        assert(isinstance(focal_region, np.ndarray)), f"Error: region should be an np.ndarray, not {type(focal_region)}"
        self.focal_region = focal_region
        self.c = c
        self.steepness = steepness
        self.debug = False
        self.shape = focal_region.shape
        self.category = category

    def similarity_score(self, region1: np.ndarray, region2: np.ndarray) -> float:
        '''Jaccard similarity score between two regions.'''
        assert region1.shape == region2.shape, f"Regions must have the same shape (but got {region1.shape} and {region2.shape})\n{region1}\n\n{region2}"
        return np.sum(region1 == region2) / np.prod(region1.shape)
        
    def get_region(self, n_cols: int, idx:int) -> np.ndarray:
        if idx + n_cols <= self.focal_region.shape[1]:
            region = self.focal_region[:, idx:idx+n_cols]
        else:
            max_idx = ((idx + n_cols) - self.focal_region.shape[1])
            # print(f'(({idx} + {n_cols}) - {self.focal_region.shape[1]}) = {max_idx}')
            region_ahead = self.focal_region[:, idx:]
            # print(f'Region ahead:\n{region_ahead}')
            region_behind = self.focal_region[:, 0:max_idx]
            # print(f'Region behind:\n{region_behind}')
            region = np.concatenate((region_ahead, region_behind), axis=1)
        return region

    def get_similarity_scores(self, history: np.ndarray) -> List[float]:
        scores = []
        if self.debug:
            print('='*60)
        for i in range(self.focal_region.shape[1]):
            n_cols = history.shape[1]
            region = self.get_region(n_cols, i)
            score = self.similarity_score(history, region)
            scores.append(score)
            if self.debug:
                print(f'\tCicle from column {i}:\n{region}')
                print(f'\tSimilarity score: {score}')
                print('-'*60)
        return scores

    def get_long_history_similarity_score(self, history: np.ndarray) -> List[float]:
        scores = []
        if self.debug:
            print('='*60)
        num_repetitions = history.shape[1] // self.focal_region.shape[1]
        for j in range(self.focal_region.shape[1]):
            score_segment_list = []
            for i in range(num_repetitions):
                start_col = j + i * self.focal_region.shape[1]
                end_col = j + (i + 1) * self.focal_region.shape[1]
                if end_col > history.shape[1]:
                    break
                history_segment = history[:, start_col:end_col]
                scores_segment = self.similarity_score(history_segment, self.focal_region)
                score_segment_list.append(scores_segment)
            score = np.mean(score_segment_list)
            scores.append(score)
        return max(scores)

    def get_action_preferences(
                self, 
                history: np.ndarray,
                agent_id: int
            ) -> np.ndarray:
        scores = self.get_similarity_scores(history)
        if self.debug:
            print('-'*60)
            print(f'Scores: {scores}')
            print(f'Finding preferences for player {agent_id} according to region')
        action_preferences = np.zeros(2)
        num_columns_region = self.focal_region.shape[1]
        len_history = history.shape[1]
        for idx_col in range(num_columns_region):
            # Align history with region at column idx and find next column idx
            next_idx_col = (idx_col + len_history) % num_columns_region
            # Find action according to pattern at next column idx
            action = int(self.focal_region[agent_id, next_idx_col])
            if self.debug:
                msg = f"Pattern at column {idx_col} assigns similarity {scores[idx_col]} to action={'go' if action == 1 else 'no-go'}"
                print(msg)
            # Assign preferences according to similarity score
            raw_preferences = np.zeros(2)
            raw_preferences[action] = scores[idx_col]
            raw_preferences[1 - action] = 1 - scores[idx_col]
            if self.debug:
                print(f"\tRaw preferences: {raw_preferences}")
            # Pass through logistic
            logistic_preferences = self.normalized_logistic(raw_preferences)
            if self.debug:
                print(f"\tLogistic preferences: {logistic_preferences}")
            # Add to action preferences
            action_preferences += logistic_preferences
        if self.debug:
            print(f"Added and normalized preferences: {action_preferences}")
            print('-'*60)
        return action_preferences

    def get_action_preferences_max(
                self, 
                history: np.ndarray,
                agent_id: int
            ) -> np.ndarray:
        scores = self.get_similarity_scores(history)
        if self.debug:
            print(f'Scores: {scores}')
        idx_similarity = np.argmax(scores)
        idx_col = (idx_similarity + history.shape[1]) % self.focal_region.shape[1]
        action = int(self.focal_region[agent_id, idx_col])
        action_preferences = np.zeros(2)
        action_preferences[action] = scores[idx_similarity]
        return action_preferences

    def __str__(self):
        return '-'*60 + '\n' + str(self.focal_region) + '\n' + '-'*60

    @staticmethod
    def cycle_region(region: np.ndarray, idx: int) -> np.ndarray:
        n_cols = region.shape[1]
        indices = np.arange(0, n_cols)
        indices = np.roll(indices, -idx)
        region = region[:, indices] 
        return region

    @staticmethod
    def draw_region(
                region: np.ndarray, 
                title: Optional[str] = None,
                axes:Union[plt.Axes, None]=None,
                file:Union[Path, None]=None
            ) -> plt.Axes:
        # Get number of rounds and agents
        num_rounds = region.shape[1]
        num_agents = region.shape[0]
        region = np.flipud(region)
        len_padding = 0
        # Create plot
        if axes is None:
            fig, axes = plt.subplots(
                figsize=(num_rounds, num_agents)
            )
        # Determine step sizes
        step_x = 1/num_rounds
        step_y = 1/num_agents
        # Determine color
        go_color='blue'
        no_go_color='lightgray'
        # Draw rectangles (go_color if player goes, gray if player doesnt go)
        tangulos = []
        for r in range(num_rounds):
            for p in range(num_agents):
                if region[p, r] == 1:
                    color = go_color
                elif region[p, r] == 0:
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
        for p in range(num_agents + 1):
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
            axes.add_patch(t)
        axes.axis('off')
        if title is not None:
            axes.set_title(title)
        if file is not None:
            plt.savefig(file, dpi=300)
        return axes

    def normalized_logistic(self, x: np.ndarray) -> float:
        """
        Normalized logistic map [0,1] -> [0,1].

        Parameters
        ----------
        x : float
            Input in [0,1].
        steepness : float
            Controls the slope of the transition. Higher = sharper.
        threshold : float
            The midpoint of the S‐curve (where f(x)=0.5).

        Returns
        -------
        float
            f(x) in [0,1].
        """
        # raw logistic
        raw = 1.0 / (1.0 + np.exp(-self.steepness * (x - self.c)))

        # compute endpoints
        raw0 = 1.0 / (1.0 + np.exp( self.steepness * self.c))      # f(0) before normalization
        raw1 = 1.0 / (1.0 + np.exp(-self.steepness * (1.0 - self.c)))  # f(1) before normalization

        # shift and scale so that f(0)==0 and f(1)==1
        return (raw - raw0) / (raw1 - raw0)


class SetFocalRegions:
    '''Set of focal regions to be shared by all agents.'''
    def __init__(
                self, 
                num_agents: int, 
                threshold: float,
                len_history: int,
                c: Optional[float] = 0.9,
                steepness: Optional[float] = 20,
                max_regions: Optional[int] = 1,
                from_file: Optional[bool] = True,
                seed: Optional[Union[int, None]] = None
            ) -> None:
        self.num_agents = num_agents
        self.threshold = threshold
        self.B = int(num_agents * threshold)
        self.len_history = min(int(len_history), num_agents)
        self.c = c
        self.steepness = steepness
        self.focal_regions = []
        self.max_regions = min(int(max_regions), num_agents*2)
        self.history = None
        self.debug = False
        if seed is None:
            seed = np.random.randint(1000)
        self.rng = np.random.default_rng(seed)
        cherrypick = CherryPickEquilibria(
            num_agents=self.num_agents,
            threshold=self.threshold,
            epsilon=0,
            num_rounds=1,
            num_episodes=1,
            seed=seed
        )
        cherrypick.debug = False
        self.cherrypick = cherrypick
        self.from_file = from_file
        self.create_file_path()

    def create_file_path(self) -> None:
        # file = f'{self.max_regions}_regions'
        file = f'_{self.num_agents}_agents'
        file += f'_{self.threshold}_threshold.json'
        self.file = PATHS['focal_regions_path'] / file

    def add_history(self, obs: List[int]) -> None:
        obs_array = np.array(obs).reshape(-1, 1)
        if self.history is None:
            self.history = obs_array
        else:
            self.history = np.concatenate((self.history, obs_array), axis=1)
            self.history = self.history[:, -self.len_history:]

    def generate_focal_regions(self) -> None:
        '''Generates focal regions.'''
        if self.from_file and not self.file.exists():
            raise FileNotFoundError(f"Focal regions file {self.file} does not exist.")
        if self.from_file and self.file.exists():
            if self.debug:
                print(f'Loading focal regions from {self.file}')
            self.focal_regions = self.load_focal_regions()
            return
        if self.debug:
            print(f'Generating focal regions for {self.num_agents} agents and {self.threshold} threshold')
        fair_regions = self.generate_fair_regions()
        segmented_regions = self.generate_segmented_regions()
        mixed_regions = self.generate_mixed_regions()
        if self.debug:
            print('Equalizing region sizes')
        regions = self.equal_region_sizes([
            fair_regions, 
            segmented_regions, 
            mixed_regions
        ])
        self.focal_regions = regions
        if self.from_file:
            if self.debug:
                print(f'Saving focal regions to {self.file}')
            self.save_focal_regions()

    def load_focal_regions(self) -> List[FocalRegion]:
        '''Loads focal regions from file.'''
        if not self.file.exists():
            raise FileNotFoundError(f"Focal regions file {self.file} does not exist.")
        data = json.load(open(self.file, 'r'))
        regions = []
        for region_dict in data:
            region = np.array(region_dict['region'])
            category = region_dict.get('category')
            region_ = FocalRegion(
                focal_region=region,
                category=category,
                c=self.c,
                steepness=self.steepness
            )
            regions.append(region_)
        return regions

    def save_focal_regions(self) -> None:
        '''Saves focal regions to file.'''
        data = [{'region':region.focal_region.tolist(), 'category':region.category} for region in self.focal_regions]
        with open(self.file, 'w') as f:
            json.dump(data, f)

    def generate_segmented_regions(self) -> List[FocalRegion]:
        regions = []
        for region in self.cherrypick.get_all_standard_segmented_equilibriums(period=self.num_agents):
            region_ = FocalRegion(
                focal_region=region,
                category='segmented',
                c=self.c,
                steepness=self.steepness
            )
            regions.append(region_)
        return regions

    def generate_fair_regions(self) -> List[FocalRegion]:
        regions = []
        for region in self.cherrypick.get_all_standard_fair_periodic_equilibrium(period=self.num_agents):
            region_ = FocalRegion(
                focal_region=region,
                category='alternation',
                c=self.c,
                steepness=self.steepness
            )
            regions.append(region_)
        return regions

    def generate_mixed_regions(self) -> List[FocalRegion]:
        regions = []
        for region in self.cherrypick.get_all_standard_mixed_periodic_equilibrium(period=self.num_agents):
            region_ = FocalRegion(
                focal_region=region,
                category='mixed',
                c=self.c,
                steepness=self.steepness
            )
            regions.append(region_)
        return regions

    def get_action_preferences(self, agent_id: int) -> np.ndarray:
        # Clipping history
        self.history = self.history[:, -self.len_history:]
        # Print for debug
        if self.debug:
            print('='*60)
            print(f"Considering preferences from the viewpoint of agent {agent_id}")
            print('-'*60)
        action_preferences = np.zeros(2)
        for i, region in enumerate(self.focal_regions):
            raw_preferences = region.get_action_preferences(self.history, agent_id)
            # preferences = self.sigmoid(raw_preferences)
            preferences = self.normalized_logistic(raw_preferences)
            if self.debug:
                print(f'Similarities according to region {i}: {raw_preferences}')
                print(f'\tSigmoid similarities: {preferences}')
            action_preferences += preferences
        if self.debug:
            print(f'Aggregated preferences: (no go={action_preferences[0]}; go={action_preferences[1]})')
        if np.sum(action_preferences) == 0:
            action_preferences = np.array([0.5, 0.5])
        else:
            action_preferences /= np.sum(action_preferences)
        if self.debug:
            print(f'Normalized preferences: (no go={action_preferences[0]}; go={action_preferences[1]})')
        return action_preferences

    def normalized_logistic(self, x: np.ndarray) -> float:
        """
        Normalized logistic map [0,1] -> [0,1].

        Parameters
        ----------
        x : float
            Input in [0,1].
        steepness : float
            Controls the slope of the transition. Higher = sharper.
        threshold : float
            The midpoint of the S‐curve (where f(x)=0.5).

        Returns
        -------
        float
            f(x) in [0,1].
        """
        # raw logistic
        raw = 1.0 / (1.0 + np.exp(-self.steepness * (x - self.c)))

        # compute endpoints
        raw0 = 1.0 / (1.0 + np.exp( self.steepness * self.c))      # f(0) before normalization
        raw1 = 1.0 / (1.0 + np.exp(-self.steepness * (1.0 - self.c)))  # f(1) before normalization

        # shift and scale so that f(0)==0 and f(1)==1
        return (raw - raw0) / (raw1 - raw0)

    def equal_region_sizes(self, list_regions: List[List[FocalRegion]]) -> List[List[FocalRegion]]:
        '''Making sure all regions have the same length'''
        #----------------------------------------
        # Finding lengths of each type of region
        #----------------------------------------
        lengths = [len(regions) for regions in list_regions]
        non_zero_lengths = [l for l in lengths if l > 0]
        m = len(non_zero_lengths)
        n = self.max_regions // m
        res = self.max_regions % m
        target_lengths = [n if l > 0 else 0 for l in lengths]
        first_non_zero_idx = next((i for i, l in enumerate(lengths) if l > 0), None)
        if first_non_zero_idx is not None:
            target_lengths[first_non_zero_idx] += res
        #----------------------------------------
        # Equalizing
        #----------------------------------------
        for i, regions in enumerate(list_regions):
            if len(regions) > target_lengths[i]:
                idx_regions = self.rng.choice(
                    range(len(regions)),
                    size=target_lengths[i],
                    replace=False
                )
                list_regions[i] = [regions[i] for i in idx_regions]
            elif 0 < len(regions) < target_lengths[i]:
                idx_regions = [i % len(regions) for i in range(target_lengths[i])]
                list_regions[i] = [regions[i] for i in idx_regions]
        return [region for sublist in list_regions for region in sublist]

    def __str__(self):
        cadena = ''
        for i, region in enumerate(self.focal_regions):
            cadena += '=' * 60 + '\n'
            cadena += f"Region {i}\n"
            cadena += str(region) + '\n'
        return cadena
    
    def __len__(self):
        return len(self.focal_regions)



