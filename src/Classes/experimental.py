import numpy as np

from itertools import permutations
from typing import List, Optional, Dict

from Classes.cognitive_model_agents import CogMod
from Utils.cherrypick_simulations import CherryPickEquilibria

class FocalRegion:
    '''
    Determines next action based on the focal region.
    '''

    def __init__(self, focal_region: np.ndarray):
        self.focal_region = focal_region
        self.debug = False

    def similarity_score(self, region1: np.ndarray, region2: np.ndarray) -> float:
        '''Jaccard similarity score between two regions.'''
        return np.sum(region1 == region2) / np.prod(region1.shape)
        
    def get_region(self, history: np.ndarray, idx:int) -> np.ndarray:
        n_cols = history.shape[1]
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
        for i in range(self.focal_region.shape[1]):
            region = self.get_region(history, i)
            score = self.similarity_score(region, history)
            scores.append(score)
            if self.debug:
                print(f'Cicle from column {i}:\n{region}')
                print(f'Similarity score: {score}')
        return scores

    def get_action_preferences(
                self, 
                history: np.ndarray,
                agent_id: int
            ) -> np.ndarray:
        scores = self.get_similarity_scores(history)
        if self.debug:
            print(f'Scores: {scores}')
        idx_similarity = np.argmax(scores)
        # print(f'Idx similarity: {idx_similarity}')
        idx_col = (idx_similarity + history.shape[1]) % self.focal_region.shape[1]
        # print(f'Idx col: {idx_col}')        
        action = int(self.focal_region[agent_id, idx_col])
        # print(f'Action: {action}')
        action_preferences = np.zeros(2)
        action_preferences[action] = scores[idx_similarity]
        # action_preferences[1 - action] = 1 - scores[idx_similarity]
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


class SetFocalRegions:
    '''Set of focal regions to be shared by all agents.'''
    def __init__(
                self, 
                num_agents: int, 
                threshold: float,
                len_history: int,
                max_regions: Optional[int] = None,
            ) -> None:
        self.num_agents = num_agents
        self.threshold = threshold
        self.len_history = len_history
        self.B = int(num_agents * threshold)
        self.focal_regions = []
        self.max_regions = max_regions
        self.history = None
        self.debug = False

    def add_history(self, obs: List[int]) -> None:
        obs_array = np.array(obs).reshape(-1, 1)
        if self.history is None:
            self.history = obs_array
        else:
            self.history = np.concatenate((self.history, obs_array), axis=1)
            self.history = self.history[:, -self.len_history:]

    def generate_focal_regions(self) -> None:
        '''Generates focal regions.'''
        cherrypick = CherryPickEquilibria(
            num_agents=self.num_agents,
            threshold=self.threshold,
            epsilon=0,
            num_rounds=1,
            num_episodes=1,
            seed=42
        )
        cherrypick.debug = False
        regions = []
        equilibrium = cherrypick.get_fair_periodic_equilibrium(period=self.num_agents)
        for variation in permutations(range(equilibrium.shape[1]), self.num_agents):
            # print(f'Variation: {variation}')
            indices = np.array(variation)
            good_variation = True
            for i in range(indices.shape[0]):
                rolled_variation = np.roll(indices, i)
                # print(f'\tRolled variation: {rolled_variation}')
                if str(rolled_variation) in regions:
                    # print(f'\t\tRolled variation already in regions: {rolled_variation}')
                    good_variation = False
                    break
            if good_variation:
                region = equilibrium[:, indices]
                self.focal_regions.append(FocalRegion(region))
                regions.append(str(indices))
                if self.max_regions is not None and len(self.focal_regions) >= self.max_regions:
                    break

    def get_action_preferences(self, agent_id: int) -> np.ndarray:
        self.history = self.history[:, -self.len_history:]
        action_preferences = np.zeros(2)
        for i, region in enumerate(self.focal_regions):
            preferences = region.get_action_preferences(self.history, agent_id)
            if self.debug:
                print(f'Region {i}: {preferences}')
            action_preferences += preferences
        return action_preferences

    def __str__(self):
        cadena = ''
        for region in self.focal_regions:
            cadena += str(region)
        return cadena


class FocalRegionAgent(CogMod):
    '''
    Agent that uses focal regions to determine next action.
    '''
    def __init__(
                self,
                free_parameters: Optional[Dict[str, any]] = {},
                fixed_parameters: Optional[Dict[str, any]] = {},
                n: Optional[int] = 1
            ) -> None:
        super().__init__(free_parameters, fixed_parameters, n)
        self.len_history = free_parameters['len_history']
        sfr = SetFocalRegions(
            num_agents=self.num_agents,
            threshold=self.threshold,
            len_history=self.len_history, 
            max_regions=fixed_parameters['max_regions']
        )
        sfr.generate_focal_regions()
        self.sfr = sfr

    def determine_action_preferences(self) -> List[float]:
        preferences = self.sfr.get_action_preferences(self.number)
        if self.debug:
            print(f'Preferences: {preferences}')
        return preferences

    def update(self, score:int, obs_state:List[int]) -> None:
        self.sfr.add_history(obs_state)
        super().update(score, obs_state)

    @staticmethod
    def name():
        return 'FRA'
