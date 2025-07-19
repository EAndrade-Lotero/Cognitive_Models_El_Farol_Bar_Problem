import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from itertools import permutations, combinations
from typing import Optional, Union, Tuple, List

from Classes.bar import Bar

class CherryPickEquilibria:
    '''Creates periods of Nash equilibria'''

    def __init__(
                self, 
                num_agents: int, 
                threshold: float, 
                epsilon: float,
                num_rounds: Optional[int]=100,
                num_episodes: Optional[int]=100,
                seed: Optional[Union[int, None]]=None
            ) -> None:
        self.num_agents = num_agents
        self.agents = list(range(self.num_agents))
        assert(0 <= threshold <= 1), f'The threshold must be between 0 and 1 but got {threshold}'
        self.threshold = threshold
        self.B = int(num_agents * threshold)
        assert(0 <= epsilon <= 1), f'The epsilon must be between 0 and 1 but got {epsilon}'
        assert(num_rounds > 0), f'The number of rounds must be positive but got {num_rounds}'
        assert(num_episodes > 0), f'The number of episodes must be positive but got {num_episodes}'
        self.epsilon = epsilon
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.debug = True

    def generate_data(self, kind:str) -> pd.DataFrame:
        assert(kind in ['segmentation', 'alternation', 'random'])
        df_list = list()
        for i in tqdm(range(self.num_episodes), desc='Running episodes', leave=False):
            if kind == 'segmentation':
                df = self.generate_segmentation_simulation()
            elif kind == 'alternation':
                df = self.generate_alternation_simulation()
            else:
                df = self.generate_random_simulation()
            df['id_sim'] = f'{self.num_agents}-{self.threshold}-{self.epsilon}-{kind}-{i}'
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        return df

    def generate_segmentation_simulation(self) -> pd.DataFrame:
        # Generate go array
        go_array = self.get_segmented_equilibrium(self.num_rounds)
        go_array = self.apply_epsilon(go_array).T
        # Generate dataframe
        df = self.generate_dataframe(go_array)
        return df

    def generate_alternation_simulation(self) -> pd.DataFrame:
        # Generate go array
        # go_array = self.get_fair_periodic_equilibrium(self.num_rounds)
        go_array = self.random_fair_periodic_equilibrium(self.num_rounds)
        go_array = self.apply_epsilon(go_array).T
        # Generate dataframe
        df = self.generate_dataframe(go_array)
        return df
    
    def generate_random_simulation(self) -> pd.DataFrame:
        # Generate go array
        # go_array = self.rng.integers(low=0, high=2, size=(self.num_agents, self.num_rounds))
        go_array = (self.rng.random((self.num_agents, self.num_rounds)) < self.threshold).astype(int)
        # Generate dataframe
        df = self.generate_dataframe(go_array)
        return df

    def generate_dataframe(self, go_array:np.ndarray) -> pd.DataFrame:
        # Get scores
        bar = Bar(self.num_agents, self.threshold)
        list_scores = list()
        for decisions in go_array:
            _, scores = bar.step(decisions)
            list_scores.append(scores)
        scores_array = np.array(list_scores)
        # Get correct list of rounds
        rounds = [[i + 1] * self.num_agents for i in range(self.num_rounds)]
        rounds = np.array(rounds)
        df_dict = {'round':rounds.flatten()}
        df_dict['id_player'] = list(range(self.num_agents)) * self.num_rounds
        df_dict['decision'] = go_array.flatten().astype(int)
        df_dict['score'] = scores_array.flatten()
        df = pd.DataFrame.from_dict(df_dict)
        df['num_agents'] = self.num_agents
        df['threshold'] = self.threshold
        return df

    def random_fair_periodic_equilibrium(self, period:int) -> np.ndarray:
        periodic_equilibrium = self.get_fair_periodic_equilibrium(period)
        periodic_equilibrium = periodic_equilibrium.T
        np.random.shuffle(periodic_equilibrium)
        return periodic_equilibrium.T

    def apply_epsilon(self, periodic_equilibrium:np.ndarray) -> np.ndarray:
        new_equilibrium = [
            [
                1 - a if self.rng.uniform(0,1) < self.epsilon else a for a in row
            ] for row in periodic_equilibrium
        ]
        return np.array(new_equilibrium)

    def get_all_standard_segmented_equilibriums(self, period:int) -> List[np.ndarray]:
        regions = []
        goers = combinations(range(self.num_agents), self.B)
        for goers in goers:
            region = np.zeros((self.num_agents, 1))
            region[goers, 0] = 1
            go_agents = np.concatenate([region] * period, axis=1)
            regions.append(go_agents)
        return regions
    
    def get_all_standard_fair_periodic_equilibrium(self, period:int) -> List[np.ndarray]:
        # Create base equilibrium
        base_equilibrium = self.get_fair_periodic_equilibrium(period)
        num_rows, num_cols = base_equilibrium.shape
        list_equilibriums = []
        list_str = []
        # Permute rows of base equilibrium
        for row_indices in permutations(range(num_rows)):
            equilibrium = base_equilibrium[row_indices, :].copy()
            # Permute columns of base equilibrium and skip if one variation is included
            included = False
            for col_idx in range(num_cols):
                rolled_variation = np.roll(equilibrium, col_idx, axis=1)
                if str(rolled_variation) in list_str:
                    included = True
                    break
            if not included:
                list_equilibriums.append(equilibrium)
                list_str.append(str(equilibrium))
        return list_equilibriums

    def get_all_standard_mixed_periodic_equilibrium(self, period:int) -> List[np.ndarray]:
        list_equilibriums = []
        list_str = []
        for num_seg in range(1, self.B):
            # Get mixed periodic equilibrium
            mixed_equilibrium = self.get_mixed_periodic_equilibrium(num_seg, period)
            num_rows, num_cols = mixed_equilibrium.shape
            # Permute rows of base equilibrium
            for row_indices in permutations(range(num_rows)):
                equilibrium = mixed_equilibrium[row_indices, :].copy()
                # Permute columns of base equilibrium and skip if one variation is included
                included = False
                for col_idx in range(num_cols):
                    rolled_variation = np.roll(equilibrium, col_idx, axis=1)
                    if str(rolled_variation) in list_str:
                        included = True
                        break
                if not included:
                    list_equilibriums.append(equilibrium)
                    list_str.append(str(equilibrium))
        return list_equilibriums

    def get_fair_periodic_equilibrium(self, period:int) -> np.ndarray:
        # Get number of lowest payoff players
        L = self.num_agents - self.B
        # Creare array with player's decisions
        go_array = np.zeros((self.num_agents, period))
        # Create basic one-shot equilibrium
        go_agents = np.concatenate([np.zeros((L,)), np.ones((self.B,))]) 
        # Permute one-shot equilibrium
        for i in range(period):
            index = i % self.num_agents
            round_go_agents = np.concatenate([go_agents[index:], go_agents[:index]])
            go_array[:,i] = round_go_agents    
        if self.debug:
            print(f'Periodic equilibrium:\n{go_array}')            
        return go_array

    def get_mixed_periodic_equilibrium(self, num_seg:int, period:int) -> np.ndarray:
        assert(num_seg > 0), f'Number of agents always going to the bar (num_seg={num_seg}) must be positive'
        assert(num_seg < self.B), f'Number of agents always going to the bar (num_seg={num_seg}) cannot be higher than, or equal to, the bar capacity (B={self.B})'
        # Get number of alternating players
        num_alt = self.num_agents - num_seg
        # Creare array with player's decisions
        # Get number of lowest payoff players
        L = self.num_agents - self.B
        go_array = np.zeros((self.num_agents, period))
        # Create basic one-shot equilibrium
        go_agents = np.concatenate([np.ones((self.B,)), np.zeros((L,))]) 
        # Permute one-shot equilibrium
        for i in range(period):
            index = num_seg + (i % num_alt)
            round_go_agents = np.concatenate([
                go_agents[:num_seg],
                go_agents[index:], 
                go_agents[num_seg:index]
            ])
            go_array[:,i] = round_go_agents    
        if self.debug:
            print(f'Periodic equilibrium:\n{go_array}')            
        return go_array

    def get_segmented_equilibrium(self, period:int) -> np.ndarray:
        go_agents = self.random_one_shot_equilibrium()
        one_shot_equilibrium = np.zeros((self.num_agents,))
        one_shot_equilibrium[go_agents] = 1
        equilibrium = [one_shot_equilibrium.tolist()  for _ in range(period)]
        equilibrium = np.array(equilibrium).T
        if self.debug:
            print(f'Segmented equilibrium:\n{equilibrium}')
        return equilibrium

    def get_num_highest_payoff_players(self, period:int) -> int:
        # Initialize records
        num_highest_payoff_players = None
        # Calculate  fair nucleus cycle
        nucleus_cycle = np.ceil(self.num_agents / self.B)
        # Check if period is multiple of Fair Period
        if period % nucleus_cycle == 0:
            # Check if bar's capacity is higher than half number of agents
            if 2 * self.B > self.num_agents:
                num_highest_payoff_players = 2 * self.B - self.num_agents
            elif 2 * self.B == self.num_agents:
                num_highest_payoff_players = self.num_agents
            else:
                num_highest_payoff_players = self.num_agents % self.B
        else:
            num_highest_payoff_players = self.B * (period % nucleus_cycle)
        assert(num_highest_payoff_players <= self.num_agents)
        if self.debug:
            print(f'Number of highest payed players: {num_highest_payoff_players}')
        return int(num_highest_payoff_players)
        
    def get_fair_quantities(self, period:int) -> Tuple[int, int]:
        # Calculate Fair Period
        nucleus_cycle = np.ceil(self.num_agents / self.B)
        if self.debug:
            print(f'Nucleus cycle: {nucleus_cycle}')
        # Get Fair Quantity given period
        fair_quantity = int(self.threshold * period)
        if self.debug:
            print(f'Fair quantity: {fair_quantity}')
        # Check if Fair Quantity is integer
        if fair_quantity == self.threshold * period:
            low_fair_quantity = fair_quantity - 1
            high_fair_quantity = fair_quantity
        else:
            low_fair_quantity = fair_quantity
            high_fair_quantity = fair_quantity + 1
        if self.debug:
            print(f'Lowest fair quantity: {low_fair_quantity}')
            print(f'Highest fair quantity: {high_fair_quantity}')        
        return low_fair_quantity, high_fair_quantity

    def random_one_shot_equilibrium(self) -> np.ndarray:
        go_agents = self.rng.choice(self.agents, size=self.B, replace=False)
        if self.debug:
            print(f'Equilibrium of {self.B} agents:\n{go_agents}')
        return go_agents

    def random_periodic_equilibrium(self, period:int) -> np.ndarray:
        period = int(period)
        assert(period <= self.num_agents)
        one_shot_equilibria = list(permutations(self.agents, r=self.B))
        periodic_equilibrium = self.rng.choice(one_shot_equilibria, size=period, replace=True)
        if self.debug:
            for k, agents in enumerate(periodic_equilibrium):
                print(f'On round {k} agents that go are {agents}')
        return periodic_equilibrium