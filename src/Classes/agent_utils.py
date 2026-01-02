import numpy as np
from pathlib import Path
from random import choice
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from typing import List, Tuple, Dict, Union, Optional


class ProxyDict :

    def __init__(
                self,
                keys: List[Tuple[int]],
                initial_val: float,
                round_dec: Optional[int]=2
            ) -> None:
        assert(isinstance(keys, list)), f'keys should be a list, not {type(keys)}.\nThe keys received are: {keys}'
        self._keys = keys
        self.initial_val = initial_val
        self.data_dict = {key:self.initial_val for key in self._keys}
        self.round_dec = round_dec

    def __len__(self) -> int:
        return len(self.data_dict)

    def reset(self):
        self.data_dict = {key:self.initial_val for key in self._keys}

    def __call__(self, key: List[int]) -> float:
        try:
            return self.data_dict[key]
        except KeyError:
            raise KeyError(f'Key {key} not found in ProxyDict. Available keys are: {self.keys()}')

    def __getitem__(self, key: List[int]) -> float:
        try:
            return self.data_dict[key]
        except KeyError:
            raise KeyError(f'Key {key} not found in ProxyDict. Available keys are: {self.keys()}')

    def __setitem__(self, key: List[int], value:any) -> None:
        self.update(key, value)

    def update(self, key: List[int], new_value: any) -> None:
        self.data_dict[key] = new_value

    def increment(self, key: List[int]) -> None:
        self.data_dict[key] += 1

    def keys(self) -> List[Tuple[int]]:
        return self._keys

    def sum(self) -> float:
        return sum(self.data_dict.values())
    
    def normalize(self) -> 'ProxyDict':
        pd = ProxyDict(
            keys=self._keys,
            initial_val=self.initial_val,
            round_dec=self.round_dec
        )
        row_sum = self.sum()
        row_sum = row_sum if row_sum != 0 else 1
        for key in self.data_dict.keys():
            value = self.data_dict[key] / row_sum
            pd[key] = round(value, self.round_dec)
        return pd

    def as_array(self) -> np.ndarray:
        return np.array(list(self.data_dict.values()))

    def from_dict(self, given_dict: Dict[Tuple[int], float]) -> None:
        for key, value in given_dict.items():
            self.update(
                key=key,
                new_value=value
            )

    def __str__(self) -> str:
        table = PrettyTable(field_names=list(self.data_dict.keys()))
        row = [round(value, self.round_dec) for value in self.data_dict.values()]
        table.add_row(row)
        return str(table)


class TransitionsFrequencyMatrix :

    def __init__(
                self,
                num_agents: int,
                size: Optional[Union[Tuple[int, int], None]]=None,
                round_dec: Optional[int]=2,
                uniform: Optional[bool]= True
            ) -> None:
        self.num_agents = num_agents
        self.round_dec = round_dec
        if size is None:
            self.num_rows = np.power(2, self.num_agents)
            self.num_cols = np.power(2, self.num_agents)
        else:
            assert(isinstance(size, tuple))
            assert(isinstance(size[0], Union[int, np.int64])), f"Error: expected int but got {type(size[0])} (value={size[0]})"
            assert(isinstance(size[1], Union[int, np.int64])), f"Error: expected int but got {type(size[1])} (value={size[1]})"
            self.num_rows = size[0]
            self.num_cols = size[1]
        self.uniform = uniform
        if uniform:
            self.default_value = 1 / self.num_cols
        else:
            self.default_value = 0
        self.trans_freqs = dict()
    
    def reset(self):
        if self.uniform:
            self.default_value = 1 / self.num_cols
        else:
            self.default_value = 0
        self.trans_freqs = dict()

    def __len__(self) -> int:
        return len(self.trans_freqs)

    def __call__(self, transition: Tuple[List[int], List[int]]) -> float:
        row, col = self.get_states(transition)
        if row in self.trans_freqs.keys():
            row_dict = self.trans_freqs[row]
            value = row_dict.get(col, self.default_value)
            return value
        return self.default_value
    
    def iter_rows(self) -> List[Tuple[int]]:
        return [self.get_state_from_index(i) for i in range(self.num_rows)]
    
    def iter_cols(self) -> List[Tuple[int]]:
        return [self.get_state_from_index(i) for i in range(self.num_cols)]

    def row_sum(self, row:int) -> Union[int, float]:
        if row in self.trans_freqs.keys():
            row_dict = self.trans_freqs[row]
            known_sum = sum([value for key, value in row_dict.items()])
            unknown_sum = (self.num_cols - len(row_dict)) * self.default_value
            return known_sum + unknown_sum
        else:
            return self.num_cols * self.default_value
    
    def row_normalized(self) -> 'TransitionsFrequencyMatrix':
        normalized_tm = TransitionsFrequencyMatrix(
            num_agents=self.num_agents,
            size=(self.num_rows, self.num_cols),
            round_dec=self.round_dec,
            uniform=True
        )
        for row in self.trans_freqs.keys():
            row_dict = self.trans_freqs[row]
            row_sum = self.row_sum(row)
            if row_sum == 0: row_sum = 1
            for col in self.iter_cols():
                transition = (row, col)
                value = row_dict.get(col, 0) / row_sum
                normalized_tm.update(transition, value)
        return normalized_tm
    
    def get_states(self, transition: Tuple[List[int], List[int]]) -> Tuple[int, int]:
        prev_state, state = transition
        row = "".join(str(x) for x in prev_state)
        col = "".join(str(x) for x in state)
        return (row, col)

    def get_indices(self, transition: Tuple[List[int], List[int]]) -> Tuple[int, int]:
        prev_state, state = transition
        row = int("".join(str(x) for x in prev_state), 2)
        col = int("".join(str(x) for x in state), 2)
        return (row, col)

    def get_state_from_index(self, index: int) -> Tuple[int]:
        # return index
        binary = "{0:b}".format(index)
        binary = list(binary)
        binary = [0 for _ in range(self.num_agents - len(binary))] + binary
        tupla = tuple(binary)
        return "".join(str(x) for x in tupla)

    def get_convergence(self):
        return np.max(self.trans_freqs)
    
    def update(
                self, 
                transition: Tuple[List[int], List[int]],
                value: float
            ) -> None:
        row, col = self.get_states(transition)
        if row in self.trans_freqs.keys():
            self.trans_freqs[row][col] = value
        else:
            self.trans_freqs[row] = {col: value}

    def increment(
                self, 
                transition: Tuple[List[int], List[int]],
            ) -> None:
        row, col = self.get_states(transition)
        if row in self.trans_freqs.keys():
            row_dict = self.trans_freqs[row]
            if col in row_dict.keys():
                self.trans_freqs[row][col] += 1
            else:
                self.trans_freqs[row][col] = 1
        else:
            self.trans_freqs[row] = {col: 1}

    def from_dict(self, trans_probs: Dict[Tuple[int], float]) -> None:
        for transition in trans_probs.keys():
            self.update(
                transition=transition,
                value=trans_probs[transition]
            )

    def from_proxydict(self, trans_probs: Dict[Tuple[int], float]) -> None:
        for transition in trans_probs.keys():
            self.update(
                transition=transition,
                value=trans_probs(transition)
            )

    def __str__(self) -> str:
        row_states = self.iter_rows()
        col_states = self.iter_cols()
        table = PrettyTable(field_names=[''] + col_states)
        for x in row_states:
            row = [x] + [round(self((x, y)), self.round_dec) for y in col_states]
            table.add_row(row)
        return str(table)

    def as_array(self) -> np.ndarray:
        my_array = np.ones((self.num_rows, self.num_cols)) * self.default_value
        for row in self.trans_freqs.keys():
            row_dict = self.trans_freqs[row]
            for col in row_dict.keys():
                row_idx, col_idx = self.get_indices((row, col))
                my_array[row_idx, col_idx] = self.trans_freqs[row][col]
        return my_array



