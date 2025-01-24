import torch
import numpy as np
from pathlib import Path
from random import choice
from copy import deepcopy
from itertools import product
from prettytable import PrettyTable
from torch.nn import MSELoss # CrossEntropyLoss
from typing import List, Tuple, Dict, Union, Optional


class ProxyDict :

    def __init__(
                self,
                keys: List[Tuple[int]],
                initial_val: float,
                round_dec: Optional[int]=2
            ) -> None:
        assert(isinstance(keys, list))
        self._keys = keys
        self.initial_val = initial_val
        self.data_dict = {key:self.initial_val for key in self._keys}
        self.round_dec = round_dec

    def __len__(self) -> int:
        return len(self.data_dict)

    def reset(self):
        self.data_dict = {key:self.initial_val for key in self._keys}

    def __call__(self, key: List[int]) -> float:
        return self.data_dict[key]

    def update(self, key: List[int], new_value: any) -> None:
        self.data_dict[key] = new_value

    def increment(self, key: List[int]) -> None:
        self.data_dict[key] += 1

    def keys(self) -> List[Tuple[int]]:
        return self._keys

    def sum(self) -> float:
        return sum(self.data_dict.values())
    
    def normalize(self) -> None:
        row_sum = self.sum()
        row_sum = row_sum if row_sum != 0 else 1
        for key in self.data_dict.keys():
            self.data_dict[key] /= row_sum

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
                round_dec: Optional[int]=2,
                uniform: Optional[bool]= True
            ) -> None:
        self.num_agents = num_agents
        self.round_dec = round_dec
        num_rows = np.power(2, self.num_agents)
        num_cols = np.power(2, self.num_agents)
        if uniform:
            self.trans_freqs = np.ones((num_rows, num_cols)) * (1 / num_cols)
        else:
            self.trans_freqs = np.zeros((num_rows, num_cols))

    def reset(self):
        num_rows = np.power(2, self.num_agents)
        num_cols = np.power(2, self.num_agents)
        self.trans_freqs = np.ones((num_rows, num_cols)) * (1 / num_cols)

    def __len__(self) -> int:
        return self.trans_freqs.shape[0]

    def __call__(self, transition: Tuple[List[int], List[int]]) -> float:
        row, col = self.get_indices(transition)
        return self.trans_freqs[row, col]

    def get_indices(self, transition: Tuple[List[int], List[int]]) -> Tuple[int, int]:
        prev_state, state = transition
        row = int("".join(str(x) for x in prev_state), 2)
        col = int("".join(str(x) for x in state), 2)
        return (row, col)

    def get_state_from_index(self, index: int) -> Tuple[int]:
        binary = "{0:b}".format(index)
        binary = list(binary)
        binary = [0 for _ in range(self.num_agents - len(binary))] + binary
        return tuple(binary)

    def get_convergence(self):
        return np.max(self.trans_freqs)
    
    def update(
                self, 
                transition: Tuple[List[int], List[int]],
                value: float
            ) -> None:
        row, col = self.get_indices(transition)
        self.trans_freqs[row, col] = value

    def increment(
                self, 
                transition: Tuple[List[int], List[int]],
            ) -> None:
        row, col = self.get_indices(transition)
        self.trans_freqs[row, col] += 1

    def normalize(self, rows: Optional[bool]=True) -> None:
        if rows:
            for i in range(self.trans_freqs.shape[0]):
                row_sum = sum(self.trans_freqs[i])
                row_sum = row_sum if row_sum != 0 else 1
                self.trans_freqs[i] /= row_sum
        else:
            for i in range(self.trans_freqs.shape[1]):
                col_sum = sum(self.trans_freqs[i])
                col_sum = col_sum if col_sum != 0 else 1
                self.trans_freqs[:,i] /= col_sum

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
        states = list(product([0,1], repeat=self.num_agents))
        table = PrettyTable(field_names=[''] + states)
        for x in states:
            row = [x] + [round(self((x,y)), self.round_dec) for y in states]
            table.add_row(row)
        return str(table)


class MLP(torch.nn.Module):
    '''
    A Multi-layer Perceptron
    '''
    def __init__(
                self, 
                sizes: List[int], 
                intermediate_activation_function: any,
                last_activation_function: Union[None, any],
                alpha: float
            ) -> None:
        """
        Args:
            sizes (list): list with the sizes of the layers. 
                          The convention is sizes[0] is the size of the input layer
                          and sizes[-1] is the size of the output layer.
            last_activation_function (an activation function)
        """
        super().__init__()
        assert(len(sizes) > 1)
        self.sizes = sizes
        self.intermediate_activation_function = intermediate_activation_function
        self.last_activation_function = last_activation_function
        self.model = torch.nn.Sequential()
        # -------------------------------------
        # Create layers
        # -------------------------------------
        for i in range(len(sizes) - 1):
            n_from = sizes[i]
            n_to = sizes[i+1]
            self.model.append(torch.nn.Linear(n_from, n_to))
            if i < len(sizes) - 2:
                self.model.append(self.intermediate_activation_function)
                # self.model.append(torch.nn.LayerNorm(n_to))
        if self.last_activation_function is not None:
            self.model.append(self.last_activation_function)
        # -------------------------------------
        # Define model parameters
        # -------------------------------------
        # self.loss_fn = CrossEntropyLoss
        self.loss_fn = MSELoss()
        self.alpha = 1e-3
        network_parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(network_parameters, lr=alpha)
        self.losses = []
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.backup_NN = deepcopy(self.model)

    def forward(self, x_in: List[int]):
        """The forward pass of the network        
        Args:
            x_in (torch.Tensor): an input data tensor. 
        Returns:
            the resulting tensor.
        """
        # Convert list to tensor
        x_in = torch.tensor(x_in, dtype=torch.float32).to(self.device)
        # Run the input through layers 
        return self.model(x_in)

    def predict(self, state, action):
        with torch.no_grad():
            # Get predicted Q values
            Qs = self.model(state)
            if len(Qs.shape) > 1:
                Qs = Qs.squeeze()
            # Transform to list
            Qs = Qs.data.tolist()
        return Qs[action]
    
    def values_vector(self, state):
        with torch.no_grad():
            if isinstance(state, list):
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
            elif isinstance(state, tuple):
                state = torch.tensor(list(state), dtype=torch.float32).to(self.device)
            elif isinstance(state, torch.tensor):
                state = state.to(self.device)
            # Get predicted Q values
            Qs = self.model(state)
            if len(Qs.shape) > 1:
                Qs = Qs.squeeze()
            # Transform to list
            Qs = Qs.data.tolist()
        return Qs
    
    def update(self, prev_state: List[int], obs_state: List[int]) -> None:
        '''
        Trains the NN with the given obs_state
        '''
        X = torch.tensor(prev_state, dtype=torch.float32, requires_grad=True).to(self.device)
        print(f'State:{X}')
        Y = torch.tensor(obs_state, dtype=torch.float32, requires_grad=True).to(self.device)
        print(f'Next state:{Y}')
        # Clear the gradient
        self.optimizer.zero_grad()
        # Get the batch predicted probabilities
        hat_probs = self.model(X)
        print(f'hat_probs:{hat_probs}')
        # Determine loss
        loss = self.loss_fn(hat_probs, Y)
        print(f'Loss: {loss}')
        self.losses.append(loss.item())
        # Find the gradients by backward propagation
        loss.backward()
        # Update the weights with the optimizer
        self.optimizer.step()

    def save(self, file:Path):
        torch.save(self.NN.state_dict(), file)

    def load(self, file:Path):
        self.NN.load_state_dict(torch.load(file))

    def restart(self):
        pass

    def reset(self):
        self.restart()
        # Instantiate original model
        self.model = deepcopy(self.backup_NN)
        # Create optimizer
        network_parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(network_parameters, lr=self.alpha)
        # Restart losses
        self.losses = []

    def summary(self):
        table = PrettyTable(['Modules', 'Parameters'])
        total_params = 0
        for name, parameter in self.NN.model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f'Total Trainable Params: {total_params}')
        print(f'Model device: {self.device}')

    def __str__(self):
        states = list(product([0,1], repeat=self.sizes[0]))
        table = PrettyTable([''] + [str(s) for s in range(self.sizes[0])])
        for prev_state in states:
            dummies = [round(x,2) for x in self.values_vector(prev_state)]
            table.add_row([str(prev_state)] + dummies)
        return str(table)


# class MLP(Module) :

#     def __init__(
#             self, 
#             n_x:int,
#             n_layers:List[int],
#             n_y:int
#         ) -> None:
#         super().__init__()
#         n_layers_all = [n_x] + n_layers + [n_y]
#         self.model = Sequential()
#         for i in range(len(n_layers_all) - 1):
#             n_in = n_layers_all[i]
#             n_out = n_layers_all[i+1]
#             self.model.append(Linear(n_in, n_out))
#             if i < len(n_layers_all) - 1:
#                 self.model.append(torch.nn.Sigmoid())
#         self.model.append(torch.nn.Softmax())
#         self.losses = []
#         self.loss_fn = CrossEntropyLoss()
#         self.optimizer = Adam(self.model.parameters(), lr=1e-3)

#     def forward(self, x_in):
#         return self.model(x_in)
    
#     def learn(self, ds_loader:DataLoader):
#         '''
#         Trains the NN with the given dataset
#         '''
#         for batch_states, batch_probs in tqdm(ds_loader, desc='Batch', leave=False):
#             # print(f'batch_states:{batch_states}')
#             # print(f'batch_probs:{batch_probs}')
#             # Clear the gradient
#             self.optimizer.zero_grad()
#             # Get the batch predicted probabilities
#             batch_hat_probs = self.NN.forward(batch_states)
#             # print(f'batch_hat_probs:{batch_hat_probs}')
#             # Determine loss
#             # print(f'X:{batch_X} --- Y:{batch_Y}')
#             loss = self.loss_fn(batch_hat_probs, batch_probs)
#             self.losses.append(loss.item())
#             # Find the gradients by backward propagation
#             loss.backward()
#             # Update the weights with the optimizer
#             self.optimizer.step()

#     def save(self, file:Path):
#         torch.save(self.NN.state_dict(), file)

#     def load(self, file:Path):
#         self.NN.load_state_dict(torch.load(file))


# class ExperienceDataset(Dataset):
#     '''
#     Creates the dataset out of the experience stream
#     '''
#     def __init__(
#                 self, 
#                 states:List[torch.Tensor], 
#                 probabilities:List[torch.Tensor]
#             ) -> None:
#         self.states = states
#         self.probabilities = probabilities
#         assert (len(self.probabilities) == len(self.states))

#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, idx:int):
#         # print(type(self.states[idx]))
#         # print(type(self.probabilities[idx]))
#         state = self.states[idx].to(torch.float32)
#         probability = self.probabilities[idx].to(torch.float32) 
#         return state, probability