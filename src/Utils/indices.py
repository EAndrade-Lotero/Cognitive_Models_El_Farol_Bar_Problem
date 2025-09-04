import torch
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import product
from typing import Optional, Union, Dict, List, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from Config.config import PATHS
from Utils.utils import GetMeasurements, PPT
from Utils.bar_utils import BarRenderer
from Classes.NN import SimpleMLP, SimpleCNN
from Utils.cherrypick_simulations import CherryPickEquilibria

class AlternationIndex:
    '''Estimates the alternation index'''

    def __init__(
                self, 
                num_points: Optional[int]=20,
                num_rounds: Optional[int]=100,
                num_episodes: Optional[int]=20,
                max_agents: Optional[int]=8, 
                max_epsilon: Optional[float]=0.025,
                seed: Optional[Union[int, None]]=None,
                fancy_2P: Optional[bool]=False,
            ) -> None:
        self.num_points = num_points
        self.num_rounds = num_rounds
        self.max_agents = max_agents
        self.max_epsilon = max_epsilon
        self.num_episodes = num_episodes
        self.seed = seed
        self.fancy_2P = fancy_2P
        self.rng = np.random.default_rng(seed=seed)
        self.configuration_points = self.create_configurations()
        # self.measures = ['bounded_efficiency', 'entropy', 'conditional_entropy', 'inequality']
        self.measures = ['bounded_efficiency', 'inequality']
        self.data = None
        self.full_data = None
        self.sklearn_coefficients = None
        self.statsmodels_coefficients = None
        self.model = None
        self.index_path = PATHS['index_path']
        self.priority = 'statsmodels'
        self.debug = True
        self.alternation_threshold = 0.75

    def __call__(self, df:pd.DataFrame) -> np.ndarray:
        '''Calculate the index from the dataframe'''
        # Get index of alternation
        probabilities = self.calculate_probabilities('alternation', df)
        return probabilities
    
    def calculate_probabilities(self, category:str, df:pd.DataFrame) -> np.ndarray:
        # Get index of category
        classes = CherryPickEquilibria.get_categories()
        idx_category = classes.index(category)
        # Obtain probabilities from model
        if self.model is None:
            self.create_index_calculator()
        if self.priority == 'cnn':
            df_, _ = self.get_x_y_values(df)
        else:
            df_ = df[self.measures]
        probabilities = self.model.predict_proba(df_)
        return probabilities[:, idx_category]
    
    def classify(self, df:pd.DataFrame) -> np.ndarray:

        def get_class(line):
            thresholds = [x > self.alternation_threshold for x in line]
            assert(sum(thresholds) <= 1), f"Oops: {line} fits in more than one category with threshold {self.alternation_threshold}"
            if sum(thresholds) == 0:
                return 'random'
            else:
                idx = thresholds.index(True)
                return classes[idx]

        '''Classify the simulations from the dataframe'''
        # Get classes
        classes = CherryPickEquilibria.get_categories()
        # Obtain probabilities from model
        if self.model is None:
            self.create_index_calculator()
        if self.priority == 'cnn':
            df_, _ = self.get_x_y_values(df)
        else:
            df_ = df[self.measures]
        probabilities = self.model.predict_proba(df_)
        predictions = [get_class(line) for line in probabilities]
        # predictions = [classes[np.argmax(line)] for line in probabilities]
        return predictions

    def alt_precentage(
                self, 
                df: pd.DataFrame,
                columns: Optional[List[str]]=None
            ) -> float:
        '''Calculate the alternation percentage'''
        data = df.copy()
        data['probabilities'] = self(data)
        data['alternation'] = data['probabilities'] > self.alternation_threshold  
        if columns is not None:
            data_ = data.groupby(columns).agg(
                alternation=pd.NamedAgg(column="alternation", aggfunc=lambda x: x.sum()),
                count=pd.NamedAgg(column="alternation", aggfunc=lambda x: x.count()),
                alternation_percentage=pd.NamedAgg(column="alternation", aggfunc=lambda x: x.mean())
            )
            return data_
        else:
            return data['alternation'].mean()

    def create_index_calculator(self) -> None:
        '''Create the index calculator based on the priority'''
        if self.priority == 'sklearn':
            self.create_index_sklearn()
        elif self.priority == 'mlp':
            self.create_index_mlp()
        elif self.priority == 'cnn':
            self.create_index_cnn()
        elif self.priority == 'statsmodels':
            raise NotImplementedError('Statsmodels coefficients are not implemented yet')
        else:
            raise ValueError('Priority must be sklearn or statsmodels')        
        
    def create_index_cnn(self) -> None:

        assert(self.full_data is not None)
        df = self.full_data

        categories = CherryPickEquilibria.get_categories()
        df['target'] = df['data_type'].apply(lambda x: categories.index(x)).astype("category")

        print('Preparing training data...')
        X_values, y_values = self.get_x_y_values(df)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, 
            y_values, 
            test_size=0.2,
            random_state=0, 
            stratify=y_values
        )

        clf = SimpleCNN(
            categories=categories,
            epochs=150
        )
        clf.fit(X_train, y_train)
        print("\nTest performance:")
        y_test = np.vectorize(lambda x: categories[x])(y_test)
        clf.evaluate(X_test, y_test)        

        # Save to file
        index_path = self.index_path / Path('cnn_coefficients.pt')
        torch.save(clf.model.state_dict(), index_path)

    def get_x_y_values(self, df:pd.DataFrame) -> Tuple[List]:

        def get_history_from_group(grp):
            bar_renderer = BarRenderer(data=grp)
            history = np.array(bar_renderer.get_history()).T[:, :25]
            padding = np.zeros((12 - history.shape[0], history.shape[1]))
            history = torch.tensor(np.vstack((history, padding)), dtype=torch.float32).unsqueeze(0)#.unsqueeze(0)  # Shape: (1, 1, 30, 12)
            return history

        X_values, y_values = [], []
        if 'model' in df.columns:
            group_column = 'model'
        else:
            group_column = PPT.get_group_column(df.columns)
        num_agents_column = PPT.get_num_player_column(df.columns)

        for id_sim, grp in df.groupby(group_column):

            num_agents = grp[num_agents_column].values[0]
            num_rounds = grp['round'].nunique() + 1
            history = get_history_from_group(grp)
            X_values.append(history)

            if 'target' in grp.columns:
                y_values.append(grp['target'].values[0])

        return X_values, y_values

    def create_index_mlp(self) -> None:
        assert(self.data is not None)
        df = self.data

        categories = CherryPickEquilibria.get_categories()
        df['target'] = df['data_type'].apply(lambda x: categories.index(x)).astype("category")
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df[self.measures], 
            df['target'], 
            test_size=0.2,
            random_state=0, 
            stratify=df['target']
        )

        clf = SimpleMLP(
            input_size=len(self.measures),
            hidden_size=32, 
            categories=categories,
            epochs=250
        )
        clf.fit(X_train, y_train)
        print("\nTest performance:")
        y_test = np.vectorize(lambda x: categories[x])(y_test)
        clf.evaluate(X_test, y_test)        

        # Save to file
        index_path = self.index_path / Path('mlp_coefficients.pt')
        torch.save(clf.model.state_dict(), index_path)

    def create_index_sklearn(self) -> None:
        assert(self.data is not None)
        df = self.data
        # Create target variable
        # 1 for alternation, 0 for segmentation/random
        # df['target'] = df['data_type'].apply(lambda x: 1 if x == 'alternation' else 0)
        df['target'] = df['data_type'].astype("category")
        # Split into train/test

        X_train, X_test, y_train, y_test = train_test_split(
            df[self.measures], 
            df['target'], 
            test_size=0.2,
            random_state=0, 
            stratify=df['target']
        )

        pipe = make_pipeline(
            StandardScaler(with_mean=False),        # sparse‑safe scaler
            LogisticRegression(
                penalty="l2",
                C=1.0,               # inverse of regularization strength
                solver="lbfgs",      # supports multinomial loss
                multi_class="multinomial",
                max_iter=1_000,
                n_jobs=-1
            )
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # # Fit logistic regression
        # clf = LogisticRegression()
        # clf.fit(X_train, y_train)
        # # Predict and evaluate
        # y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        # Save to file
        index_path = self.index_path / Path('sklearn_coefficients.pkl')
        joblib.dump(pipe, index_path)
        if self.debug:
            dict_measures = {measure: pipe.named_steps['logisticregression'].coef_[:, i] for i, measure in enumerate(self.measures)}
            dict_measures['intercept'] = pipe.named_steps['logisticregression'].intercept_.tolist()
            dict_measures['target'] = df['target'].cat.categories.tolist()
            df_index = pd.DataFrame().from_dict(dict_measures, orient='index').T
            df_index.index = df_index['target']
            df_index = df_index.drop(columns=['target'])
            print(df_index)
            print('Saved sklearn coefficients to', index_path)
    
    def create_index_statsmodels(self) -> Dict[str, float]:
        assert(self.data is not None)
        df = self.data
        # Create target variable
        # 1 for alternation, 0 for segmentation/random
        # df['target'] = df['data_type'].apply(lambda x: 1 if x == 'alternation' else 0)
        df['target'] = df['data_type'].astype("category")
        # Fit logistic regression
        X = sm.add_constant(df[self.measures])
        y = df['target']
        model = sm.MNLogit(y, X)
        result = model.fit(method='newton', disp=0)
        # model = sm.Logit(y, X)
        # result = model.fit(disp=0)
        print(result.summary())
        df_index = pd.DataFrame(result.params).T
        self.statsmodels_coefficients = df_index
        # Save to file
        index_path = self.index_path / Path('statsmodels_coefficients.csv')
        df_index.to_csv(index_path, index=False)
        if self.debug:
            print('Saved statsmodels coefficients to', index_path)
        return df_index

    def simulate_data(self) -> pd.DataFrame:        
        data_types = ['alternation', 'segmentation', 'mixed', 'random']
        df_full_list = list()
        df_list = list()
        for data_type in data_types:
            df, df_full = self.simulate_data_kind(data_type)
            df['data_type'] = data_type
            df_list.append(df)
            df_full['data_type'] = data_type
            df_full_list.append(df_full)
        df = pd.concat(df_list, ignore_index=True)
        df['num_agents'] = df['num_agents'].astype(int)
        df['threshold'] = df['threshold'].astype(float)
        self.data = df
        df_full = pd.concat(df_full_list, ignore_index=True)
        df_full['num_agents'] = df_full['num_agents'].astype(int)
        df_full['threshold'] = df_full['threshold'].astype(float)
        self.full_data = df_full
        return df
       
    def simulate_data_kind(self, data_type:str) -> Tuple[pd.DataFrame]:
        assert(data_type in ['segmentation', 'alternation', 'mixed', 'random'])
        num_episodes = deepcopy(self.num_episodes)
        df_full_list = list()
        df_list = list()
        for num_agents, threshold, epsilon in tqdm(self.configuration_points, desc=f'Running configurations for {data_type}'):
            B = int(num_agents * threshold)
            if data_type == 'mixed' and (num_agents == 2 or B == 1):
                # Skip mixed simulation
                continue
            eq_generator = CherryPickEquilibria(
                num_agents=int(num_agents),
                threshold=threshold,
                num_rounds=self.num_rounds,
                epsilon=epsilon,
                num_episodes=num_episodes,
                seed=self.seed,
                fancy_2P=self.fancy_2P
            )
            eq_generator.debug = False
            df_alternation = eq_generator.generate_data(data_type)
            df_full_list.append(df_alternation)
            get_m = GetMeasurements(
                data=df_alternation, 
                measures=self.measures,
                normalize=False,
            )
            df = get_m.get_measurements() 
            df['epsilon'] = epsilon
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)
        df_full = pd.concat(df_full_list, ignore_index=True)
        return df, df_full
    
    @staticmethod
    def from_file(priority:Optional[str]='sklearn'):
        '''Load the index from file'''
        index = AlternationIndex()
        index_path = PATHS['index_path']
        if priority == 'sklearn':
            index_path = index_path / Path('sklearn_coefficients.pkl')
            index.model = joblib.load(index_path)
        elif priority == 'mlp':
            index_path = index_path / Path('mlp_coefficients.pt')
            categories = CherryPickEquilibria.get_categories()
            if torch.cuda.is_available():
                index.model = SimpleMLP(
                    input_size=len(index.measures),
                    hidden_size=32,
                    categories=categories, 
                )
                index.model.model.load_state_dict(torch.load(index_path))
            else:
                index.model = SimpleMLP(
                    input_size=len(index.measures),
                    hidden_size=32, 
                    categories=categories, 
                    device="cpu"
                )
                state_dict = torch.load(index_path, map_location="cpu")
                index.model.model.load_state_dict(state_dict, strict=True)
        elif priority == 'cnn':
            index_path = index_path / Path('cnn_coefficients.pt')
            categories = CherryPickEquilibria.get_categories()
            if torch.cuda.is_available():
                index.model = SimpleCNN(
                    categories=categories, 
                )
                index.model.model.load_state_dict(torch.load(index_path))
            else:
                index.model = SimpleCNN(
                    categories=categories, 
                    device="cpu"
                )
                state_dict = torch.load(index_path, map_location="cpu")
                index.model.model.load_state_dict(state_dict, strict=True)
        elif priority == 'statsmodels':
            raise NotImplementedError('Statsmodels coefficients are not implemented yet')
            # index_path = index_path / Path('statsmodels_coefficients.csv')
            # df = pd.read_csv(index_path)
            # index.statsmodels_coefficients = df['coefficient'].values
        else:
            raise ValueError('Priority must be sklearn or statsmodels')
        index.priority = priority
        return index
    
    @staticmethod
    def complete_measures(measures: List[str]) -> List[str]:
        dict_check = AlternationIndex.check_alternation_index_in_measures(measures)
        return dict_check['measures']

    @staticmethod
    def check_alternation_index_in_measures(measures: List[str]) -> Dict[str, any]:
        measures_ = deepcopy(measures)
        if 'alternation_index' in measures_:
            index = measures_.index('alternation_index')
            measures_.pop(index)
            # measures_ += ['bounded_efficiency', 'inequality', 'entropy', 'conditional_entropy']
            measures_ += ['bounded_efficiency', 'inequality']
            measures_ = list(set(measures_))
            check = True
        else:
            check = False
        dict_check = {
            'measures': measures_,
            'check': check 
        }
        return dict_check
    
    def create_configurations(self, beta: float = 1.0, gamma: float = 1.0):
        """
        Create biased configurations:
        - Smaller num_agents are more likely (p(n) ∝ 1 / n^beta).
        - B is more likely at the extremes (p(B|n) ∝ min(B, n-B)^(-gamma)).
        Uses self.rng (numpy.random.Generator) and returns up to self.num_points unique triplets:
            (num_agents, B/num_agents, epsilon)
        """
        if self.max_agents < 2:
            raise ValueError("max_agents must be at least 2")

        # Epsilon values as before (uniform grid); you can bias these too if desired.
        eps_values = np.linspace(0.0, self.max_epsilon, 10)

        # Distribution over number of agents (favor small n)
        n_vals = np.arange(2, self.max_agents + 1)
        n_weights = 1.0 / (n_vals ** beta)
        n_probs = n_weights / n_weights.sum()

        configurations = set()  # to avoid repetitions
        # Cap attempts to avoid an endless loop if the space is small
        max_attempts = max(10 * self.num_points, 1000)
        attempts = 0

        while len(configurations) < self.num_points and attempts < max_attempts:
            attempts += 1

            # Sample num_agents with bias towards small values
            n = int(self.rng.choice(n_vals, p=n_probs))

            # Sample epsilon uniformly from grid (can also bias if needed)
            epsilon = float(self.rng.choice(eps_values))

            # Build U-shaped distribution over B in {1,...,n-1}
            if n > 1:
                Bs = np.arange(1, n)  # 1..n-1
                # Heavier weight near extremes (1 or n-1)
                min_side = np.minimum(Bs, n - Bs).astype(float)
                # Ensure no division-by-zero (min_side >= 1 in this range)
                B_weights = 1.0 / (min_side ** gamma)
                B_probs = B_weights / B_weights.sum()
                B = int(self.rng.choice(Bs, p=B_probs))
            else:
                # Degenerate (shouldn't happen since n>=2), but keep safe default
                B = 1

            triplet = (n, B / n, epsilon)
            configurations.add(triplet)

        configurations = list(configurations)
        if len(configurations) < self.num_points:
            print("Warning: Not enough unique configurations; returning all generated.")

        return configurations
