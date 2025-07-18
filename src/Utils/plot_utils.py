'''
Helper functions to gather and process data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import (
    List, 
    Union, 
    Optional, 
    Dict, 
    Tuple
)
from seaborn import (
    lineplot, 
    swarmplot, 
    barplot, 
    histplot, 
    boxplot,
    violinplot,
    kdeplot,
    scatterplot,
    heatmap
)

from Utils.utils import (
    PPT,
    OrderStrings, 
    ConditionalEntropy,
    PathUtils,
    GetMeasurements,
    Grid
)        
from Utils.indices import AlternationIndex


class PlotStandardMeasures :
    '''
    Plots standard measures
    '''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    regular_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
    ]
    standard_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
        'alternation_index'
    ]
    
    def __init__(self, data:pd.DataFrame) -> None:
        '''
        Input:
            - data, pandas dataframe
        '''
        self.data = data

    def plot_measures(
                self, 
                measures: List[str], 
                folder: Union[None, Path],
                kwargs: Optional[Union[Dict[str, str], None]]=None,
                categorical: Optional[bool]=False,
                suffix: Optional[Union[None, str]]=None
            ) -> List[Path]:
        # Tid up suffix
        if suffix is None:
            suffix = ''
        else:
            suffix = '_' + suffix
        # Tidy kwargs
        if kwargs is None:
            kwargs = dict()
        # Determine the number of rounds to plot
        T = kwargs.get('T', 20)
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        elif 'model_names' in kwargs.keys():
            assert(isinstance(kwargs['model_names'], dict))
            try:
                self.data.model = self.data.model.map(kwargs['model_names'])
            except:
                print("Warning: applying model names from kwargs didn't work.")
        try:
            self.data['model'] = self.data['model'].astype(int)
        except:
            try:
                self.data['model'] = self.data['model'].astype(float)
            except:
                pass
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        kwargs['num_models'] = num_models
        kwargs['vs_models'] = vs_models
        # Obtain data
        data = self.get_data(measures, T)
        # Initialize output list
        list_of_paths = list()
        # Plot per measure
        for m in measures:
            if folder is not None:
                file_ = PathUtils.add_file_name(folder, f'{m}{suffix}', self.extension)
            else:
                file_ = None
            print(f'Plotting {m}...')
            kwargs_ = kwargs.copy()
            if 'title' not in kwargs_.keys():
                kwargs_['title'] = m[0].upper() + m[1:]
            self.plot(
                measure=m, 
                data=data,
                kwargs=kwargs_,
                categorical=categorical,
                file=file_
            )
            if folder is not None:
                list_of_paths.append(file_)
        return list_of_paths	

    def plot(
                self, 
                measure: str,
                data: pd.DataFrame,
                kwargs: Dict[str,any],
                categorical: Optional[bool]=False,
                file: Optional[Union[Path, None]]=None
            ) -> Union[plt.axis, None]:
        '''
        Plots the variable against the models.
        Input:
            - kwargs: dict with additional setup values for plots
            - file, path of the file to save the plot on.
        Output:
            - None.
        '''
        assert(measure in self.standard_measures), f'Measure {measure} cannot be ploted by this class.'
        num_models = kwargs['num_models']
        vs_models = kwargs['vs_models']
        # Create the plot canvas
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (self.width * num_models, self.height)
        fig, ax = plt.subplots(
            figsize=figsize,
            tight_layout=True
        )
        variable = measure
        if 'with_treatment' in kwargs.keys() and kwargs['with_treatment']:
            if 'treatment' in data.columns:
                hue = 'treatment'
            else:
                hue = None
        else:
            hue = None
        if vs_models:
            if not categorical:
                lineplot(
                    x='model', y=variable, 
                    data=data, ax=ax, 
                    marker='o',
                    errorbar=('ci', 95)
                )
            else:
                boxplot(
                    x='model', y=variable, 
                    hue=hue,
                    data=data, ax=ax, 
                )
            ax.set_xlabel('Model')
            ax.set_ylabel(variable)
            # ax.set_ylim([-1.1, 1.1])
        else:
            histplot(
                data[variable],
                ax=ax
            )
            ax.set_xlabel(variable)
            # ax.set_xlim([-1.1, 1.1])
            ax.set_ylabel('Num. of episodes')
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save or return plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
            plt.close()
        else:
            print(f'Warning: No plot saved by plot_{measure}. To save plot, provide file name.')
            return ax

    @staticmethod
    def _customize_ax(
                ax:plt.axis, 
                kwargs:Dict[str,any]
            ) -> plt.axis:
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        return ax

    def plot_sweep2(
                self, 
                parameter1:str,
                parameter2:str,
                measure:str,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average measure according to sweep of two parameters.
        Input:
            - parameter1, string with the first parameter name.
            - parameter2, string with the first parameter name.
            - measure, string with the measure.
            - T, integer with the length of the tail sequence.
            - file, string with the name of file to save the plot on.
            - kwargs, dictionary with extra tweaks for the plots.
        Output:
            - None.
        '''
        if T is None:
            T = 20
        annot = kwargs.get('annot', False)
        # If measure is alternation index, need to get more measures
        to_measure = AlternationIndex.complete_measures([measure])
        # Obtain data
        get_meas = GetMeasurements(
            self.data, measures=to_measure, T=T)
        get_meas.columns += [parameter2, parameter1]
        df = get_meas.get_measurements()
        if measure == 'alternation_index':
            index_gen = AlternationIndex.from_file()
            df['alternation_index'] = index_gen(df)
        df = df.groupby([parameter2, parameter1])[measure].mean().reset_index()
        values1 = df[parameter1].unique()
        values2 = df[parameter2].unique()
        df = pd.pivot(
            data=df,
            index=[parameter1],
            values=[measure],
            columns=[parameter2]
        ).reset_index().to_numpy()[:,1:]
        # Plotting...
        fig, ax = plt.subplots(figsize=(6,6))
        heatmap(data=df, ax=ax, annot=True)
        ax.set_xticklabels(np.round(values2, 2))
        ax.set_xlabel(parameter2)
        ax.set_yticklabels(np.round(values1, 2))
        ax.set_ylabel(parameter1)
        ax.set_title(f'Av. {measure}')
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        plt.close()

    def get_data(self, measures:List[str], T:int) -> pd.DataFrame:
        # Check if alternation index is in measures
        ai_dict = AlternationIndex.check_alternation_index_in_measures(measures)
        # Get other measures
        get_meas = GetMeasurements(
            self.data, 
            measures=ai_dict['measures'], 
            T=T
        )
        data = get_meas.get_measurements()
        ordered_models = OrderStrings.dict_as_numeric(data['model'].unique())
        data['model'] = data['model'].map(ordered_models)
        data.sort_values(by='model', inplace=True)
        # Add alternation index
        if ai_dict['check']:
            ai = AlternationIndex.from_file(priority='statsmodels')
            data['alternation_index'] = ai(data)
        return data


class PlotRoundMeasures(PlotStandardMeasures):
    '''
    Plot measures per round
    '''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    round_measures = [
        'round_efficiency',
        'round_conditional_entropy'
    ]

    def __init__(self, data:pd.DataFrame) -> None:
        '''
        Input:
            - data, pandas dataframe
        '''
        super().__init__(data)

    def get_data(self, measures, T):
        get_meas = GetMeasurements(
            self.data, measures=measures, T=T, per_round=True
        )
        data = get_meas.get_measurements()
        ordered_models = OrderStrings.dict_as_numeric(data['model'].unique())
        data['model'] = data['model'].map(ordered_models)
        data.sort_values(by='model', inplace=True)
        return data

    def plot(
                self, 
                measure: str,
                data: pd.DataFrame,
                kwargs: Dict[str,any],
                categorical: Optional[bool]=False,
                file: Optional[Union[Path, None]]=None
            ) -> Union[plt.axis, None]:
        '''
        Plots the variable vs round per model.
        Input:
            - measure: str with the name of the measure to plot.
            - data: pandas dataframe with the data.
            - kwargs: dict with additional setup values for plots
            - file, path of the file to save the plot on.
        Output:
            - None or plt.axis.
        '''
        assert(measure in self.round_measures), f'Measure {measure} cannot be ploted by this class.'
        num_models = kwargs['num_models']
        vs_models = kwargs['vs_models']
        # Create the plot canvas
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (self.width * num_models, self.height)
        fig, ax = plt.subplots(
            figsize=figsize,
            tight_layout=True
        )
        variable = self.get_variable_from_measure(measure)
        if vs_models:
            lineplot(
                x='round', y=variable, 
                hue='model',
                data=data, ax=ax, 
                errorbar=('ci', 95)
            )
        else:
            lineplot(
                x='round', y=variable, 
                data=data, ax=ax, 
                errorbar=('ci', 95)
            )
        ax.set_xlabel('Round')
        ax.set_ylabel(variable)
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save or return plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
            plt.close()
        else:
            print('Warning: No plot saved by plot_efficiency. To save plot, provide file name.')
            return ax
        
    def get_variable_from_measure(self, measure:str) -> str:
        if measure == 'round_efficiency':
            return 'efficiency'
        if measure == 'round_conditional_entropy':
            return 'conditional_entropy'


class PlotVSMeasures:
    '''Plot 2D scatter plots'''
    dpi = 300
    extension = 'png'
    width = 3
    height = 3.5
    cmaps = ["Blues", "Reds", "Greens", "Yellows"]
    standard_measures = [
        'attendance',
        'efficiency', 
        'inequality',
        'entropy',
        'conditional_entropy',
        'alternation_index'
    ]

    def __init__(self, data:pd.DataFrame) -> None:
        self.data = data
        self.warnings = True

    def two_way_comparisons(
                self, 
                measure_pairs:List[str],
                file:Path,
                kwargs:Optional[Dict[str,any]]={}
            ) -> Union[None, plt.axes]:
        vertical = kwargs.get('vertical', True)
        grid = Grid(len(measure_pairs), vertical)
        fig, axes = plt.subplots(
            grid.rows, grid.cols,
            figsize=(self.width*grid.cols, self.height*grid.rows)
        )
        kwargs_ = {
            'x_label':None,
            'y_label':None
        }
        kwargs.update(kwargs_)
        info_list = list()
        measures = list()
        for idx, pair_idx in enumerate(grid):
            if grid.length > 1:
                ax = axes[pair_idx]
            else:
                ax = axes
            # Get pair of measures to plot
            pair_measures = measure_pairs[idx]
            # Add measures to list
            if pair_measures[0] not in measures:
                measures.append(pair_measures[0])
            if pair_measures[1] not in measures:
                measures.append(pair_measures[1])
            # Get plot and plot's info
            info = self.plot_vs(
                pair_measures=pair_measures,
                ax=ax,
                file=None, kwargs=kwargs
            )
            info_list.append(info)
        # Get max and min values for each measure
        dict_max_min = dict()
        for measure in measures:
            min_m = min([
                info[measure]['min'] for info in info_list if measure in info.keys()
            ])
            max_m = max([
                info[measure]['max'] for info in info_list if measure in info.keys()
            ])
            min_m = min_m - 0.1*(max_m - min_m)
            max_m = max_m + 0.1*(max_m - min_m)
            dict_max_min[measure] = [min_m, max_m]
        # Customize axes
        for idx, pair_idx in enumerate(grid):
            if grid.length > 1:
                ax = axes[pair_idx]
            else:
                ax = axes
            pair_measures = measure_pairs[idx]
            ax.set_xlabel(pair_measures[0])
            ax.set_ylabel(pair_measures[1])
            ax.set_xlim(dict_max_min[pair_measures[0]])
            ax.set_ylim(dict_max_min[pair_measures[1]])
            if idx == grid.length - 1:
                # Get legend handles/labels from the last axes
                handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
        # Place the legend below all subplots
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
        # Adjust spacing to accommodate legend
        fig.subplots_adjust(bottom=0.15)  # Add space for legend
        plt.tight_layout()
        # Save plot
        plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        print('Plot saved to', file)

    def plot_vs(
                self,
                pair_measures:List[str],
                ax:Optional[Union[plt.axes, None]]=None,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> Union[None, plt.axes]:
        # Determine the number of rounds to plot
        T = kwargs.get('T', 20)
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        # Record measure names
        measure1, measure2 = pair_measures
        assert(measure1 in self.standard_measures)
        assert(measure2 in self.standard_measures)
        measures = [measure1, measure2]
        # Measure on the given measures
        # gm = GetMeasurements(self.data, measures, T=T)
        # df_measures = gm.get_measurements()
        df_measures = self.get_data(measures, T)
        # Jitter measures for better display
        sigma = 0.01
        df_measures[measure1] += np.random.normal(0,sigma, len(df_measures[measure1]))
        df_measures[measure2] += np.random.normal(0,sigma, len(df_measures[measure2]))
        # Save extremes
        info = dict()
        for measure in measures:
            info[measure] = dict()
            info[measure]['min'] = df_measures[measure].min()
            info[measure]['max'] = df_measures[measure].max()
        # Create the plot canvas
        if ax is None:
            fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel(f'{measure1[0].upper()}{measure1[1:]}')
        ax.set_ylabel(f'{measure2[0].upper()}{measure2[1:]}')
        # Plotting...
        if vs_models:
            # kdeplot(
            #     data=df_measures,
            #     x=measure1,
            #     y=measure2,
            #     hue='model',
            #     cmap=self.cmaps[:num_models], 
            #     fill=True,
            #     ax=ax,
            # )
            scatterplot(
                data=df_measures,
                x=measure1,
                y=measure2,
                hue='model',
                style='model',
                legend=True,
                ax=ax
            )
        else:
            # kdeplot(
            #     data=df_measures,
            #     x=measure1,
            #     y=measure2,
            #     cmap=self.cmaps[0],
            #     fill=True,
            #     ax=ax
            # )
            scatterplot(
                data=df_measures,
                x=measure1,
                y=measure2,
                ax=ax
            )
        # Set further information on plot
        ax = PlotStandardMeasures._customize_ax(ax, kwargs)
        # Save plot
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            if self.warnings:
                print('Warning: No plot saved. To save plot, provide file name.')
            # fig.canvas.draw()
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to (H, W, 3)
            # return image
        return info

    def get_data(self, measures:List[str], T:int) -> pd.DataFrame:
        # Check if alternation index is in measures
        ai_dict = AlternationIndex.check_alternation_index_in_measures(measures)
        # Get other measures
        get_meas = GetMeasurements(
            self.data, 
            measures=ai_dict['measures'], 
            T=T
        )
        data = get_meas.get_measurements()
        ordered_models = OrderStrings.dict_as_numeric(data['model'].unique())
        data['model'] = data['model'].map(ordered_models)
        data.sort_values(by='model', inplace=True)
        # Add alternation index
        if ai_dict['check']:
            ai = AlternationIndex.from_file(priority='statsmodels')
            data['alternation_index'] = ai(data)
        return data


class PlotsAndMeasures :
    '''
    Plots frequently used visualizations.
    '''

    def __init__(self, data:pd.DataFrame) -> None:
        '''
        Input:
            - data, pandas dataframe
        '''
        assert('model' in data.columns)
        self.data = data
        self.dpi = 300
        self.extension = 'pdf'
        self.width = 3
        self.height = 3.5
        self.cmaps = ["Blues", "Reds", "Greens", "Yellows"]

    def plot_measures(
                self, 
                measures: List[str], 
                folder: Union[None, Path],
                kwargs: Optional[Union[Dict[str, str], None]]=None,
                suffix: Optional[Union[None, str]]=None
            ) -> List[Path]:
        '''
        DOCUMENTATION MISSING
        '''
        standard_measures = [
            'attendance', 
            'deviation', 
            'efficiency', 
            'inequality',
            'conditional_entropy',
            'entropy',
            'hist_states',
            'hist_state_transitions'
        ]
        if kwargs is None:
            kwargs = dict()
        if suffix is None:
            suffix = ''
        else:
            suffix = '_' + suffix
        list_of_paths = list()
        if 'T' in kwargs.keys():
            T = kwargs['T']
        else:
            T = 20
        for m in measures:
            if folder is not None:
                file_ = PathUtils.add_file_name(folder, f'{m}{suffix}', self.extension)
            print(f'Plotting {m}...')
            if m in standard_measures:
                kwargs_ = kwargs.copy()
                if 'title' not in kwargs_.keys():
                    kwargs_['title'] = m[0].upper() + m[1:]
                instruccion = f'self.plot_{m}(file=file_, T=T, kwargs=kwargs_)'
                exec(instruccion)
            if m == 'convergence':
                kwargs['title'] = m[0].upper() + m[1:]
                self.plot_convergence(
                    T = T,
                    file=file_, 
                    kwargs=kwargs
                )
            elif m == 'round_attendance':
                if folder is not None:
                    file_ = PathUtils.add_file_name(folder, f'round_attendance', self.extension)
                else:
                    file_ = None
                self.plot_round_attendance(file=file_, kwargs=kwargs)
            elif m == 'round_efficiency':
                self.plot_round_efficiency(file=file_)
            elif m == 'score':
                if folder is not None:
                    file_ = folder / Path(f'scores.{self.extension}')
                else:
                    file_ = None
                self.plot_scores(file=file_, kwargs=kwargs)
            elif m == 'eq_coop':
                if folder is not None:
                    file_ = folder / Path(f'eq_coop.{self.extension}')
                else:
                    file_ = None
                mu = self.agents[0].threshold
                self.plot_EQ(mu=mu, file=file_, kwargs=kwargs)
            list_of_paths.append(file_)
        return list_of_paths
        
    def plot_round_attendance(
                self, 
                T: Optional[int]=np.inf,
                file: Optional[Union[str, None]]=None,
                kwargs: Optional[Union[Dict[str, any], None]]=None
            ) -> plt.axis:
        '''
        Plots the average attendance per round.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Proportion of going')
        ax.set_ylim([-0.1, 1.1])
        ax.grid()
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        group_column = PPT.get_group_column(self.data.columns)
        columns = ['model', group_column, 'round']
        decision_column = PPT.get_decision_column(self.data.columns)
        data = pd.DataFrame(data.groupby(columns)[decision_column].mean().reset_index())
        if vs_models:
            ax = lineplot(x='round', y=decision_column, hue='model', data=data)
        else:
            ax = lineplot(x='round', y=decision_column, data=data)
        # Set information on plot
        if kwargs is not None:
            if 'title' in kwargs.keys():
                ax.set_title(kwargs['title'])			
            if 'title_size' in kwargs.keys():
                ax.title.set_size(kwargs['title_size'])
            if 'x_label_size' in kwargs.keys():
                ax.xaxis.label.set_size(kwargs['x_label_size'])
            if 'y_label_size' in kwargs.keys():
                ax.yaxis.label.set_size(kwargs['y_label_size'])		
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig

    def plot_round_efficiency(
                self, 
                T:Optional[int]=np.inf,
                file:Optional[Union[str, None]]=None
            ) -> plt.axis:
        '''
        Plots the average score per round.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Av. score')
        ax.set_ylim([-1.1, 1.1])
        ax.grid()
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        columns = ['model', 'id_sim', 'round']
        data = pd.DataFrame(data.groupby(columns)['score'].mean())
        if vs_models:
            ax = lineplot(x='round', y='score', hue='model', data=data)
        else:
            ax = lineplot(x='round', y='score', data=data)
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig
    
    def plot_round_inequality(
                self, 
                T:Optional[int]=np.inf,
                file:Optional[Union[str, None]]=None
            ) -> plt.axis:
        '''
        Plots the std of the scores per round.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Std. score')
        ax.grid()
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        columns = ['model', 'id_sim', 'round']
        data = pd.DataFrame(data.groupby(columns)['score'].std())
        if vs_models:
            ax = lineplot(x='round', y='score', hue='model', data=data)
        else:
            ax = lineplot(x='round', y='score', data=data)
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig

    def plot_round_convergence(
                self, 
                T:Optional[int]=np.inf,
                file:Optional[Union[str, None]]=None
            ) -> plt.axis:
        '''
        Plots the average score per round.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Round')
        ax.set_ylabel('Convergence')
        ax.grid()
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        if vs_models:
            ax = lineplot(x='round', y='convergence', hue='model', data=data)
        else:
            ax = lineplot(x='round', y='convergence', data=data)
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig

    def plot_attendance(
                self, 
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average attendance in the last T rounds, averaged over simulations.
        Input:
            - T: int with the number of last T rounds.
            - file, path of the file to save the plot on.
            - kwargs: dict with additional setup values for plots
        Output:
            - None.
        '''
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get only last T rounds from data
        num_rounds = max(self.data["round"].unique())
        if 'T' in kwargs.keys():
            T = kwargs['T']
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        # Average by model and episode
        group_column = PPT.get_group_column(data.columns)
        decision_column = PPT.get_decision_column(data.columns)
        columns = ['model', group_column, 'round']
        data = pd.DataFrame(data.groupby(columns)[decision_column].mean().reset_index())
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        # Plot according to number of models
        if vs_models:
            # boxplot(x='model', y='decision', data=data, ax=ax)
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y=decision_column, data=data, ax=ax, order=order)
            swarmplot(x='model', y=decision_column, data=data, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Attendance')
            # ax.set_ylim([-0.1, 1.1])
        else:
            raise Exception('Ooops!!!!!')
            # histplot(data[decision_column], ax=ax)
            # ax.set_xlabel('Attendance')
            # # ax.set_xlim([-0.1, 1.1])
            # ax.set_ylabel('Num. of groups')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_attendance. To save plot, provide file name.')

    def plot_deviation(
                self, 
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the std of the average attendances per round.
        Input:
            - T: int with the number of last T rounds.
            - file, path of the file to save the plot on.
            - kwargs: dict with additional setup values for plots
        Output:
            - None.
        '''
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        group_column = PPT.get_group_column(self.data.columns)
        decision_column = PPT.get_decision_column(self.data.columns)
        sims = self.data[group_column].unique()
        vs_sims = True if len(sims) > 1 else False
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        # Get only last T rounds from data
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        # Average by model, episode and round (get bar's attendance in a round)
        columns = ['model', group_column, 'round']
        data = pd.DataFrame(data.groupby(columns)[decision_column].mean().reset_index())
        # Get std of round attendance
        columns = ['model', group_column]
        data = pd.DataFrame(data.groupby(columns)[decision_column].std().reset_index())
        if vs_models:
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y=decision_column, data=data, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Deviation')
        else:
            histplot(data[decision_column], ax=ax)
            ax.set_xlabel('Deviation')
            ax.set_ylabel('Num. of groups')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_deviation. To save plot, provide file name.')

    def plot_efficiency(
                self, 
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average score over all players over all last T rounds.
        Input:
            - T: int with the number of last T rounds.
            - file, path of the file to save the plot on.
            - kwargs: dict with additional setup values for plots
        Output:
            - None.
        '''
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        group_column = PPT.get_group_column(self.data.columns)
        sims = self.data[group_column].unique()
        vs_sims = True if len(sims) > 1 else False
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        # Get only last T rounds from data
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        # Get average score per model and episode
        player_column = PPT.get_player_column(self.data.columns)
        columns = ['model', group_column, player_column]
        data = pd.DataFrame(data.groupby(columns)['score'].mean().reset_index())
        if vs_models:
            # boxplot(x='model', y='score', data=data, ax=ax)
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y='score', data=data, ax=ax, order=order)
            swarmplot(x='model', y='score', data=data, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Av. score')
            # ax.set_ylim([-1.1, 1.1])
        else:
            histplot(data['score'], ax=ax)
            ax.set_xlabel('Av. score')
            # ax.set_xlim([-1.1, 1.1])
            ax.set_ylabel('Num. of players')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_efficiency. To save plot, provide file name.')

    def plot_inequality(
                self, 
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the std of the average score per player over all last T rounds.
        Input:
            - T: int with the number of last T rounds.
            - file, path of the file to save the plot on.
            - kwargs: dict with additional setup values for plots
        Output:
            - None.
        '''
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        group_column = PPT.get_group_column(self.data.columns)
        player_column = PPT.get_player_column(self.data.columns)
        sims = self.data[group_column].unique()
        vs_sims = True if len(sims) > 1 else False
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        # Get only last T rounds from data
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        # Get average score per model, episode and player
        columns = ['model', group_column, player_column]
        data = pd.DataFrame(data.groupby(columns)['score'].mean().reset_index())
        # Get std of average score per player
        columns = ['model', group_column]
        data = pd.DataFrame(data.groupby(columns)['score'].std().reset_index())
        if vs_models:
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y='score', data=data, ax=ax, order=order)
            swarmplot(x='model', y='score', data=data, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Std. of av. score per player')
        else:
            histplot(data['score'], ax=ax)
            ax.set_xlabel('Std. of av. score per player')
            ax.set_ylabel('Num. of groups')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_inequality. To save plot, provide file name.')

    def plot_conditional_entropy(
                self,
                file:Optional[Union[Path, None]]=None,
                T:Optional[int]=20,
                kwargs:Optional[Dict[str,any]]={},
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Calculate conditional entropy for each model
        if vs_models:
            df_list = list()
            for key, grp in self.data.groupby(['model']):
                model = grp.model.unique()[0]
                ce = ConditionalEntropy(
                    data=grp,
                    T=T
                )
                conditional_entropy = ce.get_contidional_entropy()
                df = pd.DataFrame({
                    'model':[model]*len(conditional_entropy), 
                    'ce': conditional_entropy
                })
                df_list.append(df)
            df = pd.concat(df_list, ignore_index=True)
        else:
            ce = ConditionalEntropy(
                data=self.data,
                T=T
            )
            conditional_entropy = ce.get_contidional_entropy()
            model = models[0]
            df = pd.DataFrame({
                'model':[model]*len(conditional_entropy), 
                'ce': conditional_entropy
            })
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        if vs_models:
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y='ce', data=df, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Conditional entropy')
        else:
            histplot(df['ce'], ax=ax)
            ax.set_xlabel('Conditional entropy')
            ax.set_ylabel('Num. of groups')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_conditional_entropy. To save plot, provide file name.')

    def plot_entropy(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Calculate conditional entropy for each model
        if vs_models:
            df_list = list()
            for key, grp in self.data.groupby(['model']):
                model = grp.model.unique()[0]
                ce = ConditionalEntropy(
                    data=grp,
                    T = T
                )
                entropy = ce.get_entropy()
                df = pd.DataFrame({
                    'model':[model]*len(entropy), 
                    'ce': entropy
                })
                df_list.append(df)
            df = pd.concat(df_list, ignore_index=True)
        else:
            ce = ConditionalEntropy(
                data=self.data,
                T = T
            )
            entropy = ce.get_entropy()
            model = models[0]
            df = pd.DataFrame({
                'model':[model]*len(entropy), 
                'ce': entropy
            })
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        if vs_models:
            order = OrderStrings.order_as_float(models)
            violinplot(x='model', y='ce', data=df, ax=ax, order=order)
            swarmplot(x='model', y='ce', data=df, ax=ax, order=order)
            ax.set_xlabel('Model')
            ax.set_ylabel('Entropy')
        else:
            histplot(df['ce'], ax=ax)
            ax.set_xlabel('Entropy')
            ax.set_ylabel('Num. of groups')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_entropy. To save plot, provide file name.')

    def plot_convergence(
                self, 
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average convergence over all players over all last T rounds.
        Input:
            - T: int with the number of last T rounds.
            - file, path of the file to save the plot on.
            - kwargs: dict with additional setup values for plots
        Output:
            - None.
        '''
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        sims = self.data.id_sim.unique()
        vs_sims = True if len(sims) > 1 else False
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        # Get only last T rounds from data
        num_rounds = max(self.data["round"].unique())
        data = pd.DataFrame(self.data[self.data["round"] >= num_rounds - T])
        # Get average score per model and episode
        columns = ['model', 'id_sim']
        data = pd.DataFrame(data.groupby(columns)['convergence'].mean().reset_index())
        if vs_models:
            order = OrderStrings.order_as_float(models)
            boxplot(x='model', y='convergence', data=data, ax=ax, order=order)
        else:
            histplot(data['convergence'], ax=ax)
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        ax.set_xlabel('Model')
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        ax.set_ylabel('Av. convergence')
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_convergence. To save plot, provide file name.')

    def plot_hist_scores(
                self, 
                mu:float, 
                file: Optional[Union[str, None]]=None,
                ax: Optional[Union[plt.axis, None]]=None,
            ) -> plt.axis:
        '''
        Plots the histogram of scores.
        Input:
            - mu, threshold defining the bar's capacity
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        df = self.data.copy()
        df = df.groupby(['model', 'id_sim', 'id_player'])['score'].mean().reset_index(name='av_score')
        if ax is None:
            fig, ax = plt.subplots(figsize=(4,3.5))
        if vs_models:
            swarmplot(x=df['av_score'], hue='model', size=3, ax=ax)
        else:
            ax = swarmplot(x=df['av_score'], size=3, ax=ax)
        ax.axvline(x=mu, color='red', label='Fair quantity')
        ax.set_xlabel('Av. score per player')
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return ax

    def plot_decisions(self, file:str=None) -> plt.figure:
        '''
        Plots the decision per round for each agent.
        Input:
            - file, string with the name of file to save the plot on.
        Output:
            - fig, a plt object, or None.
        '''
        models = self.data.model.unique()
        vs_models = True if len(models) > 1 else False
        if vs_models:
            fig, axes = plt.subplots(len(models), 1, figsize=(4*i,3.25), tight_layout=True)
            for i in range(len(models)):
                axes[i].set_xlabel('Round')
                axes[i].set_ylabel('Player\'s decision')
                axes[i].set_ylim([-0.1, 1.1])
                axes[i].grid()
                axes[i] = lineplot(x='round', y='decision', hue='model', data=self.data, ci=None)
        else:
            fig, ax = plt.subplots(figsize=(4,3.5))
            ax.set_xlabel('Round')
            ax.set_ylabel('Player\'s decision')
            ax.set_ylim([-0.1, 1.1])
            ax.grid()
            ax = lineplot(x='round', y='decision', hue='id_player', data=self.data, ci=None)
        if file is not None:
            print('Plot saved to', file)
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
        return fig
    
    def plot_hist_states(
                self,
                T: Optional[int]=20,
                ax: Optional[Union[plt.axis, None]]=None,
                file: Optional[Union[Path, None]]=None,
                kwargs: Optional[Dict[str,any]]={}
            ) -> plt.axis:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        elif 'model_names' in kwargs.keys():
            assert(isinstance(kwargs['model_names'], dict))
            self.data.model = self.data.model.map(kwargs['model_names'])
        models = self.data.model.unique()
        num_models = len(models)
        # Get only last T rounds
        M = self.data['round'].unique().max()
        data = pd.DataFrame(self.data[self.data['round'] >= (M - T)]).reset_index(drop=True)
        # Get state frequency
        list_num_states = list()
        df_list = list()
        for model, grp in data.groupby('model'):
            group_column = PPT.get_group_column(grp.columns)
            sims = grp[group_column].unique()
            assert(len(sims) == 1)
            ce = ConditionalEntropy(grp)
            tm = ce.get_group_states(grp)
            # print(model, ce.get_entropy())
            tm.normalize()
            # Create the plot canvas
            if 'crop' in kwargs.keys() and kwargs['crop']:
                df = pd.DataFrame({
                    'state':[str(x) for x in tm.data_dict.keys() if tm.data_dict[x] != 0],
                    'frequency': [x for x in tm.data_dict.values() if x != 0]
                })
            else:
                df = pd.DataFrame({
                    'state':[str(x) for x in tm.data_dict.keys()],
                    'frequency': [x for x in tm.data_dict.values()]
                })
            df['model'] = model
            df_list.append(df)
            num_states = len(df.state.unique())
            list_num_states.append(num_states)
        df = pd.concat(df_list, ignore_index=True)
        if ax is None:
            fig, ax = plt.subplots(figsize=(1/4*self.width*max(list_num_states), self.height))
        if num_models == 1:
            barplot(x='state', y='frequency', data=df, ax=ax)
        else:
            barplot(x='state', y='frequency', data=df, hue='model', ax=ax)
        ax.set_xlabel('State')
        plt.xticks(rotation=90)
        ax.set_ylabel('Relative frequency')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        if 'legend' in kwargs.keys():
            plt.legend(title=kwargs['legend'])
        ax.grid()
        ax.tick_params(axis='x', labelrotation=90)
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_hist_states. To save plot, provide file name.')
        return ax 
    
    def plot_hist_state_transitions(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        assert(len(models) == 1)
        group_column = PPT.get_group_column(self.data.columns)
        player_column = PPT.get_player_column(self.data.columns)
        sims = self.data[group_column].unique()
        assert(len(sims) == 1)
        # Get state frequency
        ce = ConditionalEntropy(
            data=self.data,
            T = T
        )
        M = self.data['round'].unique().max()
        df = pd.DataFrame(self.data[self.data['round'] >= (M - T)]).reset_index(drop=True)
        tm = ce.get_group_transitions(df)
        tm.normalize()
        if 'crop' in kwargs.keys() and kwargs['crop']:
            df = pd.DataFrame({
                'transition':[
                    str((tm.get_state_from_index(i), tm.get_state_from_index(j))) 
                    for i in range(tm.trans_freqs.shape[0]) 
                    for j in range(tm.trans_freqs.shape[1])
                    if tm.trans_freqs[i,j] != 0
                ],
                'frequency': [
                    tm.trans_freqs[i,j]
                    for i in range(tm.trans_freqs.shape[0]) 
                    for j in range(tm.trans_freqs.shape[1])
                    if tm.trans_freqs[i,j] != 0
                ]
            })
        else:
            df = pd.DataFrame({
                'transition':[
                    str((tm.get_state_from_index(i), tm.get_state_from_index(j))) 
                    for i in range(tm.trans_freqs.shape[0]) 
                    for j in range(tm.trans_freqs.shape[1])
                ],
                'frequency': [
                    tm.trans_freqs[i,j]
                    for i in range(tm.trans_freqs.shape[0]) 
                    for j in range(tm.trans_freqs.shape[1])
                ]
            })
        # Create the plot canvas
        num_transition = len(df.transition.unique())
        fig, ax = plt.subplots(figsize=(1/4*self.width*num_transition, self.height))
        barplot(x='transition', y='frequency', data=df, ax=ax)
        ax.set_xlabel('State')
        plt.xticks(rotation=90)
        ax.set_ylabel('Relative frequency')
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_hist_state_transitions. To save plot, provide file name.')

    def plot_scores_sweep2(
                self, 
                parameter1:str,
                parameter2:str,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> None:
        '''
        Plots the average scores according to sweep of two parameters.
        Input:
            - parameter1, string with the first parameter name.
            - parameter2, string with the first parameter name.
            - file, string with the name of file to save the plot on.
        Output:
            - axis, a plt object, or None.
        '''
        if T is None:
            T = 20
        annot = kwargs.get('annot', False)
        # Keep only last T of rounds
        num_rounds = self.data['round'].max()
        df = self.data[self.data['round'] > num_rounds - T].reset_index()
        # Find average score per pair of parameters' values
        df = df.groupby([parameter2, parameter1])['score'].mean().reset_index()
        values1 = df[parameter1].unique()
        values2 = df[parameter2].unique()
        df = pd.pivot(
            data=df,
            index=[parameter1],
            values=['score'],
            columns=[parameter2]
        ).reset_index().to_numpy()[:,1:]
        # Plotting...
        fig, ax = plt.subplots(figsize=(6,6))
        heatmap(data=df, ax=ax, annot=annot)
        ax.set_xticklabels(np.round(values2, 2))
        ax.set_xlabel(parameter2)
        ax.set_yticklabels(np.round(values1, 2))
        ax.set_ylabel(parameter1)
        ax.set_title('Av. Score')
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        plt.close()

    def plot_EQ(self, mu:float, file:str=None) -> plt.axis:
        '''
        Plots the equitative cooperation index.
        Input:
            - file, string with the name of file to save the plot on.
            - mu, threshold defining the bar's capacity
        Output:
            - axis, a plt object, or None.
        '''
        df = self._get_eq(mu=mu)
        models = df.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        if vs_models:
            ax = barplot(x='model', y='Eq', data=df)
        else:
            ax = barplot(df['Eq'])
        ax.set_xlabel('Model')
        ax.set_ylabel('Equitable cooperation')
        plt.xticks(rotation = 25)
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig

    def plot_EQ_scores(self, mu:float, file:str=None) -> plt.axis:
        '''
        Plots the equitative cooperation index.
        Input:
            - file, string with the name of file to save the plot on.
            - mu, threshold defining the bar's capacity
        Output:
            - axis, a plt object, or None.
        '''
        df = self._get_eq(mu=mu)
        models = df.model.unique()
        vs_models = True if len(models) > 1 else False
        fig, ax = plt.subplots(figsize=(4,3.5))
        ax.set_xlabel('Sim')
        ax.set_ylabel('EQ')
        #ax.set_ylim([-1.1, 1.1])
        ax.grid()
        if vs_models:
            ax = barplot(x='model', y='Eq', hue='model', data=df, errorbar=None)
            ax.set(xticklabels=[])
            ax.tick_params(bottom=False)
        else:
            ax = barplot(x='model', y='Eq', data=df, errorbar=None)
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            print('Plot saved to', file)
        return fig
    
    def _get_eq(self, mu:float, keep_player:bool=True) -> pd.DataFrame:
        '''
        Returns the equanimity index.
        Input:
            - mu, threshold defining the bar's capacity
            - keepdims, if true, keeps per player info 

        Output:
            - df, a dataframe with one more column from self.data,
                but results gruped by id_sim (if keep_player=False)
                or by id_player (if keep_player=True) 
        '''
        # Create copy of input dataframe
        df = self.data.copy()
        # Keep only last 20 rounds
        keep_round = df['round'].max() - 20
        df = df[df['round'] >= keep_round]
        # determine mean scores per simulaiton
        df = df.groupby(['model', 'id_sim', 'id_player'])['score'].mean().reset_index(name='av_score')
        # Determine equitable cooperation index
        if keep_player:
            df['Eq'] = df.groupby(['model', 'id_sim'])['av_score'].transform(lambda x: eq_coop(20, mu, x))
        else:
            df = df.groupby(['model', 'id_sim'])['av_score'].apply(lambda x: eq_coop(20, mu, x)).rename('Eq').reset_index()
        return df

    def plot_inequality_vs_attendance(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get inequality and attendance
        measures = ['inequality', 'attendance']
        gm = GetMeasurements(self.data, measures, T=T)
        df_measures = gm.get_measurements()
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel('Attendance')
        ax.set_ylabel('Inequality')
        if vs_models:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='inequality',
                hue='model',
                cmap=self.cmaps[:num_models], 
                fill=True,
                ax=ax,
            )
            scatterplot(
                data=df_measures,
                x='attendance',
                y='inequality',
                hue='model',
                legend=False,
                ax=ax
            )
        else:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='inequality',
                cmap=self.cmaps[0],
                fill=True,
                ax=ax
            )
            scatterplot(
                data=df_measures,
                x='attendance',
                y='inequality',
                ax=ax
            )
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_inequality_vs_attendance. To save plot, provide file name.')

    def plot_inequality_vs_efficiency(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get inequality and efficiency
        measures = ['inequality', 'efficiency']
        gm = GetMeasurements(self.data, measures, T=T)
        df_measures = gm.get_measurements()
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel('Efficiency')
        ax.set_ylabel('Inequality')
        if vs_models:
            kdeplot(
                data=df_measures,
                x='efficiency',
                y='inequality',
                hue='model',
                cmap=self.cmaps[:num_models], 
                fill=True,
                ax=ax,
            )
            scatterplot(
                data=df_measures,
                x='efficiency',
                y='inequality',
                hue='model',
                legend=False,
                ax=ax
            )
        else:
            kdeplot(
                data=df_measures,
                x='efficiency',
                y='inequality',
                cmap=self.cmaps[0],
                fill=True,
                ax=ax
            )
            scatterplot(
                data=df_measures,
                x='efficiency',
                y='inequality',
                ax=ax
            )
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_inequality_vs_efficiency. To save plot, provide file name.')

    def plot_attendance_vs_efficiency(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get attendance and efficiency
        measures = ['attendance', 'efficiency']
        gm = GetMeasurements(self.data, measures, T=T)
        df_measures = gm.get_measurements()
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel('Attendance')
        ax.set_ylabel('Efficiency')
        if vs_models:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                hue='model',
                cmap=self.cmaps[:num_models], 
                fill=True,
                ax=ax,
            )
            scatterplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                hue='model',
                legend=False,
                ax=ax
            )
        else:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                cmap=self.cmaps[0],
                fill=True,
                ax=ax
            )
            scatterplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                ax=ax
            )
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_attendance_vs_efficiency. To save plot, provide file name.')

    def plot_inequality_vs_entropy(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get inequality and entropy
        measures = ['inequality', 'entropy']
        gm = GetMeasurements(self.data, measures, T=T)
        df_measures = gm.get_measurements()
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Inequality')
        if vs_models:
            kdeplot(
                data=df_measures,
                x='entropy',
                y='inequality',
                hue='model',
                cmap=self.cmaps[:num_models], 
                fill=True,
                ax=ax,
            )
            scatterplot(
                data=df_measures,
                x='entropy',
                y='inequality',
                hue='model',
                legend=False,
                ax=ax
            )
        else:
            kdeplot(
                data=df_measures,
                x='entropy',
                y='inequality',
                cmap=self.cmaps[0],
                fill=True,
                ax=ax
            )
            scatterplot(
                data=df_measures,
                x='entropy',
                y='inequality',
                ax=ax
            )
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_inequality_vs_entropy. To save plot, provide file name.')

    def plot_attendance_vs_efficiency_per_player(
                self,
                T:Optional[int]=20,
                file:Optional[Union[Path, None]]=None,
                kwargs:Optional[Dict[str,any]]={}
            ) -> pd.DataFrame:
        # Determine the number of model in data
        if 'only_value' in kwargs.keys():
            if kwargs['only_value']:
                self.data.model = self.data.model.apply(lambda x: x.split('=')[-1])
        models = self.data.model.unique()
        num_models = len(models)
        vs_models = True if len(models) > 1 else False
        # Get attendance and efficiency
        measures = ['attendance', 'efficiency']
        gm = GetMeasurements(
            data=self.data, 
            measures=measures, 
            T=T,
            per_player=True
        )
        df_measures = gm.get_measurements()
        # Create the plot canvas
        fig, ax = plt.subplots(figsize=(self.width*num_models, self.height))
        ax.set_xlabel('Av. going')
        ax.set_ylabel('Av. score')
        if vs_models:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                hue='model',
                cmap=self.cmaps[:num_models], 
                fill=True,
                ax=ax,
            )
            scatterplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                hue='model', 
                color="gray",
                legend=False,
                ax=ax
            )
        else:
            kdeplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                cmap=self.cmaps[0],
                fill=True,
                ax=ax
            )
            group_column = PPT.get_group_column(df_measures.columns)
            scatterplot(
                data=df_measures,
                x='attendance',
                y='efficiency',
                color="gray",
                style=group_column,
                legend=False,
                ax=ax
            )
        # Set information on plot
        if 'title' in kwargs.keys():
            ax.set_title(kwargs['title'])			
        if 'title_size' in kwargs.keys():
            ax.title.set_size(kwargs['title_size'])
        if 'x_label' in kwargs.keys():
            ax.set_xlabel(kwargs['x_label'])
        if 'x_label_size' in kwargs.keys():
            ax.xaxis.label.set_size(kwargs['x_label_size'])
        if 'y_label_size' in kwargs.keys():
            ax.yaxis.label.set_size(kwargs['y_label_size'])
        ax.grid()
        if file is not None:
            plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            print('Plot saved to', file)
        else:
            print('Warning: No plot saved by plot_attendance_vs_efficiency_per_player. To save plot, provide file name.')


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
        