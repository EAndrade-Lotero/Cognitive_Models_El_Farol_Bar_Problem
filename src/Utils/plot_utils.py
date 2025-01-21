'''
Helper functions to gather and process data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from typing import List, Union, Optional, Dict
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
	GetMeasures
)



class PlotsAndMeasures :
	'''
	Plots frequently used visualizations.
	'''

	def __init__(self, data:pd.DataFrame) -> None:
		'''
		Input:
			- data, pandas dataframe
		'''
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

	def plot_hist_scores(self, mu:float, file:str=None) -> plt.axis:
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
		fig, ax = plt.subplots(figsize=(4,3.5))
		if vs_models:
			ax = swarmplot(x=df['av_score'], hue='model', size=3)
		else:
			ax = swarmplot(x=df['av_score'], size=3)
		ax.axvline(x=mu, color='red', label='Fair quantity')
		ax.set_xlabel('Av. score per player')
		ax.grid()
		if file is not None:
			plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
			print('Plot saved to', file)
		return fig

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
				T:Optional[int]=20,
				file:Optional[Union[Path, None]]=None,
				kwargs:Optional[Dict[str,any]]={}
			) -> pd.DataFrame:
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
		if file is not None:
			plt.savefig(file, dpi=self.dpi, bbox_inches="tight")
			print('Plot saved to', file)
		else:
			print('Warning: No plot saved by plot_hist_states. To save plot, provide file name.')
		plt.close(fig)
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
		fig.close()

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

	def plot_efficiency_vs_entropy(
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
		# Get efficiency and entropy
		measures = ['efficiency', 'entropy']
		gm = GetMeasures(self.data, measures, T=T)
		df_measures = gm.get_measures()
		# Create the plot canvas
		fig, ax = plt.subplots(figsize=(self.width, self.height))
		ax.set_xlabel('Entropy')
		ax.set_ylabel('Efficiency')
		if vs_models:
			kdeplot(
				data=df_measures,
				x='entropy',
				y='efficiency',
				hue='model',
				cmap=self.cmaps[:num_models], 
				fill=True,
				ax=ax,
			)
			scatterplot(
				data=df_measures,
				x='entropy',
				y='efficiency',
				hue='model',
				legend=False,
				ax=ax
			)
		else:
			kdeplot(
				data=df_measures,
				x='entropy',
				y='efficiency',
				cmap=self.cmaps[0],
				fill=True,
				ax=ax
			)
			scatterplot(
				data=df_measures,
				x='entropy',
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
			print('Warning: No plot saved by plot_efficiency_vs_entropy. To save plot, provide file name.')

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
		gm = GetMeasures(self.data, measures, T=T)
		df_measures = gm.get_measures()
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
		gm = GetMeasures(self.data, measures, T=T)
		df_measures = gm.get_measures()
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
		gm = GetMeasures(self.data, measures, T=T)
		df_measures = gm.get_measures()
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
		gm = GetMeasures(self.data, measures, T=T)
		df_measures = gm.get_measures()
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
		gm = GetMeasures(
			data=self.data, 
			measures=measures, 
			T=T,
			per_player=True
		)
		df_measures = gm.get_measures()
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
				images_folder: Path
			) -> None:
		self.data = data
		self.history = self.get_history()
		self.thresholds = list()
		group_column = PPT.get_group_column(data.columns)
		self.room = data[group_column].unique()[0]
		num_players_column = PPT.get_num_player_column(data.columns)
		self.num_players = data[num_players_column].unique()[0]
		self.images_folder = images_folder
		# Determine color
		self.go_color='blue'
		self.no_go_color='lightgray'
		self.dpi = 300

	def __str__(self) -> str:
		return f'room:{self.room} --- num_players:{self.num_players} --- thresholds:{self.thresholds}'

	def render(self, num_rounds: Optional[int]=30):
		file = PathUtils.add_file_name(
			path=self.images_folder, 
			file_name=f'room{self.room}',
			extension='png'
		)
		self.render_threshold(file, num_rounds)
		print(f'Bar attendance saved to file {file}')

	def get_history(self):
		history = list()
		for round, grp in self.data.groupby('round'):
			history.append(grp.decision.tolist())
		return history

	def render_threshold(
				self, 
				file:Optional[Union[Path, None]]=None, 
				title:Optional[Union[str, None]]=None,
				num_rounds:Optional[int]=30
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
		fig, axes = plt.subplots(figsize=(0.5*num_rounds,0.5*self.num_players))
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
		if file is not None:
			plt.savefig(file, dpi=self.dpi)
			plt.close()
		else:
			plt.plot()