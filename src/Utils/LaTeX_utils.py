import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

class PrintLaTeX :
	
	@staticmethod
	def wrap_with_header_and_footer(
				latex_string: str
			) -> str:
		# Header and footer of LaTeX file
		header, footer = PrintLaTeX.header_footer()
		string_list = [header, latex_string, footer]
		return '\n'.join(string_list)
	
	@staticmethod
	def save_to_file(
				latex_string: str,
				latex_file: Path
			) -> None:
		with open(latex_file, 'w', encoding='utf-8') as f:
			f.write(latex_string)
		print(f'LaTeX file written to {latex_file}')

	@staticmethod
	def print_sweep(
			parameters: Dict[str, any],
			sweep_parameter: str,
			values: List[any],
			list_of_paths: List[Path],
			before_space: Optional[bool]=True,
		) -> str:
		parameters_ = {
			key:value for key, value in parameters.items() if key != sweep_parameter
		}
		latex_string = PrintLaTeX.print_parameters(
			parameters=parameters_,
			before_space=before_space
		)
		latex_string += '\n' + r'\vspace{\baselineskip}' + '\n'
		latex_string += '\n' + r'\begin{tabular}{cc}\toprule' + '\n'
		latex_string += '\n' + r'\multicolumn{2}{c}{Parameter sweep}\\' + '\n'
		latex_string += '\n' + r'Parameter & Values\\\midrule' + '\n'
		sweep_parameter_ = sweep_parameter.replace('_', r'\_')
		latex_string += '\n' + f'{sweep_parameter_} & {values}' + r'\\\bottomrule' + '\n'
		latex_string += '\n' + r'\end{tabular}' + '\n'
		latex_string += '\n' + PrintLaTeX.print_table_from_figs(
			figs=list_of_paths,
			rows_over_cols=True,
		)
		return latex_string

	@staticmethod
	def header_footer() -> Tuple[str]:
		'''
		Creates the header and end of a LaTeX article as a string
		'''
		header = r'''
\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}

\begin{document}

'''	
		footer = r'''\end{document}'''
		return (header, footer)

	@staticmethod
	def print_parameters(
				parameters: Dict[str, any],
				size: Optional[float]=0.8,
				before_space: Optional[bool]=True,
				are_free: Optional[bool]=True,
				exclusion_list: Optional[List[str]]=[]
			) -> str:
		'''
		Transforms a dictionary with parameters into 
		a LaTeX table
		'''
		# Exclude parameters from exclusion list
		parameters_ = {key:value for key, value in parameters.items() if key not in exclusion_list}
		# Format for better printing
		formated_dict = PrintLaTeX.format_dict(parameters_)
		# Use pandas dataframe for creating LaTeX table
		df = pd.DataFrame(formated_dict)
		latex_string = ''
		if before_space:
			latex_string += '\n' + r'\vspace{\baselineskip}' + '\n\n' 
		if are_free:
			latex_string += '\n\n' + r'Free parameters:' + '\n\n'
		else:
			latex_string += '\n\n' + r'Fixed parameters:' + '\n\n'
		latex_string += r'\scalebox{' + str(size) + '}{\n'
		latex_string += r'\begin{minipage}{\textwidth}' + '\n'	
		# Create LaTeX table with pandas functionality	
		latex_string += df.style.hide(axis=0).to_latex(
			column_format='c'*len(parameters.keys()),
			hrules=True
		)
		latex_string = latex_string.replace('_', r'\_')
		latex_string += r'\end{minipage}' + '\n'		
		latex_string += '}\n'
		return latex_string

	@staticmethod
	def print_table_from_figs(
				figs: List[Path], 
				rows_over_cols: Optional[bool]=False,
				num_rows: Optional[Union[int, None]]=None,
				before_space: Optional[bool]=True
			) -> str:
		'''
		Creates a table that displays the figures
		'''
		assert(len(figs) > 0)
		if num_rows is None:
			num_rows = len(figs)
		else:
			raise Exception('Feature not defined! (num_row is not None)')
		table_string = ''
		if before_space:
			table_string += '\n' + r'\vspace{\baselineskip}' + '\n\n' 
		string_list = list()
		if not rows_over_cols:
			step = np.format_float_positional(1 / num_rows - 0.01, precision=3)
			for fig in figs:
				assert(isinstance(fig, Path))
				latex_string = fr'\includegraphics[width={step}\textwidth]' + '{' + fig.name + '}'
				string_list.append(latex_string)
			table_string += r'\begin{tabular}{' + 'c'*num_rows + '}\n'
			table_string += ' &\n'.join(string_list) 
			table_string += '\n' + r'\end{tabular}' + '\n'
		else:
			for fig in figs:
				assert(isinstance(fig, Path))
				latex_string = fr'\includegraphics[width=\textwidth]' + '{' + fig.name + '}'
				string_list.append(latex_string)
			table_string += r'\begin{tabular}{' + 'c' + '}\n'
			table_string += r'\\'.join(string_list) 
			table_string += '\n' + r'\end{tabular}' + '\n'
		return table_string

	@staticmethod
	def format_dict(old_dict: Dict[str, any]) -> Dict[str, any]:
		new_dict = dict()
		for key, value in old_dict.items():
			if isinstance(value, float):
				new_value = np.format_float_positional(value, precision=2)
			elif isinstance(value, np.ndarray):
				new_value = np.zeros(value.shape)
				for idx, x in np.ndenumerate(value):
					if isinstance(x, float):
						new_x = np.format_float_positional(x, precision=2)
					else:
						new_x = x
					new_value[idx] = new_x
				new_value = str(new_value)
			else:
				new_value = value
			new_dict[key] = [new_value]
		return new_dict 