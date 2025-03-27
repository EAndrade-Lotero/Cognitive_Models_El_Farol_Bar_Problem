from pathlib import Path

PATHS = {
    'human_data':Path.cwd() / Path('..').resolve() / Path('data', 'human'),
    'parameter_fit_results':Path.cwd() / Path('..').resolve() / Path('reports')
}