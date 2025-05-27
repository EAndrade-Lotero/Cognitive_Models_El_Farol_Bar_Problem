from pathlib import Path

PATHS = {
    'human_data':Path(Path.cwd(), '..', 'data', 'human').resolve(),
    'simulated_data':Path(Path.cwd(), '..', 'data', 'simulated').resolve(),
    'parameter_fit_results':Path.cwd() / Path('..').resolve() / Path('reports'),
    'index_path':Path(Path.cwd(), '..', 'data', 'indices').resolve()
}

# Chech if the paths exist
PATHS['human_data'].mkdir(parents=True, exist_ok=True)
PATHS['simulated_data'].mkdir(parents=True, exist_ok=True)
PATHS['parameter_fit_results'].mkdir(parents=True, exist_ok=True)
PATHS['index_path'].mkdir(parents=True, exist_ok=True)