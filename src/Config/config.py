from pathlib import Path

PATHS = {
    'human_data': Path(Path.cwd(), '..', 'data', 'human').resolve(),
    'simulated_data': Path(Path.cwd(), '..', 'data', 'simulated').resolve(),
    'parameter_fit_results': Path(Path.cwd(), '..', 'reports', 'MLE').resolve(),
    'index_path': Path(Path.cwd(), '..', 'data', 'indices').resolve(),
    'bar_images': Path(Path.cwd(), '..', 'images', 'bar_images').resolve(),
    'data_likelihoods': Path(Path.cwd(), '..', 'data', 'likelihoods').resolve(),
    'figures_for_paper': Path(Path.cwd(), '..', 'Figures for article').resolve(),
    'LaTeX': Path(Path.cwd(), '..', 'LaTeX').resolve(),
}

# Chech if the paths exist
for name, folder in PATHS.items():
    folder.mkdir(parents=True, exist_ok=True)