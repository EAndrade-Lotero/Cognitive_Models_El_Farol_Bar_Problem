import pandas as pd
from pathlib import Path

from Utils.interaction import PlotsAndMeasures

human_data = Path('..', 'data', 'human', '2-player-UR.csv')
image_folder = Path.cwd() / Path('..').resolve() / Path('images', 'tests_ce')
image_folder.mkdir(parents=True, exist_ok=True)


data = pd.read_csv(human_data)
data['model'] = '2Player'

def test_ce():
	p = PlotsAndMeasures(data)
	file = image_folder / 'ce_2_player.png'
	p.plot_measures(
		measures=[
			'round_attendance',
			'attendance', 
			'deviation', 
			'efficiency', 
			'inequality',
			'entropy',
			'conditional_entropy'
		],
		file=image_folder
	)