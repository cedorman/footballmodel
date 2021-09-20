import seaborn as sns
import pandas
import numpy as np

import logger
from football import Football

logger.getLogger().setLevel(logger.DEBUG)
logger.debug("Staring scatterplot matrix")

ft = Football()
ft.read_data()

rush_or_pass = ft.football_data.loc[ft.football_data['PlayType'].isin(['RUSH','PASS'])]
subset = rush_or_pass[["Down", "ToGo", "YardLine", "YardLineFixed", "SeriesFirstDown", "Quarter", "PlayType"]]



# Add random values to coumns
subset['Down'] = subset['Down'] + np.random.rand(subset.shape[0]) * 0.2 - 0.1
subset['Quarter'] =subset['Quarter'] + np.random.rand(subset.shape[0]) * 0.2 - 0.1
subset['SeriesFirstDown'] =subset['SeriesFirstDown'] + np.random.rand(subset.shape[0]) * 0.2 - 0.1

sns.set_theme(style='ticks')
sns.pairplot(subset, hue='PlayType', plot_kws=dict(alpha=0.4) )
logger.info("scatterplot created")