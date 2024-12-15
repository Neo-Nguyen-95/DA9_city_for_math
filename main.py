"""
- whether mean score really different from the mean
- city for the math score

"""

#%% LIB

import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from business import ExamAnalyzer


#%% IMPORT & SUPPORT FUNCTION

ea = ExamAnalyzer('diem_thi_thpt_2024_cleaned.csv', 'region_map.csv')

# check
ea.df_score
ea.df_region
ea.get_stat_by_region(category='khtn')

ea.get_two_sample_z_statistic(
    category='khxh',
    region1=16,
    region2=27
    )


# %% OVERALL SCIENCE MEAN SCORE ANALYSIS
ea.get_overall_mean_median(category='khtn')

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=ea.df_score[ea.df_score['category'] == 'khtn'],
            x='mean'
            )
plt.show();



# %% MEAN SCORE ANALYSIS BY REGION

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=ea.get_stat_by_region(category='khtn'), 
             x='z_statistic'
             )
plt.show();


