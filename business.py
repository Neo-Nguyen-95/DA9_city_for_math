import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Support function

def wrangle_score(score_data_path):
    
    df = pd.read_csv(score_data_path)
    df.drop(columns='Unnamed: 0', inplace=True)
    df['category'] = df['lich_su'].isnull().map(
        lambda x: 'khtn' if x==True else 'khxh'
        )  # classify by domain: science / social
    
    return df

def z_test(df, pop_mean, pop_std):
    df['z_statistic'] = (
        (df['mean'] - pop_mean) / (pop_std / np.sqrt(df['count']))
        )
    
    return df


#%% Class

class ExamAnalyzer:
    
    def __init__(self, score_data_path, region_data_path):
        self.df_score = wrangle_score(score_data_path = score_data_path)
        self.df_region = (pd.read_csv(region_data_path)[['region', 'name']]
                          .set_index('region')
                          )
        
    def get_stat_by_region(self, category):
        
        #  Get mean and region
        df_overall_mean_by_region = (self.df_score[self.df_score['category']==category]
                             .groupby('region')['mean']
                             .mean()
                             .to_frame()
                             )
        
        
        df_overall_mean_by_region = df_overall_mean_by_region.join(self.df_region)
        
        #  Get mean for each subject
        subjects = ['toan', 'ngu_van', 'ngoai_ngu', 'region', 'economic']
        if category == 'khtn':
            subjects += ['vat_li', 'hoa_hoc', 'sinh_hoc']
        elif category == 'khxh':
            subjects += ['lich_su', 'dia_li', 'gdcd']
        else:
            print('Wrong input of kind')
        
        
        df_subject_mean_by_region = (
            self.df_score[self.df_score['category']==category][subjects]
            .groupby('region')
            .mean()
            )
        
        
        df_stat_by_region = df_overall_mean_by_region.join(df_subject_mean_by_region)
        
        #  Get count for each subject population
        df_count_by_region = (
            self.df_score[self.df_score['category']==category][subjects]
            .groupby('region')
            .size()
            .rename('count')
            )
        
        df_stat_by_region = df_stat_by_region.join(df_count_by_region)
        
        
        # Get Z-statistic for each subject population
        
        df_stat_by_region = df_stat_by_region.apply(
            z_test,
            axis=1,
            args=(
                self.get_population_statistic(category=category,stat='mean').values[0],
                self.get_population_statistic(category=category,stat='std').values[0]
                ))
        
        # Get standard deviation for each subject
        df_std_by_region = (self.df_score[self.df_score['category']==category]
                             .groupby('region')['mean']
                             .std()
                             .rename('std')
                             .to_frame()
                             )
        df_stat_by_region = df_stat_by_region.join(df_std_by_region)
        
        # Sort values by mean
        df_stat_by_region = df_stat_by_region.sort_values('mean', ascending=False)
        
        return df_stat_by_region 
    
    def get_population_statistic(self, category, stat):
        """
        Parameter:
        ---
        stat = 'mean' | 'median' | 'std' | 'var'
        """
        
        result = (
            self.df_score[self.df_score['category'] == category]['mean']
            .agg([stat])
            )
        
        return result
    
    def get_two_sample_z_statistic(self, category, region1, region2):
        df = self.get_stat_by_region(category=category)
        s1_mean = df.at[region1, 'mean']
        s1_std = df.at[region1, 'std']
        s1_n = df.at[region1, 'count']
        
        s2_mean = df.at[region2, 'mean']
        s2_std = df.at[region2, 'std']
        s2_n = df.at[region2, 'count']
        
        z = (
            (s1_mean - s2_mean) 
             / np.sqrt(s1_std ** 2 / s1_n + s2_std ** 2 / s2_n)
             )
        
        return z
        
    
    


