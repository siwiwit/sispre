# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
## Global Variables

# %%

import os
import sys
import unittest
import urllib.error
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython
from pandas import DataFrame

# get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib inline
get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)


class DiamondsPredictive:

        data_dir = '../data'
        file_name = 'diamonds.csv'
        diamonds = DataFrame()

        numerical_features = ['price', 'carat', 'depth', 'table', 'x', 'y', 'z']
        categorical_features = ['cut', 'color', 'clarity']

        def __init__(self):
                pass
        def func_read_data_csv(self):
                file_url = os.path.join(self.data_dir, self.file_name)
                self.diamonds = pd.read_csv(file_url)
                print('func_read_data_csv end')

        def func_data_cleansing(self):
                self.diamonds = self.diamonds.loc[(self.diamonds['x']>0) | (self.diamonds['y']>0)]
                self.diamonds = self.diamonds.loc[~((self.diamonds['y'] > 30) | (self.diamonds['z'] > 30))]
                print('func_data_cleansing end')
        def func_data_encode_categorical_feature(self):
                for x in categorical_features:                        
                        self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds[x], prefix=x, drop_first=True)], axis=1)
                # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['cut'], prefix='cut', drop_first=True)], axis=1)
                # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['color'], prefix='color', drop_first=True)], axis=1)
                # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['clarity'], prefix='clarity', drop_first=True)], axis=1)
                print('func_data_encode_categorical_feature end')
        def func_data_preparation(self):
                self.func_data_cleansing()
                # diamond_categorical_cut_nodrop = pd.get_dummies(self.diamonds['cut'], prefix='cut')
                # diamond_categorical_cut = pd.get_dummies(self.diamonds['cut'], prefix='cut', drop_first=True)
                self.func_data_encode_categorical_feature()
                print('func_data_preparation end')

        def desc_num_feature(feature_name, bins=30, edgecolor='k', **kwargs):
                fig, ax = plt.subplots(figsize=(8,4))
                diamonds[feature_name].hist(bins=bins, edgecolor=edgecolor, ax=ax, **kwargs)
                ax.set_title(feature_name, size=15)
        def func_explanatory_data_analysis(self):
                for x in numerical_features:
                        self.desc_num_feature(x)
                print('func_explanatory_data_analysis end')
        
        
        model_01 = 0.0
        def create_model_01(self):
                self.model_01 = np.mean(self.diamonds['price']/self.diamonds['carat'])
        def first_ml_model(self,carat):
                return self.model_01 * carat

        def train_test_split(self):
                from sklearn.model_selection import train_test_split
                X = self.diamonds.drop(['cut','color','clarity','price'], axis=1)
                y = self.diamonds['price']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
                print('plotting train test')
                sns.pairplot(X_train[['x','y','z']], plot_kws={"s": 3})
                # print(sns.pairplot(X_train[['x','y','z']], plot_kws={"s": 3}));

class DiamondsTest():
        diamondPredictive = DiamondsPredictive()
        
        def allTest(self):
                print('coba')
                dp = self.diamondPredictive
                dp.func_read_data_csv()
                dp.func_data_preparation()
                dp.func_explanatory_data_analysis()
                data = dp.diamonds
                # print('describe',diamond.describe().round(2))
                # print('head',diamond.head())
                # self.cobaPredict()
                dp.train_test_split()
        
        def cobaPredict(self):
                carat_values = np.arange(0.5, 5.5, 0.5)
                self.diamondPredictive.create_model_01()
                preds = self.diamondPredictive.first_ml_model(carat_values)
                pdframe = pd.DataFrame({"Carat": carat_values, "Predicted price":preds})
                print('prediction',pdframe.round(2))

if __name__ == '__main__':
        test = DiamondsTest()
        test.allTest()






# %%
