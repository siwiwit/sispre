# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# Global Variables

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


# %matplotlib inline
# get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)

# %%


class DiamondsTest():
    diamondPredictive = DiamondsPredictive()

    def allTest(self, data_dir, file_name):
        abspath = os.path.abspath('.')
        file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
        # file_url = os.path.join(abspath,self.data_dir, self.file_name)
        print('abspath', abspath)
        dp = self.diamondPredictive
        dp.func_read_data_csv(file_url)
        dp.func_descriptif_data_analysis()
        # dp.func_explanatory_data_analysis()
        dp.func_data_preparation()
        data = dp.diamonds
        # print('describe',diamond.describe().round(2))
        # print('head',diamond.head())
        # self.cobaPredict()
        # dp.train_test_split()

    def cobaPredict(self):
        carat_values = np.arange(0.5, 5.5, 0.5)
        self.diamondPredictive.create_model_01()
        preds = self.diamondPredictive.first_ml_model(carat_values)
        pdframe = pd.DataFrame(
            {"Carat": carat_values, "Predicted price": preds})
        print('prediction', pdframe.round(2))

# %%
class DiamondsPredictive:

    data_dir = 'data'
    file_name = 'diamonds.csv'
    diamonds = DataFrame()

    numerical_features = ['price', 'carat', 'depth', 'table', 'x', 'y', 'z']
    categorical_features = ['cut', 'color', 'clarity']

    def __init__(self):
        pass

    def func_read_data_csv(self, file_url):

        print(' fileurl', file_url)
        self.diamonds = pd.read_csv(file_url)
        print('-----------------func_read_data_csv end')

    def func_descriptif_data_analysis(self):
        
        self.func_tabular_data_analysis()
        self.func_statistical_data_analysis()
        self.func_graphical_data_analysis()
        print('-----------------func_descriptif_data_analysis end')

    def func_tabular_data_analysis(self):
        print('HEAD 20 records :\n', self.diamonds.head(20))
        print('TAIL 20 records :\n', self.diamonds.tail(20))
        print('COUNT EACH Column : :\n', self.diamonds.count())
        print('DESCRIPTIVE : :\n',self.diamonds.describe())
        print('COLUMNS : :\n',self.diamonds.columns)
        print('MEAN Per CUT : :\n',self.diamonds.groupby('cut').mean())
        print('SUM Per Color : :\n',self.diamonds.groupby('color').sum())
        print('DATA TYPES : :\n',self.diamonds.dtypes)
        print('INDEX : :\n',self.diamonds.index)
        # self.diamonds.values
        for categ_col in self.categorical_features:
            print('UNIQUE DATA of : :\n',categ_col, self.diamonds[categ_col].unique())
        print('SORT By Cut and Color  : :\n',self.diamonds.sort_values(by =['cut','color'], ascending=[True,True]).head(10))
        # self.diamonds.sort_values(by =self.categorical_features, ascending=[True,True])
        print('Filter 10 rows with price above mean  : :\n',self.diamonds[self.diamonds['price']> self.diamonds['price'].mean()].head(10))
        # print('',df[df['Species'].isin(['versicolor', 'virginica'])]
        # df.duplicated()
        # df.drop_duplicates(['first_name'], keep='first')
        print('COVARIAN ANALYSIS ::\n',self.diamonds.cov())
        print('CORELATION ANALYSIS ::\n', self.diamonds.corr())
        print('PIVOT :\n',self.diamonds.pivot_table(values='price', index=['cut', 'color'], columns=['clarity'], aggfunc=np.mean, fill_value=0))
        print('-----------------func_tabular_data_analysis end')

    def func_graphical_data_analysis(self):
        self.diamonds.hist()
        plt.show()
        self.diamonds.plot()
        plt.show()
        self.diamonds.boxplot()
        plt.show()
        print('-----------------func_graphical_data_analysis end')

    def func_statistical_data_analysis(self):
        print('-----------------func_statistical_data_analysis end')

    def func_explanatory_data_analysis(self):
        for x in self.numerical_features:
            self.desc_num_feature(x)
        print('-----------------func_explanatory_data_analysis end')

    def func_data_cleansing(self):
        self.diamonds = self.diamonds.loc[(self.diamonds['x'] > 0) | (self.diamonds['y'] > 0)]
        self.diamonds = self.diamonds.loc[~((self.diamonds['y'] > 30) | (self.diamonds['z'] > 30))]
        print('-----------------func_data_cleansing end')

    def func_data_encode_categorical_feature(self):
        for x in self.categorical_features:
            self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds[x], prefix=x, drop_first=True)], axis=1)
        # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['cut'], prefix='cut', drop_first=True)], axis=1)
        # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['color'], prefix='color', drop_first=True)], axis=1)
        # self.diamonds = pd.concat([self.diamonds, pd.get_dummies(self.diamonds['clarity'], prefix='clarity', drop_first=True)], axis=1)
        print('-----------------func_data_encode_categorical_feature end')

    def func_data_preparation(self):
        self.func_data_cleansing()
        # diamond_categorical_cut_nodrop = pd.get_dummies(self.diamonds['cut'], prefix='cut')
        # diamond_categorical_cut = pd.get_dummies(self.diamonds['cut'], prefix='cut', drop_first=True)
        self.func_data_encode_categorical_feature()
        print('-----------------func_data_preparation end')

    def desc_num_feature(self, feature_name, bins=30, edgecolor='k', **kwargs):
        fig, ax = plt.subplots(figsize=(8, 4))
        self.diamonds[feature_name].hist(bins=bins, edgecolor=edgecolor, ax=ax, **kwargs)
        ax.set_title(feature_name, size=15)
        plt.show()

    model_01 = 0.0

    def create_model_01(self):
        self.model_01 = np.mean(self.diamonds['price']/self.diamonds['carat'])

    def first_ml_model(self, carat):
        return self.model_01 * carat

    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        X = self.diamonds.drop(['cut', 'color', 'clarity', 'price'], axis=1)
        y = self.diamonds['price']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=123)
        print('plotting train test')
        sns.pairplot(X_train[['x', 'y', 'z']], plot_kws={"s": 3})

        # print(sns.pairplot(X_train[['x','y','z']], plot_kws={"s": 3}));


def main():
    get_ipython().run_line_magic('matplotlib', 'inline')
    data_dir = '../data'
    file_name = 'diamonds.csv'
    # test = DiamondsTest()
    # test.allTest(data_dir, file_name)
    # abspath = os.path.abspath('.')
    # file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
    # file_url = os.path.join(abspath,self.data_dir, self.file_name)
    file_url = os.path.join(data_dir, file_name)
    # print('abspath', abspath)
    dp = DiamondsPredictive()
    dp.func_read_data_csv(file_url)
    dp.func_descriptif_data_analysis()
    # dp.func_explanatory_data_analysis()
    # dp.func_data_preparation()
    data = dp.diamonds

if __name__ == "__main__":
    main()


# %%

# if __name__ == '__main__':
#         data_dir = '../data'
#         file_name = 'diamonds.csv'
#         test = DiamondsTest()
#         test.allTest(data_dir,file_name)


# %%
