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


class CreditCardPredictiveTest():
    creditCardPredictive = CreditCardPredictive()

    def allTest(self, data_dir, file_name):
        abspath = os.path.abspath('.')
        file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
        # file_url = os.path.join(abspath,self.data_dir, self.file_name)
        print('abspath', abspath)
        dp = self.creditCardPredictive
        dp.func_read_data_csv(file_url)
        dp.func_descriptif_data_analysis()
        dp.func_data_preparation()
        data = dp.diamonds

    def cobaPredict(self):
        carat_values = np.arange(0.5, 5.5, 0.5)
        self.creditCardPredictive.create_model_01()
        preds = self.creditCardPredictive.first_ml_model(carat_values)
        pdframe = pd.DataFrame(
            {"Carat": carat_values, "Predicted price": preds})
        print('prediction', pdframe.round(2))

# %%
class CreditCardPredictive:

    data_dir = 'data'
    file_name = 'credit_card_default.csv'
    creditCardDefault = DataFrame()

    numerical_features = []
    categorical_features = []

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
        
        print('-----------------func_data_cleansing end')

    def func_data_encode_categorical_feature(self):
        
        print('-----------------func_data_encode_categorical_feature end')

    def func_data_preparation(self):
        self.func_data_cleansing()
        self.func_data_encode_categorical_feature()
        print('-----------------func_data_preparation end')

    def desc_num_feature(self, feature_name, bins=30, edgecolor='k', **kwargs):
        print('-----------------desc_num_feature end')

    model_01 = 0.0

    def create_model_01(self):
        self.model_01 = np.mean(self.diamonds['price']/self.diamonds['carat'])

    def first_ml_model(self, carat):
        return self.model_01 * carat

    def train_test_split(self):
        print('-----------------train_test_split end')

        # print(sns.pairplot(X_train[['x','y','z']], plot_kws={"s": 3}));


def main():
    get_ipython().run_line_magic('matplotlib', 'inline')
    data_dir = '../data'
    file_name = 'credit_card_default.csv'
    
    file_url = os.path.join(data_dir, file_name)
    # print('abspath', abspath)
    dp = CreditCardPredictive()
    dp.func_read_data_csv(file_url)
    dp.func_descriptif_data_analysis()
    # dp.func_explanatory_data_analysis()
    # dp.func_data_preparation()
    data = dp.creditCardDefault

if __name__ == "__main__":
    main()


# %%

# if __name__ == '__main__':
#         data_dir = '../data'
#         file_name = 'diamonds.csv'
#         test = DiamondsTest()
#         test.allTest(data_dir,file_name)


# %%
