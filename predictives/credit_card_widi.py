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
import pandas_log as pl
import seaborn as sns
import logging
from IPython import get_ipython
from pandas import DataFrame


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# %matplotlib inline
# get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', 1000)


class PredictionError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


class PredictiveProcessor:

    data_dir = 'data'
    file_name = 'credit_card_default.csv'
    file_url = ''
    dataframe = DataFrame()

    numerical_features = []
    categorical_features = []
    # bill_amt_features = ['bill_amt'+ str(i) for i in range(1,7)]
    # pay_amt_features = ['pay_amt'+ str(i) for i in range(1,7)]
    # numerical_features = ['limit_bal','age'] + self.bill_amt_features + self.pay_amt_features

    def __init__(self, *args, **kwargs):
        logger.info("LL: -------------")
        logger.info("LL: init")
        logger.info("LL: -------------")

    def data_read_csv(self, *args, **kwargs) -> DataFrame:
        logger.info("LL: -------------------------------")
        logger.info("LL: data_read_csv")
        logging.debug(' fileurl : \n %s  ', self.file_url)
        df = pd.read_csv(self.file_url)
        self.dataframe = df
        logger.info("LL: -------------------------------")
        return df

    def problem_understanding(self, *args, **kwargs)-> DataFrame:
        logger.info("LL: -----------------------------------------------")
        logger.info("LL: problem_understanding start")
        # df.shape
        logger.info("LL: dataFrame.shape() \n%s ", df.shape)
        # df.head()
        # logger.info("LL: df.head() \n%s", df.head())
        # df.columns
        logger.info("LL: df.columns \n%s", df.columns)
        # df.dtypes    
        # logger.info("LL: df.dtypes \n%s", df.dtypes)
        columns = df.columns
        if 'ID' in columns:
            # df.set_index(columns[0])
            df.set_index(columns[columns.get_loc('ID')],inplace=True)
            df.reindex()
        else :
            idxlist = np.arange(1,df.shape[0]+1)
            df["ID"]= idxlist
            df.set_index("ID",inplace=True)
            df.reindex()
        # df.head()
        # logger.info("LL: df.head() \n%s", df.head())
        # df.describe().transpose()
        # logger.info("LL: df.describe() \n%s", df.describe().transpose())
        # df.min()
        # logger.info("LL: df.min() \n%s", df.min())
        # df.max()
        # logger.info("LL: df.max() \n%s", df.max())
        # df.median
        # logger.info("LL: df.median \n%s", df.median())
        # df.id
        # logger.info("LL: df.id \n%s", df.id)
        # df.idxmin()
        # logger.info("LL: df.idxmin() \n%s", df.idxmin())
        # df.idxmax()
        # logger.info("LL: df.idxmax() \n%s", df.idxmax())
        # logger.info("LL: df.sort_values(columns[1]).head(10)\n%s", df.sort_values(columns[1]).head(10))
        numerical_features = []
        categorical_features = []

        for column in df.columns:
            # logger.info("amount of unique column \n %s is : %s",column,df[column].unique().size)
            if(df[column].unique().size<=11):
                # logger.info("LL: column categorical %s", column)
                categorical_features.append(column)
            else:
                # logger.info("LL: column numerical %s", column)
                numerical_features.append(column)
                

        logger.info("LL: numerical_features \n%s", numerical_features)
        logger.info("LL: categorical_features \n%s", categorical_features)
        
        sns.set()
        # sns.scatterplot(x='AGE', y='BILL_AMT2', data=df)

        # sns.scatterplot(x='AGE', y='BILL_AMT2', data=df,hue='EDUCATION',fit_reg=False, palette='bright',markers=['o','x','v','+'])
        
        # sns.distplot(df['BILL_AMT2'], bins=15)

        sns.jointplot(x='AGE', y='BILL_AMT2', data=df, kind='scatter', marginal_kws=dict(bins=15))
        
        sns.jointplot(x='AGE', y='BILL_AMT2', data=df, kind='kde', marginal_kws=dict(bins=15))

        sns.violinplot(x='AGE',y='BILL_AMT2', data=df)

        sns.pairplot(data=df)
        vars_to_plot = ['CRIM', 'AGE', 'DIS', 'LSTAT', 'MEDV']
        sns.pairplot(data=df, vars=vars_to_plot)
        logger.info("LL: problem_understanding end")
        logger.info("LL: -----------------------------------------------")

    def data_preparation(self, *args, **kwargs):
        logger.info("LL: -------------------------------------")
        logger.info("LL: data_preparation start")
        logger.info("LL: data_preparation end")
        logger.info("LL: -------------------------------------")

    def data_analysis_exploratory(self, *args, **kwargs):
        logger.info("LL: ----------------------------------")
        logger.info("LL: data_analysis_exploratory start")
        logger.info("LL: data_analysis_exploratory end")
        logger.info("LL: ----------------------------------")

    def data_model_building(self, *args, **kwargs):
        logger.info("LL: -------------------------------------------")
        logger.info("LL: data_model_building start")
        logger.info("LL: data_model_building end")
        logger.info("LL: -------------------------------------------")

    def model_evaluation(self, *args, **kwargs):
        logger.info("LL: -------------------------------------")
        logger.info("LL: model_evaluation start")
        logger.info("LL: model_evaluation end")
        logger.info("LL: -------------------------------------")

    def model_deployment(self, *args, **kwargs):
        logger.info("LL: -------------------------------------")
        logger.info("LL: model_deployment start")
        logger.info("LL: model_deployment end")
        logger.info("LL: -------------------------------------")


def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None


def main():
    # get_ipython().run_line_magic('matplotlib', 'inline')
    data_dir = 'data'
    file_name = 'credit_card_default.csv'
    file_url = ''
    if not is_running_from_ipython():
        abspath = os.path.abspath('.')
        file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
    else:
        file_url = os.path.join('../', data_dir, file_name)
    # logging.debug('abspath %s', abspath)
    pp = PredictiveProcessor()
    pp.file_url = file_url
    df = pp.data_read_csv()
    pp.problem_understanding()
    pp.data_preparation()
    pp.data_analysis_exploratory()
    pp.data_model_building()
    pp.model_evaluation()
    pp.model_deployment()
    logger.info("LL: -----------------------------------------------")    
    
    
    
    logger.info("LL: -----------------------------------------------")

if __name__ == "__main__":
    main()
    # pass


# %%
