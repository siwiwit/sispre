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
import pandas_log as pl
import seaborn as sns
import logging
import bokeh
import plotly
import pyglet
import geoplotlib
import missingno
from IPython import get_ipython
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    categorical_features_nonint = []
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
        df = self.dataframe
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
        
        # sns.violinplot(x='Default',y='BILL_AMT2', data=df)

        sns.pairplot(data=df)
        vars_to_plot = ['CRIM', 'AGE', 'DIS', 'LSTAT', 'MEDV']
        sns.pairplot(data=df, vars=vars_to_plot)        
        logger.info("LL: problem_understanding end")
        logger.info("LL: -----------------------------------------------")
        return df

    def data_preparation(self, *args, **kwargs)-> DataFrame:
        logger.info("LL: -------------------------------------")
        logger.info("LL: data_preparation start")
        logger.info("LL: data_preparation end")
        logger.info("LL: -------------------------------------")
        return self.dataframe

    def data_preparation_encode_categorical_nonint(self, *args, **kwargs):
        logger.info("LL: -------------------------------------------")
        logger.info("LL: data_preparation_encode_categorical_nonint start")
        logger.info("LL: data_preparation_encode_categorical_nonint end")
        logger.info("LL: -------------------------------------------")

    def data_analysis_exploratory(self, *args, **kwargs)-> DataFrame:
        logger.info("LL: ----------------------------------")
        logger.info("LL: data_analysis_exploratory start")
        logger.info("LL: data_analysis_exploratory end")
        logger.info("LL: ----------------------------------")
        return self.dataframe

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
    #column_names = ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month
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
    pp.df = df
    # pp.problem_understanding()
    pp.data_preparation()
    pp.data_analysis_exploratory()
    pp.data_model_building()
    pp.model_evaluation()
    pp.model_deployment()
    logger.info("LL: -----------------------------------------------")    
    
    # check is any null values
    for column in df.columns:
        # logger.info("LL: column %s datatype %s is having null : %s , about %s", column, df.dtypes[column], df[column].isnull().values.any(), df[column].isnull().values.sum())
        # logger.info("LL: column %s datatype %s is having NA : %s , about %s", column, df.dtypes[column], df[column].isna().values.any(), df[column].isna().values.sum())
        if(df.dtypes[column], df[column].isnull().values.any() and df.dtypes[column], df[column].isna().values.any()):
            if(df.dtypes[column]=='int64'):
                logger.info("LL: yes int64")
                # df_example = df[column].fillna(0)
            else:
                logger.info("LL: not int64")
    #alternatively use missingno
    #-----------missingno

    
    logger.info("LL: df.shape \n%s", df.shape)    
    
    
    logger.info("LL: df correlation \n%s", df.corr().round(2))
    logger.info("LL: df correlation columns \n%s", df.corr().columns)
    logger.info("LL: df correlation index \n%s", df.corr().index)
    logger.info("LL: df covariance \n%s", df.cov().round(2))
    logger.info("LL: df covariance columns \n%s", df.cov().columns)
    logger.info("LL: df covariance index \n%s", df.cov().index)
    # drop rows with null values
    df_dropped_rows_na = df.dropna(axis=0)    
    logger.info("LL: df_dropped_rows_na.shape %s", df_dropped_rows_na.shape)
    # drop columns with null values
    df_dropped_cols_na = df.dropna(axis=1)
    logger.info("LL: df_dropped_cols_na.shape %s", df_dropped_cols_na.shape)

    #impute nan with new values
    # missing_values = form of missing values in your data. (For example nan, 0, or "n/a".
    # strategy = how to impute (choices are "mean", "median", "most frequent", and "constant". 
    # If you pass strategy=constant, then you can use the optional argument fill_value to pass your constant.

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    cols_to_impute = df.columns
    out_imp = imputer.fit_transform(df[cols_to_impute])
    df_new = pd.DataFrame(data = out_imp, columns = cols_to_impute)
    # df_new = pd.concat([df_new, df[['species']]], axis = 1)

    minmax_scaler = MinMaxScaler()
    cols_to_minmaxscale = df.columns
    out_scaled_minmax = minmax_scaler.fit_transform(df[cols_to_minmaxscale])
    
    standard_scaler = StandardScaler()
    cols_to_standardscale = df.columns
    out_scaled_standard = standard_scaler.fit_transform(df[cols_to_standardscale])

    # encode categorical nonint features
    categorical_features_nonint = []
    for column in df.columns:
        if(df.dtypes[column]!='int64' or df.dtypes[column]!='int32'):
                logger.info("LL: no int64 or int32")
                categorical_features_nonint.append(column)
                # df_example = df[column].fillna(0)
            else:
                logger.info("LL: yes int64 or int32")
    
    from sklearn.preprocessing import OrdinalEncoder
    enc_ordinal = OrdinalEncoder()
    out_enc_ord_catg_feat_nonint = enc_ordinal.fit_transform(df[categorical_features_nonint])
    logger.info("LL: out_enc categories \n%s",enc_ordinal.categories_)
    logger.info("LL: out_enc  \n%s",out_enc_ord_catg_feat_nonint)
    df[categorical_features_nonint] = out_enc_ord_catg_feat_nonint
    logger.info("LL: df_new  \n%s",df.head())

    # One-hot Enconding
    from sklearn.preprocessing import OneHotEncoder
    enc_onehot = OneHotEncoder(sparse=False)
    out_enc_onehot_catg_feat_nonint = enc_onehot.fit_transform(df[categorical_features_nonint])
    new_cols_onehot_catg_feat_nonint = enc_onehot.get_feature_names(categorical_features_nonint).tolist()
    logger.info("LL: new_cols \n%s", new_cols_onehot_catg_feat_nonint)


    # Label encoding
    from sklearn import preprocessing
    enc_label = preprocessing.LabelEncoder()
    out_enc_label = enc_label.fit_transform(categorical_features_nonint)

    # Dimension Reduction
    #  Feature Selection
    #  Feature Filtering
    #   Variance Treshold
    #   Correlation Coefficient

    #Variance Treshold
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    cols = df.columns
    # cols = categorical_features_nonint
    selector.fit(df[cols])
    
    # check feature variances before selection
    logger.info("LL: variance treshold \n%s", selector.variances_)
    
    # set threshold into selector object
    selector.set_params(threshold=1.0)
    out_sel = selector.fit_transform(df[cols])

    logger.info("LL: selector.get_support() \n%s", selector.get_support)
    df_sel = df.iloc[:, selector.get_support()]

    # add labels to new dataframe and sanity check
    df_sel = pd.concat([df_sel, df[['default payment next month']]], axis = 1)
    logger.info("LL: df_sel.head() \n%s", df_sel.head())


    #Correlation Coefficient
    cor = df.corr()
    sns.heatmap(cor, annot=False, cmap=plt.cm.Blues)
    logger.info("LL: plt.show() \n%s", plt.show())
    
    # get correlation values with target variable
    cor_target = abs(cor['default payment next month'])
    logger.info("LL: cor_target \n%s", cor_target)

    #For demonstration purposes, we will choose 0.6 as the threshold and then filter. From the output, you should expect columns 5 and 12 (0.69 and 0.74) to be selected:
    selected_cols = cor_target[cor_target>0.6]
    logger.info("LL: selected columns, correlation with target > 0.6")
    logger.info("LL: selected_cols \n%s", selected_cols)
    # filter in the selected features
    df_sel = df[selected_cols.index]
    logger.info("LL: df_sel.head() \n%s," def_sel.head())

    # Wrapper Methods
    #  Sequential Feature Selection
    #   Forward Sequential Selection and Backward Sequential Selection
    #   LinearRegression() for continuous target variables and RandomForestClassifier() for categorical target variables

    # We will use the Support Vector Machine Classifier ("SVC") as the estimator for our example RFE. 
    # Now let's import our modules and define the independent (X) and dependent (y) variables for the SVC:
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVC

    cols = df.columns
    X = df[cols]
    y = df['default payment next month']
    
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=2, step=1)
    rfe.fit(X, y)
    
    logger.info("LL: cols \n%s", cols)
    logger.info("LL: rfe.ranking_)

    logger.info("LL: -----------------------------------------------")

if __name__ == "__main__":
    main()
    # pass


# %%
