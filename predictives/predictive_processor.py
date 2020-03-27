
##  Chapter 1: Data Mining and Getting Started with Python Tools
##     Descriptive, predictive, and prescriptive analytics
##     What will and will not be covered in this book
##        Recommended readings for further explanation
##     Setting up Python environments for data mining
##     Installing the Anaconda distribution and Conda package manager
##        Installing on Linux
##        Installing on Windows
##        Installing on macOS
##     Launching the Spyder IDE
##     Launching a Jupyter Notebook
##     Installing high-performance Python distribution
##     Recommended libraries and how to install
##        Recommended libraries
##     Summary

# %%

from IPython import get_ipython
from pandas import DataFrame
from scipy.cluster import hierarchy

from matplotlib.colors import ListedColormap

from sklearn import impute
from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE

from sklearn import decomposition
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.svm import SVC

from sklearn import discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


from sklearn.datasets import load_boston
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier

from sklearn import pipeline
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
# from predictive_processor import PredictiveProcessor
from predictive_processor import *


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
sns.set_context("paper", font_scale=1.5)
sns.set_style("white")

##  Chapter 2: Basic Terminology and Our End-to-End Example
##     Basic data terminology
##        Sample spaces
##        Variable types
##        Data types
##     Basic summary statistics
##     An end-to-end example of data mining in Python
##        Loading data into memory viewing and managing with ease using pandas
##        Plotting and exploring data harnessing the power of Seaborn
##        Transforming data PCA and LDA with scikit-learn
##        Quantifying separations k-means clustering and the silhouette score
##        Making decisions or predictions
##     Summary

# %%


class PredictiveProcessor:
    import predictive_processor

    data_dir = 'data'
    file_name = 'credit_card_default.csv'
    file_url = ''
    dataframe = DataFrame()
    X_train = DataFrame
    X_test = DataFrame
    y_train = DataFrame
    y_test = DataFrame
    #column_names = ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month
    
    column_names = []
    numerical_features = []
    categorical_features = []
    categorical_features_nonint = []
    idx_col_name = 'ID'
    target_col_name = ''
    # numerical_features =['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    # categorical_features =['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month']
    
    # bill_amt_features = ['bill_amt'+ str(i) for i in range(1,7)]
    # pay_amt_features = ['pay_amt'+ str(i) for i in range(1,7)]
    # numerical_features = ['limit_bal','age'] + self.bill_amt_features + self.pay_amt_features

    def __init__(self, *args, **kwargs):
        logger.info("LL: -------------")
        logger.info("LL: init")
        logger.info("LL: -------------")


##  Chapter 3: Collecting, Exploring, and Visualizing Data
##     Types of data sources and loading into pandas
##        Databases
##            Basic Structured Query Language (SQL) queries
##        Disks
##        Web sources
##            From URLs
##            From Scikit-learn and Seaborn-included sets

    def data_read_csv(self, *args, **kwargs) :
        logger.info("LL: -------------------------------")
        logger.info("LL: data_read_csv")
        logging.debug(' fileurl : \n %s  ', self.file_url)
        self.dataframe = pd.read_csv(self.file_url)        
        logger.info("LL: -------------------------------")
        # return df

##      Access, search, and sanity checks with pandas
    def datainfo_all(self, *args, **kwargs):
        logger.info("LL: -----------------------------------------------")
        logger.info("LL: datainfo start")
        df = self.dataframe
        self.datainfo_matrix_basic()
        self.datainfo_basic_getindex()
        self.datainfo_matrix_stats()
        self.datainfo_matrix_features()
        self.datainfo_plot_sns_base()
        self.datainfo_plot_sns_scatter()
        self.datainfo_plot_sns_histdist()
        self.datainfo_plot_sns_join()
        self.datainfo_plot_sns_violin()
        self.datainfo_plot_sns_pair()
        logger.info("LL: datainfo end")
        logger.info("LL: -----------------------------------------------")
        
    def datainfo_matrix_basic(self, *args, **kwargs):
        df = self.dataframe        
        # df.shape
        logger.info("LL: dataFrame.shape() \n%s ", df.shape)
        # df.head()
        logger.info("LL: df.head() \n%s", df.head())
        # df.columns
        logger.info("LL: df.columns \n%s", df.columns)
        # df.dtypes
        logger.info("LL: df.dtypes \n%s", df.dtypes)
        
    def datainfo_basic_getindex(self, *args, **kwargs):
        logger.info("LL: datainfo_df_index start")
        df = self.dataframe
        columns = df.columns
        if self.idx_col_name in columns:
            # df.set_index(columns[0])
            df.set_index(columns[columns.get_loc(self.idx_col_name)],inplace=True)
            df.reindex()
            logger.info("LL: df.reindex() 1")
        else :
            idxlist = np.arange(1,df.shape[0]+1)
            df[self.idx_col_name]= idxlist
            df.set_index(self.idx_col_name,inplace=True)
            df.reindex()
            logger.info("LL: df.reindex() 2")
        logger.info("LL: datainfo_df_index end")
        self.df = df

    def datainfo_matrix_stats(self, *args, **kwargs):
        df = self.dataframe
        df.head()
        logger.info("LL: df.head() \n%s", df.head())
        df.describe().transpose()
        logger.info("LL: df.describe() \n%s", df.describe().transpose())
        df.min()
        logger.info("LL: df.min() \n%s", df.min())
        df.max()
        logger.info("LL: df.max() \n%s", df.max())
        df.median
        logger.info("LL: df.median \n%s", df.median())
        df.id
        logger.info("LL: df.id \n%s", df.id)
        df.idxmin()
        logger.info("LL: df.idxmin() \n%s", df.idxmin())
        df.idxmax()
        logger.info("LL: df.idxmax() \n%s", df.idxmax())
        logger.info("LL: df.sort_values(columns[1]).head(10)\n%s", df.sort_values(columns[1]).head(10))
        
    def datainfo_matrix_features(self, *args, **kwargs):
        df = self.dataframe
        column_names = []    
        numerical_features = []
        categorical_features = []
        categorical_features_nonint = []
        for column in df.columns:
            # logger.info("amount of unique column \n %s is : %s",column,df[column].unique().size)
            column_names.append(column)
            if(df[column].unique().size<=11):
                # logger.info("LL: column categorical %s", column)
                categorical_features.append(column)
            else:
                # logger.info("LL: column numerical %s", column)
                numerical_features.append(column)
            if(df.dtypes[column]!=np.int64 and df.dtypes[column]!=np.int32):
                # logger.info("LL: no int64 or int32")
                categorical_features_nonint.append(column)
                # df_example = df[column].fillna(0)
            else:
                # logger.info("LL: yes int64 or int32")
                pass
        logger.info("LL: numerical_features \n%s", numerical_features)
        logger.info("LL: categorical_features \n%s", categorical_features)
        logger.info("LL: categorical_features_nonint \n%s", categorical_features_nonint)
        self.column_names = column_names
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.categorical_features_nonint = categorical_features_nonint

##      Basic plotting in Seaborn
    def datainfo_plot_sns_base(self, *args, **kwargs):
        df = self.dataframe
        sns.set()
    
##      Popular types of plots for visualizing data
##           Scatter plots        
    def datainfo_plot_sns_scatter(self, *args, **kwargs):
        df = self.dataframe
        # sns.scatterplot(x='AGE', y='BILL_AMT2', data=df)
        # sns.scatterplot(x='AGE', y='BILL_AMT2', data=df,hue='EDUCATION',fit_reg=False, palette='bright',markers=['o','x','v','+'])
        
##           Histograms    
    def datainfo_plot_sns_histdist(self, *args, **kwargs):
        df = self.dataframe
        sns.distplot(df['BILL_AMT2'], bins=15)
        
##           Jointplots
    def datainfo_plot_sns_join(self, *args, **kwargs):
        df = self.dataframe
        sns.jointplot(x='AGE', y='BILL_AMT2', data=df, kind='scatter', marginal_kws=dict(bins=15))
        sns.jointplot(x='AGE', y='BILL_AMT2', data=df, kind='kde', marginal_kws=dict(bins=15))
##           Violin plots
    def datainfo_plot_sns_violin(self, *args, **kwargs):
        df = self.dataframe
        sns.violinplot(x='AGE',y='BILL_AMT2', data=df)
        # sns.violinplot(x='Default',y='BILL_AMT2', data=df)

##           Pairplots
    def datainfo_plot_sns_pair(self, *args, **kwargs):
        df = self.dataframe
        sns.pairplot(data=df)
        vars_to_plot = ['CRIM', 'AGE', 'DIS', 'LSTAT', 'MEDV']
        sns.pairplot(data=df, vars=vars_to_plot)        

##      Summary



    
    ##  Chapter 4: Cleaning and Readying Data for Analysis
    ##      The scikit-learn transformer API
    ##      Cleaning input data
    ##           Missing values
    def dataprep_check_features(self, *args, ** kwargs):
        df = self.dataframe
        
        column_names = []
        numerical_features = []
        categorical_features = []
        categorical_features_nonint = []
        
        for column in df.columns:
            # logger.info("amount of unique column \n %s is : %s",column,df[column].unique().size)
            column_names.append(column)
            if(df[column].unique().size<=11):
                logger.info("LL: column categorical %s", column)
                categorical_features.append(column)
            else:
                logger.info("LL: column numerical %s", column)
                numerical_features.append(column)
            logger.info("LL: df.dtypes[column] \n%s", df.dtypes[column])
            if(df.dtypes[column]!=np.int64 and df.dtypes[column]!=np.int32):
                logger.info("LL: no int64 or int32")
                categorical_features_nonint.append(column)
                # df_example = df[column].fillna(0)
            else:
                logger.info("LL: yes int64 or int32")
            # categorical_features_nonint = df.dtypes[df.dtypes==np.integer]
        self.column_names = column_names
        self.numerical_features = numerical_features
        self.categorical_features = numerical_features
        self.categorical_features_nonint = numerical_features

        # pp = dataprep_check_features(pp)

        logger.info("LL: column_names \n%s", self.column_names)
        logger.info("LL: numerical_features \n%s", self.numerical_features)
        logger.info("LL: categorical_features \n%s", self.categorical_features)
        logger.info("LL: categorical_features_nonint \n%s", self.categorical_features_nonint)

    #-------------Check Correlation and Covariance

    
    def dataprep_check_stats(self, *args,  ** kwargs):
        df = self.dataframe
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

    # dataprep_check_stats(pp)

    #-------------Check Correlation and Covariance

    ##               Finding and removing missing values
    def dataprep_find_misval(self, *args,  ** kwargs):
        df = self.dataframe
        fill_na = True
        fill_na_dict = {}
        fill_na_dict[colinf.age] = 0
        drop_rows_na = False
        drop_cols_na = False
        cols_with_missing_value = []
        df_fill_na = DataFrame()
        df_rmv_rows_na = DataFrame()
        df_rmv_cols_na = DataFrame()
        for column in df.columns:
            logger.info("LL: column %s datatype %s is having null : %s about %s and having NA : %s , about %s ", column, df.dtypes[column], df[column].isnull().values.any(), df[column].isnull().values.sum(),df[column].isna().values.any(), df[column].isna().values.sum())
            if(df.dtypes[column], df[column].isnull().values.any() and df.dtypes[column], df[column].isna().values.any()):
                cols_with_missing_value.append(column)
            if column in fill_na_dict and fill_na == True:
                df_fill_na = df
                df_fill_na[column].fillna(fill_na_dict[column])
        if drop_rows_na==True:
            df_rmv_rows_na = df
            df_rmv_rows_na.dropna(axis=0)
        if drop_cols_na==True:
            df_rmv_cols_na = df
            df_rmv_cols_na.dropna(axis=1)

        logger.info("LL: cols_with_missing_value \n%s", cols_with_missing_value)
        logger.info("LL: df_fill_na.head \n%s", df_fill_na.shape)
        logger.info("LL: df_rmv_rows_na.head \n%s", df_rmv_rows_na.shape)
        logger.info("LL: df_rmv_cols_na.head \n%s", df_rmv_cols_na.shape)

    # dataprep_find_misval(pp)
    
    #alternatively use missingno
    #-----------missingno

    ##               Imputing to replace the missing values

    def dataprep_imputation(self, *args,  ** kwargs):
        df = self.dataframe
        # impute nan with new values
        # missing_values = form of missing values in your data. (For example nan, 0, or "n/a".
        # strategy = how to impute (choices are "mean", "median", "most frequent", and "constant". 
        # If you pass strategy=constant, then you can use the optional argument fill_value to pass your constant.
        cols_to_impute = []
        impute_values = np.nan # nan, 0, or "n/a"
        impute_strategy = "mean" # "mean", "median", "most frequent", and "constant".

        #--------------change this
        cols_to_impute = df.columns
        #--------------change this
        logger.info("LL: df.shape \n%s", df.shape)
        logger.info("LL: df.head \n%s", df.head())
        imputer = SimpleImputer(missing_values=impute_values, strategy=impute_strategy)
        out_imp = imputer.fit_transform(df[cols_to_impute])
        df_new_imputer = pd.DataFrame(data = out_imp, columns = cols_to_impute)
        logger.info("LL: df_new_imputer.shape \n%s", df_new_imputer.shape)
        logger.info("LL: df_new_imputer.head \n%s", df_new_imputer.head())
        df_new_imputer = pd.concat([df_new_imputer, df[[self.target_col_name]]], axis = 1)
        logger.info("LL: df_new_imputer.shape \n%s", df_new_imputer.shape)
        logger.info("LL: df_new_imputer.head \n%s", df_new_imputer.head())


    ##           Feature scaling
    ##               Normalization
    
    def dataprep_scaling_normaliz(self, *args,  ** kwargs):
        df = self.dataframe
        cols_to_minmaxscale = []

        #--------------change this
        cols_to_minmaxscale = df.columns
        #--------------change this
        
        logger.info("LL: df.shape \n%s", df.shape)
        logger.info("LL: df.head \n%s", df.head())
        minmax_scaler = MinMaxScaler()
        out_scaled_minmax = minmax_scaler.fit_transform(df[cols_to_minmaxscale])
        df_new_minmax_scaler = pd.DataFrame(data = out_scaled_minmax, columns = cols_to_minmaxscale)
        logger.info("LL: df_new_minmax_scaler.shape \n%s", df_new_minmax_scaler.shape)
        logger.info("LL: df_new_minmax_scaler.head \n%s", df_new_minmax_scaler.head())
        df_new_minmax_scaler = pd.concat([df_new_minmax_scaler, df[[self.target_col_name]]], axis = 1)
        logger.info("LL: df_new_minmax_scaler.shape \n%s", df_new_minmax_scaler.shape)
        logger.info("LL: df_new_minmax_scaler.head \n%s", df_new_minmax_scaler.head())

    # dataprep_scaling_normaliz(pp)
    ##               Standardization
    
    #--------------change this

    def dataprep_scaling_standard(self, *args,  ** kwargs):
        df = self.dataframe
        #--------------change this
        cols_to_standardscale = df.columns
        #--------------change this

        logger.info("LL: df.shape \n%s", df.shape)
        logger.info("LL: df.head \n%s", df.head())
        standard_scaler = StandardScaler()
        out_scaled_standard = standard_scaler.fit_transform(df[cols_to_standardscale])
        logger.info("LL: out_scaled_standard \n%s", out_scaled_standard)
        # df_new_standard_scaler = pd.DataFrame(data = out_scaled_standard, columns = out_scaled_standard)
        # logger.info("LL: df_new_standard_scaler.shape \n%s", df_new_standard_scaler.shape)
        # logger.info("LL: df_new_standard_scaler.head \n%s", df_new_standard_scaler.head())
        # df_new_minmax_scaler = pd.concat([df_new_minmax_scaler, df[[self.target_col_name]]], axis = 1)
        # logger.info("LL: df_new_standard_scaler.shape \n%s", df_new_standard_scaler.shape)
        # logger.info("LL: df_new_standard_scaler.head \n%s", df_new_standard_scaler.head())

    # dataprep_scaling_standard(pp)

    ##           Handling categorical data
    ##               Ordinal encoding

    def dataprep_encd_ordinal(self, *args,  ** kwargs):
        df = self.dataframe
        ##               Ordinal encoding
        
        categorical_features_nonint = []
        #------------change this
        categorical_features_nonint = self.categorical_features_nonint
        #------------change this

        logger.info("LL: df.shape  \n%s",df.shape)
        logger.info("LL: df.head()  \n%s",df.head())

        enc_ordinal = OrdinalEncoder()
        out_enc_ord_catg_feat_nonint = enc_ordinal.fit_transform(df[categorical_features_nonint])
        logger.info("LL: out_enc categories \n%s",enc_ordinal.categories_)
        logger.info("LL: out_enc  \n%s",out_enc_ord_catg_feat_nonint)
        df[categorical_features_nonint] = out_enc_ord_catg_feat_nonint
        logger.info("LL: df.shape  \n%s",df.shape)
        logger.info("LL: df.head()  \n%s",df.head())

    # dataprep_encd_ordinal(pp)

    ##               One-hot encoding

    def dataprep_encd_onehot(self, *args,  ** kwargs):
        df = self.dataframe

        categorical_features_nonint = []
        #------------change this
        categorical_features_nonint = self.categorical_features_nonint
        #------------change this

        logger.info("LL: df.shape  \n%s",df.shape)
        logger.info("LL: df.head()  \n%s",df.head())


        enc_onehot = OneHotEncoder(sparse=False)
        out_enc_onehot_catg_feat_nonint = enc_onehot.fit_transform(df[categorical_features_nonint])
        new_cols_onehot_catg_feat_nonint = enc_onehot.get_feature_names(categorical_features_nonint).tolist()
        logger.info("LL: new_cols \n%s", new_cols_onehot_catg_feat_nonint)
        df_enc_enc_onehot = pd.DataFrame(data = out_enc_onehot_catg_feat_nonint, columns = new_cols_onehot_catg_feat_nonint)
        df_enc_enc_onehot.index = df.index

        # drop original columns and concat new encoded columns
        df_test = df
        df_test.drop(categorical_features_nonint, axis=1, inplace=True)
        df_test = pd.concat([df_test, df_enc_enc_onehot], axis = 1)
        logger.info("LL: df_test.columns \n%s",df_test.columns)

        logger.info("LL: df_test.shape  \n%s",df_test.shape)
        logger.info("LL: df_new.head()  \n%s",df_test.head())

        logger.info("LL: df.shape  \n%s",df.shape)
        logger.info("LL: df.head()  \n%s",df.head())

    # data_preparation_encoding_onehot(pp)


    ##               Label encoding

    def dataprep_encd_label(self, *args,  ** kwargs):
        df = self.dataframe
        
        enc_label = LabelEncoder()
        # --------- change this--
        # out_enc_label = enc_label.fit_transform(categorical_features_nonint)
        # --------- change this--
        out_enc_label = enc_label.fit_transform(["blue", "red", "blue", "green", "red", "red"])
        logger.info("LL: out_enc_label \n%s ",out_enc_label)

    # data_preparation_encoding_label()

    # %%
    ##      High-dimensional data
    ##      Dimension reduction
    ##           Feature selection
    ##               Feature filtering

    ##                    The variance threshold

    def dataprep_dimreduct_var_treshold(self, *args,  ** kwargs):
        df = self.dataframe
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
        df_sel = pd.concat([df_sel, df[[self.target_col_name]]], axis = 1)

        logger.info("LL: df_sel.shape \n%s", df_sel.shape)
        logger.info("LL: df_sel.head() \n%s", df_sel.head())

    # data_prep_dimreduct_variance_treshold()

    ##                    The correlation coefficient

    #-----------change this

    def dataprep_dimenreduct_corel_coef(self, *args,  ** kwargs):
        df = self.dataframe
        #-----------change this
        treshold_val = 0.6
        #-----------change this

        cor = df.corr()
        sns.heatmap(cor, annot=False, cmap=plt.cm.Blues)
        logger.info("LL: plt.show() \n%s", plt.show())

        # get correlation values with target variable
        cor_target = abs(cor[self.target_col_name])
        logger.info("LL: cor_target \n%s", cor_target)

        #For demonstration purposes, we will choose 0.6 as the threshold and then filter. From the output, you should expect columns 5 and 12 (0.69 and 0.74) to be selected:
        selected_cols = cor_target[cor_target>treshold_val]
        logger.info("LL: selected columns, correlation with target > %s", treshold_val)
        logger.info("LL: selected_cols \n%s", selected_cols)
        # filter in the selected features
        df_sel = df[selected_cols.index]
        logger.info("LL: df_sel.head() \n%s,", df_sel.head())
        logger.info("LL: df_sel.head() \n%s,", df_sel.shape)

    # datprep_dimenreduct_corre_coeff()

    ##               Wrapper methods
    ##                    Sequential feature selection

    def dataprep_featselect_sequential(self, *args,  ** kwargs):
        df = self.dataframe
        

        #----------change this
        kernel_mode ="linear"
        n_features_to_select = 2
        n_step = 1
        #----------change this

        cols = df.columns
        X = df[cols]
        y = df[self.target_col_name]

        svc = SVC(kernel=kernel_mode, C=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_features_to_select, step=n_step)
        rfe.fit(X, y)

        logger.info("LL: cols \n%s", cols)
        logger.info("LL: rfe.ranking_")

    # datprep_featsel_sequential()

    ##           Transformation

    ##               PCA Principal Component Analysis

    def dataprep_transform_pca(self, *args,  ** kwargs):
        df = self.dataframe
        
        pca = PCA(n_components=2)

        # fit and transform using 2 input dimensions
        out_pca = pca.fit_transform(df[['petal length in cm','petal width in cm',]])

        # create pca output dataframe and add label column "species" 
        df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2'])
        df_pca = pd.concat([df_pca, df[['species']]], axis = 1)

        # plot scatter of pca data
        sns.lmplot(x='pca1', y='pca2', hue='species', data=df_pca, fit_reg=False)

        logger.info("LL: pca.explained_variance_ratio_ \n%s", pca.explained_variance_ratio_)
        sns.violinplot(x='species',y='pca1', data=df_pca)
        sns.violinplot(x='species',y='pca2', data=df_pca)

    # datprep_transform_pca()

    ##               LDA-LinearDiscriminantAnalysis

    def dataprep_transform_lda(self, *args,  ** kwargs):
        df = self.dataframe
        lda = LinearDiscriminantAnalysis(n_components=2)

        # fit and transform using 2 input dimensions
        cols = ['sepal length in cm','sepal width in cm']
        lda.fit(df[cols], df['species'])
        out_lda = lda.transform(df[cols])

        # create lda output dataframe and add label column "species"
        df_lda = pd.DataFrame(data = out_lda, columns = ['lda1', 'lda2'])
        df_lda = pd.concat([df_lda, df[['species']]], axis = 1)
        sns.lmplot(x="lda1", y="lda2", hue="species", data=df_lda, fit_reg=False)

    # dataprep_transform_lda()

    ##      Summary

    ##  Chapter 5: Grouping and Clustering Data
    ##      Introducing clustering concepts
    ##           Location of the group
    ##               Euclidean space (centroids)
    ##               Non-Euclidean space (medioids)
    ##           Similarity
    ##               Euclidean space
    ##                    The Euclidean distance
    ##                    The Manhattan distance
    ##                    Maximum distance
    ##               Non-Euclidean space
    ##                    The cosine distance
    ##                    The Jaccard distance
    ##           Termination condition
    ##               With known number of groupings
    ##               Without known number of groupings
    ##               Quality score and silhouette score
    ##      Clustering methods
    # import datasets module from Sci-kit learn

    # function to create data for clustering examples
    def get_blobs():
        # build blobs for demonstration
        n_samples = 1500
        blobs = make_blobs(n_samples=n_samples,centers=5,cluster_std=[3.0, 0.9, 1.9, 1.9, 1.3],random_state=51)
        
        # create a Pandas dataframe for the data
        df = pd.DataFrame(blobs[0], columns=['Feature_1', 'Feature_2'])
        df.index.name = 'record'
        print(df.head())
        sns.lmplot(x='Feature_2', y='Feature_1', data=df, fit_reg=False)

        return df
    

    # plot scatter of blob set

    ##           Means separation

    ##               K-means

    def datagroup_clust_kmeans(self, *args,  ** kwargs):
        df = self.dataframe
        df = get_blobs()

        # import module and instantiate K-means object
        clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

        # fit to input data
        clus.fit(df)

        # get cluster assignments of input data and print first five labels
        df['K-means Cluster Labels'] = clus.labels_
        print(df['K-means Cluster Labels'][:5].tolist())

        df.head()
        sns.lmplot(x='Feature_2', y='Feature_1', hue="K-means Cluster Labels", data=df, fit_reg=False)

    # datagroup_clust_kmeans()

    ##                    Finding k

    def datagroup_clust_kmeansfinding(self, *args,  ** kwargs):
        df = self.dataframe
        df = get_blobs()

        # find best value for k using silhouette score
        # import metrics module

        # create list of k values to test and then use for loop
        n_clusters = [2,3,4,5,6,7,8]
        for k in n_clusters:
            datgroup_clustering_kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
            cluster_labels = datgroup_clustering_kmeans.predict(df)
            S = metrics.silhouette_score(df, cluster_labels)
            print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))

    # datagroup_clust_kmeansfinding()

    ##                    K-means++

    def datagroup_clust_kmeansplus(self, *args,  ** kwargs):
        df = self.dataframe
        clus = KMeans(n_clusters=5, init='k-means++', tol=0.004, max_iter=300)

    # datagroup_clust_kmeansplus()

    ##                    Mini batch K-means

    def datagroup_clust_kmeansminibatch(self, *args,  ** kwargs):
        df = self.dataframe
        clus = MiniBatchKMeans(n_clusters=5, batch_size=50,tol=0.004, max_iter=300)

    # datagroup_clust_kmeansminibatch()

    ##           Hierarchical clustering


    def datagroup_clust_hierarc(self, *args,  ** kwargs):
        df = self.dataframe
        df = get_blobs()

        # import module and instantiate HCA object
        clus = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

        # fit to input data
        clus.fit(df)

        # get cluster assignments
        df['HCA Cluster Labels'] = clus.labels_

        sns.lmplot(x='Feature_2', y='Feature_1',hue="HCA Cluster Labels", data=df, fit_reg=False)

    # datagroup_clust_hierarc()

    ##               Reuse the dendrogram to find number of clusters


    def datagroup_clust_hierarc_dendogram(self, *args,  ** kwargs):
        df = self.dataframe

        # generate blob example dataset
        df = get_blobs()

        # import module and instantiate HCA object

        # create list of k values to test and then use for loop
        n_clusters = [2,3,4,5,6,7,8]
        for num in n_clusters:
            HCA = AgglomerativeClustering(n_clusters=num,affinity='euclidean', linkage='ward',memory='./model_storage/dendrogram',compute_full_tree=True)
            cluster_labels= HCA.fit_predict(df)
            S = metrics.silhouette_score(df, cluster_labels)
            print("n_clusters = {:d}, silhouette score {:1f}".format(num, S))
        return df

    # df = datagroup_clust_hierarc_dendogram()

    ##               Plot dendrogram

    def datagroup_clust_hierarc_dendogram_plot(self, *args,  ** kwargs):
        df = self.dataframe

        # Calculate the distance between each sample
        Z = hierarchy.linkage(df, 'ward') 

        # Plot with Custom leaves (scroll down in console to see plot)
        hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)

    # datagroup_clust_hierarc_dendogram_plot()

    ##           Density clustering
    # generate blob example dataset

    def datagroup_clust_density(self, *args,  ** kwargs):
        df = self.dataframe
        # generate blob example dataset
        df = get_blobs()

        # import module and instantiate DBSCAN object
        clus = DBSCAN(eps=0.9, min_samples=5, metric='euclidean')

        # fit to input data
        clus.fit(df)

        # get cluster assignments
        df['DBSCAN Cluster Labels'] = clus.labels_

        sns.lmplot(x='Feature_2', y='Feature_1',hue="DBSCAN Cluster Labels", data=df, fit_reg=False)

    # datagroup_clust_density()

    ##           Spectral clustering
    # generate blob example dataset

    def datagroup_clust_spectral(self, *args,  ** kwargs):
        df = self.dataframe
        df = get_blobs()

        # import module and instantiate spectral clustering object
        clus = SpectralClustering(n_clusters=5, random_state=42,assign_labels='kmeans', n_init=10, affinity='nearest_neighbors', n_neighbors=10)

        # fit to input data
        clus.fit(df)

        # get cluster assignments
        df['Spectral Cluster Labels'] = clus.labels_

        sns.lmplot(x='Feature_2', y='Feature_1',hue="Spectral Cluster Labels", data=df, fit_reg=False)

    # datagroup_clust_spectral()
    ##      Summary

    ##  Chapter 6: Prediction with Regression and Classification
    ##      Scikit-learn Estimator API
    ##      Introducing prediction concepts
    ##           Prediction nomenclature
    ##           Mathematical machinery
    ##               Loss function
    ##               Gradient descent
    ##           Fit quality regimes
    ##      Regression
    ##           Metrics of regression model prediction

    ##           Regression example dataset

    # function to get boston dataset with training and test sets
    def datapred_getdata_traintest(self, *args,  ** kwargs):
        df = self.dataframe
        
        # load the boston dataset
        dataset = load_boston()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['MEDV'] = dataset.target
        df.index.name = 'record'
        # split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'MEDV'],df['MEDV'], test_size=.33, random_state=42)

        return [X_train, X_test, y_train, y_test]

    ##           Linear regression

    def datapred_regression_linear(self, *args,  ** kwargs):
        df = self.dataframe
        

        # get moon dataset
        X_train, X_test, y_train, y_test = datapred_getdata_traintest()

        # instantiate regression object and fit to training data
        clf = LinearRegression()
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred) 
        print('r2 score is = ' + str(r2))

    # datapred_regression_linear()

    ##           Extension to multivariate form
    ##           Regularization with penalized regression
    ##               Regularization penalties

    ### Lasso Regression ###


    def datapred_regression_lasso(self, *args,  ** kwargs):
        df = self.dataframe
        

        # get moon dataset
        X_train, X_test, y_train, y_test = datapred_getdata_traintest()

        # instantiate classifier object and fit to training data
        clf = Lasso(alpha=0.3)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred) 
        print('r2 score is = ' + str(r2))

    # datapred_regression_lasso()

    ### Ridge Regression ###
    # import modules

    def datapred_regresion_ridge(self, *args,  ** kwargs):
        df = self.dataframe
        # import modules
        

        # get moon dataset
        X_train, X_test, y_train, y_test = datapred_getdata_traintest()

        # instantiate classifier object and fit to training data
        clf = Ridge(alpha=0.3)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        r2 = r2_score(y_test, y_pred) 
        print('r2 score is = ' + str(r2))

    # datapred_regresion_ridge()

    ##      Classification
    ##           Classification example dataset

    # function to get toy moon dataset with training and test sets
    def get_moon_data(self, *args,  ** kwargs):
        df = self.dataframe
        # make blobs and split into train and test sets
        X, y = make_moons(n_samples=150, noise=0.4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
        return [X_train, X_test, y_train, y_test]

    # %% [markdown]
    ##           Metrics of classification model prediction
    ##           Multi-class classification
    ##               One-versus-all
    ##               One-versus-one

    ##           Logistic regression
    ### Logistic Regressiin ###
    # import modules

    def datapred_regression_logistic(self, *args,  ** kwargs):
        df = self.dataframe
        # import modules
    

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = LogisticRegression(solver='lbfgs')
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))
        return y_pred, y_test

    # y_pred, y_test = datapred_regression_logistic()

    ### plot confusion matrix ###-------------------------

    def datapred_regression_logistic_confmatrix(self, *args,  ** kwargs):
        df = self.dataframe
        

        # Creates a confusion matrix
        cm = confusion_matrix(y_pred, y_test)

        # create df and add class names
        labels = ['top crescent', 'bottom cresent']
        df_cm = pd.DataFrame(cm,index = labels, columns = labels)

        # plot figure
        plt.figure(figsize=(5.5,4))
        sns.heatmap(df_cm, cmap="GnBu", annot=True)

        #add titles and labels for the axes
        plt.title('Logistic Regression \nF1 Score:{0:.3f}'.format(f1_score(y_test, y_pred)))
        plt.ylabel('Prediction')
        plt.xlabel('Actual Class')
        plt.show()

    # datapred_regression_logistic_confmatrix()


    ##               Regularized logistic regression
    def datapred_regression_logistic_regular(self, *args,  ** kwargs):
        df = self.dataframe
        # import modules
        

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = LogisticRegression(solver='lbfgs', penalty='l2', C=0.5)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))
        return y_pred, y_test
    ##           Support vector machines

    ##               Soft-margin with C
    ### SVM Classification ###
    # import modules

    def datapred_classif_svm(self, *args,  ** kwargs):
        df = self.dataframe
        

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = SVC(kernel="linear", C=0.5)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))
        return X_train, y_train, X_test, y_test

    # X_train, y_train, X_test, y_test = datapred_classif_svm()


    ##               The kernel trick
    ### SVM with Gaussian Kernel Classification ###
    # instantiate classifier object and fit to training data
    def datapred_classif_svm_gaussian_kernel(self, *args,  ** kwargs):
        df = self.dataframe
        clf = SVC(gamma=2, C=1)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))

    ##           Tree-based classification
    ##               Decision trees

    ##                    Node splitting with Gini
    ### Decision Tree Classification ###
    # import modules

    def datapred_classif_decisiontree(self, *args,  ** kwargs):
        df = self.dataframe
        # import modules
        

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))

    # datapred_classif_decisiontree()

    ##               Random forest

    ##                    Avoid overfitting and speed up the fits
    ### Random Forest Classification ###
    # import modules

    def datapred_classif_randomforest(self, *args,  ** kwargs):
        df = self.dataframe
        # import modules
        

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = RandomForestClassifier(max_depth=4, n_estimators=4,max_features='sqrt', random_state=42)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))
        return X_train, y_train

    # X_train, y_train = datapred_classif_randomforest()

    ##                    Built-in validation with bagging
    ### Use OOB for Validation Set ###
    # instantiate classifier object and fit to training data
    def datapred_classif_randomforest_bagging(self, *args,  ** kwargs):
        df = self.dataframe
    

        # get moon dataset
        X_train, X_test, y_train, y_test = get_moon_data()

        # instantiate classifier object and fit to training data
        clf = RandomForestClassifier(max_depth=4, n_estimators=10,max_features='sqrt', random_state=42,oob_score=True)
        clf.fit(X_train, y_train)

        # predict on test set and score the predictions against y_test
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred) 
        print('f1 score is = ' + str(f1))
        # predict on test set and score the predictions with OOB
        oob_score = clf.oob_score_
        print('OOB score is = ' + str(oob_score))
        return X_train, y_train

    # X_train, y_train = datapred_classif_randomforest_bagging()


    ##      Tuning a prediction model
    ##           Cross-validation

    ##               Introduction of the validation set
    ### Cross Validation ###
    # load iris and create X and y

    def datapred_validat_cross(self, *args,  ** kwargs):
        df = self.dataframe
        # load iris and create X and y
        dataset = load_iris()
        X,y = dataset.data, dataset.target

        # import module

        # create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

        # create validation set from training set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.33)

    # datapred_validat_cross()

    ##               Multiple validation sets with k-fold method
    ### k-fold Cross Validation ###
    # load iris and create X and y

    def datapred_validat_cross_kfold(self, *args,  ** kwargs):
        df = self.dataframe
        ### k-fold Cross Validation ###
        # load iris and create X and y
        dataset = load_iris()
        X,y = dataset.data, dataset.target

        # create train and test sets
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.33)

        # instantiate classifier object and pass to cross_val_score function
        clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
        print(scores)

    # datapred_validat_cross_kfold()

    ##           Grid search for hyperparameter tuning
    ### Grid Search with k-fold Cross-validation ###
    # load iris and create X and y

    def datapred_validat_cross_kfold_gridsearch(self, *args,  ** kwargs):
        df = self.dataframe
        dataset = load_iris()
        X,y = dataset.data, dataset.target

        
        # create train and test sets
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.33)

        # instantiate svc and gridsearch object and fit
        parameters = {'kernel':['linear', 'rbf'], 'C':[1, 5, 10]}
        svc = SVC(gamma='auto')
        clf = GridSearchCV(svc, parameters, cv=5, scoring='f1_macro')
        clf.fit(X_train, y_train)

        # print best scoring classifier
        print('Best score is = ' + str(clf.best_score_))
        print('Best parameters are = ' + str(clf.best_params_))
        ##      Summary
        y_pred = clf.predict(X_test)

    # datapred_validat_cross_kfold_gridsearch()

    ##  Chapter 7: Advanced Topics - Building a Data Processing Pipeline and Deploying It
    ##      Pipelining your analysis
    ##           Scikit-learn's pipeline object
    # load iris and create X and y

    def datapipeln_loadxy(self, *args,  ** kwargs):
        df = self.dataframe
        dataset = load_iris()
        X,y = dataset.data, dataset.target
        return X, y

    # X, y = datapipeln_loadxy()

    # import modules 

    def datapipeln_scikitlearn(self, *args,  ** kwargs):
        df = self.dataframe
        
        # create train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

        pca = PCA()
        logistic = LogisticRegression(solver='liblinear', multi_class='ovr', C=1.5)

        # instantiate a pipeline and add steps to the pipeline
        pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

        # print list of steps with names
        print(pipe.steps[0])
        param_grid = {
        'pca__n_components': [2, 3, 4],
        'logistic__C': [0.5, 1, 5, 10],
        }

        # instantiate the grid search object and pass the pipe and param_grid
        model = GridSearchCV(pipe, param_grid, iid=False, cv=5, return_train_score=False)

        # fit entire pipeline using grid serach and 5-fold cross validation
        model.fit(X_train, y_train)
        print("Best parameter (CV score=%0.3f):" % model.best_score_)
        print(model.best_params_)
        y_pred = model.predict(X_test)
        return model

    # model = datapipeln_scikitlearn()

    # use the resulting pipeline to predict on new data
    
    ##      Deploying the model
    ##           Serializing a model and storing with the pickle module
    ### Store Model for Later with Pickle ###
    def datapipeln_storemodel_pickle(self, *args,  ** kwargs):
        df = self.dataframe
        # import module
        import pickle

        # save the pipeline model to disk
        pickle.dump(model, open('../model_storage/model.pkl', 'wb'))


    ##           Loading a serialized model and predicting
    def datapipeln_loadmodel_pickle(self, *args,  ** kwargs):
        df = self.dataframe

        # load the pipeline model from disk and deserialize
        model_load = pickle.load(open('../model_storage/model.pkl', 'rb'))

        # use the loaded pipeline model to predict on new data
        y_pred = model_load.predict(X_test)

    ##      Python-specific deployment concerns
    ##      Summary
    ##  Other Books You May Enjoy

    def data_preparation(self, *args, **kwargs):
        logger.info("LL: -------------------------------------")
        logger.info("LL: data_preparation start")
        logger.info("LL: data_preparation end")
        logger.info("LL: -------------------------------------")

    def data_preparation_encode_categorical_nonint(self, *args, **kwargs):
        logger.info("LL: -------------------------------------------")
        logger.info("LL: data_preparation_encode_categorical_nonint start")
        logger.info("LL: data_preparation_encode_categorical_nonint end")
        logger.info("LL: -------------------------------------------")

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

