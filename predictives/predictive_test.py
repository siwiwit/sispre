# %%
from IPython import get_ipython
import predictive_processor
import logging

import os


def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None


logger = logging.getLogger()
# sth = logging.StreamHandler()
logger.setLevel(logging.INFO)
# sth.setFormatter(logging.Formatter(fmt='%(name) %(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'))
# logger.addHandler(sth)
# Area for experimentals variables

data_dir = 'data'
file_name = 'credit_card_default.csv'


class CreditCardColumnsInformation:
    id = 'ID'
    limit_bal = 'LIMIT_BAL'
    sex = 'SEX'
    education = 'EDUCATION'
    marriage = 'MARRIAGE'
    age = 'AGE'
    pay_1 = 'PAY_1'
    pay_2 = 'PAY_2'
    pay_3 = 'PAY_3'
    pay_4 = 'PAY_4'
    pay_5 = 'PAY_5'
    pay_6 = 'PAY_6'
    bill_amt1 = 'BILL_AMT1'
    bill_amt2 = 'BILL_AMT2'
    bill_amt3 = 'BILL_AMT3'
    bill_amt4 = 'BILL_AMT4'
    bill_amt5 = 'BILL_AMT5'
    bill_amt6 = 'BILL_AMT6'
    pay_amt1 = 'PAY_AMT1'
    pay_amt2 = 'PAY_AMT2'
    pay_amt3 = 'PAY_AMT3'
    pay_amt4 = 'PAY_AMT4'
    pay_amt5 = 'PAY_AMT5'
    pay_amt6 = 'PAY_AMT6'
    default_payment_next_month = 'default payment next month'


file_url = ''
if not is_running_from_ipython():
    abspath = os.path.abspath('.')
    file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
else:
    file_url = os.path.join('../', data_dir, file_name)
# logging.debug('abspath %s', abspath)


def main():
    # get_ipython().run_line_magic('matplotlib', 'inline')
    data_dir = 'data'
    file_name = 'credit_card_default.csv'
    # column_names = ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month
    numerical_features = []
    categorical_features = []
    categorical_features_nonint = []
    # numerical_features =['ID', 'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    # categorical_features =['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'default payment next month']

    file_url = ''
    if not is_running_from_ipython():
        abspath = os.path.abspath('.')
        file_url = 'file://'+abspath+os.path.sep+data_dir+os.path.sep+file_name
    else:
        file_url = os.path.join('../', data_dir, file_name)
    # logging.debug('abspath %s', abspath)
    # import predictive_processor

    pp = predictive_processor.PredictiveProcessor()
    # pp = PredictiveProcessor()
    pp.file_url = file_url

    # logger.info('matrix stat min \n -----------')
    # logger.info(pp.datainfo_matrix_stats().get('min'))
    from typing import Mapping, Sequence

    # colvaldict = {}
    # colvaldict[1]='highscool'
    # colvaldict[2]='junior highscool'

    colinf = CreditCardColumnsInformation
    pp.target_col_name = colinf.default_payment_next_month
    pp.idx_col_name = colinf.id
    pp.file_url = file_url
    pp.data_read_csv()
    info = pp.datapred_validat_cross()
    info = pp.datainfo_matrix_basic()
    info = pp.datainfo_basic_configindex()
    info = pp.datainfo_basic_setlabel_category_vals()
    info = pp.datainfo_matrix_uniq_per_cat_columns()
    info = pp.datainfo_matrix_valuecounts()
    info = pp.datainfo_matrix_stats()
    info = pp.datainfo_matrix_detect_feature_simple()
    info = pp.datainfo_plot_sns_base()
    info = pp.datainfo_plot_sns_scatter()
    info = pp.datainfo_plot_sns_histdist()
    info = pp.datainfo_plot_sns_join()
    info = pp.datainfo_plot_sns_violin()
    info = pp.datainfo_plot_sns_pair()

    info = pp.dataprep_check_features()
    info = pp.dataprep_find_misval()
    info = pp.dataprep_imputation()

    info = pp.dataprep_scaling_normaliz()
    info = pp.dataprep_scaling_standard()

    info = pp.dataprep_encd_ordinal()
    info = pp.dataprep_encd_onehot()
    info = pp.dataprep_encd_label()

    info = pp.dataprep_dimreduct_var_treshold()
    info = pp.dataprep_dimenreduct_corel_coef()
    info = pp.dataprep_featselect_sequential()

    info = pp.dataprep_transform_pca()
    info = pp.dataprep_transform_lda()

    info = pp.datagroup_clust_kmeans()
    info = pp.datagroup_clust_kmeansfinding()
    info = pp.datagroup_clust_kmeansplus()

    info = pp.datagroup_clust_kmeansminibatch()
    info = pp.datagroup_clust_hierarc_dendogram()
    info = pp.datagroup_clust_hierarc_dendogram_plot()

    info = pp.datagroup_clust_density()
    info = pp.datagroup_clust_spectral()

    info = pp.datapred_getdata_traintest()

    info = pp.datapred_regression_linear()
    info = pp.datapred_regression_lasso()
    info = pp.datapred_regression_ridge()

    info = pp.datapred_regression_logistic()

    info = pp.datapred_regression_logistic_confmatrix()
    info = pp.datapred_regression_logistic_regular()

    info = pp.datapred_classif_svm()
    info = pp.datapred_classif_svm_gaussian_kernel()
    info = pp.datapred_classif_decisiontree()

    info = pp.datapred_classif_randomforest()
    info = pp.datapred_classif_randomforest_bagging()

    info = pp.datapred_validat_cross()
    info = pp.datapred_validat_cross_kfold()
    info = pp.datapred_validat_cross_kfold_gridsearch()

    info = pp.datapipeln_loadxy()
    info = pp.datapipeln_scikitlearn()
    info = pp.datapipeln_storemodel_pickle()

    logger.info("info keys", info.keys())

    # logger.info("datainfo_matrix_valuecounts \n %s", pp.datainfo_matrix_valuecounts([colinf.education]))
    # newdf = pp.datainfo_basic_setlabel_category_vals([colinf.education],colvaldict)
    # logger.info("newdf head \n %s",newdf.head())

    # logger.info("newdf describe \n %s",newdf.describe().transpose())
    # logger.info("matrix stat head \n %s", pp.datainfo_matrix_stats()['head'])
    # pp.datainfo_plot_sns_scatter(colinf.age,colinf.bill_amt1,colinf.education)
    # logger.info("LL: df.sort_values(columns[1]).head(10)\n%s", pp.dataframe.sort_values(pp.dataframe.columns[1]).head(10))

    # info = pp.datapipeln_scikitlearn()


if __name__ == "__main__":
    main()
    # pass
# %%
