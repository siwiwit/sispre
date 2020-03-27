# %%
from IPython import get_ipython
import predictive_processor

import os
def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None



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

# %%
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
    #column_names = ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month
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
    import predictive_processor

    pp = predictive_processor.PredictiveProcessor()
    pp.file_url = file_url
    df = pp.data_read_csv()
    pp.df = df
    colinf = CreditCardColumnsInformation
    pp.target_col_name = colinf.default_payment_next_month
    pp.idx_col_name = colinf.id
    pp.file_url = file_url
    pp.data_read_csv()

    # pp.datainfo()
    pp.datainfo_matrix_features()
    pp.datainfo_basic_getindex()
    # pp.datainfo()
    # pp.data_preparation()
    # pp.data_analysis_exploratory()
    # pp.data_model_building()
    # pp.model_evaluation()
    # pp.model_deployment()

if __name__ == "__main__":
    main()
    # pass




# %%
