import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

data_excel = '/home/pranjal/Desktop/cred/Data Set.xlsx'
cred = pd.read_excel(data_excel,engine="openpyxl")

cred.corr().to_csv('corrilation_mat.csv')

# Selecting features based on there corelation with target variables
feat = [
        'card',
        'card2',
        'age',
        'income',
        'lninc',
        'inccat',
        'debtinc',
        'lncreddebt',
        'othdebt',
        'lnothdebt',
        'default',
        'carcatvalue',
        'cardtype',
        'card2type',
        'active',
        'cardspent',
        'card2spent']

cred = cred[feat]

# deviding data into continous and catagorical to process seperatly 
cat_data = cred[['active','card2type','cardtype','default','inccat','card', 'card2']]
cont_data = cred[cred.columns.difference(cat_data.columns.to_list()+['cardspent','card2spent'])]
 
# outliar and missing value treatement 
cont_data = cont_data.apply(lambda x : x.clip(upper=x.quantile(0.99), lower=x.quantile(0.01)))
cont_data = cont_data.apply(lambda x : x.fillna(x.mean()))

# creating dummies of catagorical var
cat_data = pd.get_dummies(cat_data.astype('object'),drop_first=True)

# creating new dataframe after processing data
new_cred = pd.concat([cont_data,cat_data],axis=1)
new_cred['cardspent'] = cred['cardspent'] + cred['card2spent']
new_cred['ln_cardspent'] = np.log(new_cred['cardspent'])
 
# feature selection using f_regression and vif
f_reg_df = pd.DataFrame()
f_reg = f_regression(new_cred[new_cred.columns.difference(['cardspent','ln_cardspent'])],new_cred['ln_cardspent'])
f_reg_df["Columns"] = new_cred.columns.difference(['cardspent','ln_cardspent'])
f_reg_df["Stats"] = f_reg[0]
f_reg_df["P_value"] = f_reg[1]
f_reg_df.sort_values("P_value")

vif = pd.DataFrame()
vif["columns"] = new_cred.columns
vif["vif"] = [variance_inflation_factor(new_cred.values,i) for i in range(new_cred.shape[1])]
vif

# removing features based on information gained from f_regression and vif
_new_cred = new_cred.drop([
    'income','inccat_2',
    'inccat_3','inccat_4',
    'inccat_5','active_1',
    'card2type_2','card2type_3',
    'cardtype_3','cardtype_4',
    'lnothdebt','default_1'],axis=1)
 

# dividing data into test and train 
train, test = train_test_split(_new_cred, test_size=0.3, random_state=1234)
train = train.drop(['cardspent'],axis=1)
test = test.drop(['cardspent'],axis=1)

# stats model
formula = 'ln_cardspent ~ ' + ' + '.join(_new_cred.columns.difference(['ln_cardspent','cardspent'])) 
lm = smf.ols(formula,train).fit()
print(lm.summary())

# lm summary:
#___________________________________________
# R-squared:           |            0.326
# Adj. R-squared:      |            0.323
#______________________|____________________

# R-squared should be as large as possible but not so that the model is overfitted
# Adj. R-squared should be as close to R-squared as possible 
 
# dropping influential datapoints to increase the accuracy of the model
train.drop(np.abs(lm.resid).sort_values()[-400:].index,inplace=True)
 
lm = smf.ols(formula,train).fit()
print(lm.summary())

# lm summary:
#___________________________________________
# R-squared:           |            0.436
# Adj. R-squared:      |            0.433
#______________________|____________________

# mean_squared_error (for train)
np.sqrt(mean_squared_error(np.exp(train['ln_cardspent']),np.exp(lm.predict(train)))) 
# 211.8670606338464

# mean_squared_error (for train)
np.sqrt(mean_squared_error(np.exp(test['ln_cardspent']),np.exp(lm.predict(test)))) 
# 283.83144183796105

