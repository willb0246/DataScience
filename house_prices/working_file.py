# load all packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
warnings.filterwarnings('ignore')


'''
# here is the columns output
Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'SalePrice'],
      dtype='object')
'''

df_train = pd.read_csv('initial_train.csv')
df_train.dropna()
pd.set_option('display.max_columns', None)

# drop columns that I think are going to be cofounders
df_train = df_train.drop(['SaleType', 'SaleCondition'], axis=1)

'''
# correlation heat map
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
'''

# this script outputs a list of int64 columns, then runs for loop. For every var in list it correlates the var with SalePrice
# then it prints any correlations that have an absolute value over 0.3
y = list(df_train.select_dtypes(include=['int64']))
yy = []
correlated_vars = []
for i in y:
    if i == 'SalePrice':
        pass
    else:
        yy.append(i)
        int_var = df_train[i]
        corre = int_var.corr(df_train['SalePrice'], method='pearson')
        if abs(corre) > 0.3:
            correlated_vars.append(i)
        else:
            pass

columns = []
df_train = pd.get_dummies(df_train)
df_train = df_train.dropna()
for i in df_train.columns:
    if i == 'SalePrice':
        pass
    else:
        columns.append(i)
X = df_train[columns]
y = df_train['SalePrice']

# find the NaN
NaNFinder = np.where(np.isnan(df_train))

model = LinearRegression()
rfe = RFE(model, 12)
rfe = rfe.fit(X, y)
print('Selected features: %s' % list(X.columns[rfe.support_]))

# next step is to test this model against the testing data set
#

#plt.show()

# Selected features: ['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'FullBath', 'Fireplaces', 'GarageCars']
