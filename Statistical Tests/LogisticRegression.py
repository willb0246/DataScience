import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# IDK what this does, so it is left out for now
#import seaborn as sns

'''
        # Code to re-code original file 
data = pd.read_csv('banking.csv')

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

out = data.to_csv('edited_banking.csv')

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list = 'var' + '_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1 = data.join(cat_list)
    data = data1

output = data.to_csv('trial_output.csv')
'''
data = pd.read_csv('trial_output.csv')

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]

# y here is the 0 / 1 column that is the DV in question
X = data_final.loc[:, data_final.columns != 'y']
Y = data_final.loc[:, data_final.columns == 'y']

os = SMOTE(random_state=0)                                  #find out what test size does
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns




