import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.set_option('display.max_columns', None)
'''ds = pd.read_csv('BankChurners.csv')

ds['Attrition_Flag'].replace({'Existing Customer': 0, 'Attrited Customer': 1}, inplace=True)
ds['Gender'].replace({'M': 1, 'F': 0}, inplace=True)
ds['Education_Level'].replace({'Unknown': -99, 'Uneducated': 0, 'High School': 1, 'College': 2,
                               'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5}, inplace=True)
ds['Marital_Status'].replace({'Unknown': -99, 'Single': 1, 'Married': 2, 'Divorced': 3}, inplace=True)
ds['Income_Category'].replace({'Unknown': -99, 'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
                               '$80K - $120K': 3, '$120K +': 4}, inplace=True)
ds['Card_Category'].replace({'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}, inplace=True)

wr = ds.to_csv('Edited_BankChurners.csv')'''

df = pd.read_csv('Edited_BankChurners.csv')
hmm = df[df.columns[0:23]].corr()['Attrition_Flag'][:-1]

gender_chi = pd.crosstab(df['Attrition_Flag'], df['Gender'])
gender = stats.chi2_contingency(gender_chi)



print(hmm)
print(gender_chi)
print(gender)

