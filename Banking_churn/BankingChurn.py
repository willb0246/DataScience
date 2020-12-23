import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
ds = pd.read_csv('BankChurners.csv')

'''ds['Attrition_Flag'].replace({'Existing Customer': 1, 'Attrited Customer': 0}, inplace=True)
ds['Gender'].replace({'M': 1, 'F': 0}, inplace=True)
ds['Education_Level'].replace({'Unknown': '', 'Uneducated': 0, 'High School': 1, 'College': 2,
                               'Graduate': 3, 'Post-Graduate': 4, 'Doctorate': 5}, inplace=True)
ds['Marital_Status'].replace({'Unknown': '', 'Single': 1, 'Married': 2, 'Divorced': 3}, inplace=True)
ds['Income_Category'].replace({'Unknown': '', 'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
                               '$80K - $120K': 3, '$120K +': 4}, inplace=True)
ds['Card_Category'].replace({'Blue': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}, inplace=True)

wr = ds.to_csv('Edited_BankChurners.csv')'''

df = pd.read_csv('Edited_BankChurners.csv')

gender_chi = pd.crosstab(df['Attrition_Flag'], df['Gender'])
gender = stats.chi2_contingency(gender_chi)
anova = stats.f_oneway(df['Attrition_Flag'], df['Customer_Age'])
kruskal = stats.kruskal(df['Attrition_Flag'], df['Customer_Age'])
manwhit = stats.mannwhitneyu(df['Attrition_Flag'], df['Customer_Age'])


mob_anova = stats.f_oneway(df['Attrition_Flag'], df['Months_on_book'])
mob_kruskal = stats.kruskal(df['Attrition_Flag'], df['Months_on_book'])
mob_manwhit = stats.mannwhitneyu(df['Attrition_Flag'], df['Months_on_book'])

print('gender Chi2: ', gender)
print('anova: ', anova)
print('kruskal: ', kruskal)
print('man Whitney: ', manwhit)

print('mob anova: ', mob_anova)
print('mob kruskal: ', mob_kruskal)
print('mob man Whitney: ', mob_manwhit)
