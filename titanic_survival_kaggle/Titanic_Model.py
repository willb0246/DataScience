import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

'''
# This block of code converts the string variables into integers

train_model = pd.read_csv('train.csv')
train_model['Sex']=np.where(train_model['Sex'] == 'male', '1', train_model['Sex'])
train_model['Sex']=np.where(train_model['Sex'] == 'female', '0', train_model['Sex'])
train_model['Embarked']=np.where(train_model['Embarked'] == 'C', '1', train_model['Embarked'])
train_model['Embarked']=np.where(train_model['Embarked'] == 'Q', '2', train_model['Embarked'])
train_model['Embarked']=np.where(train_model['Embarked'] == 'S', '3', train_model['Embarked'])
write_data = train_model.to_csv('training_data.csv')

test_model = pd.read_csv('test.csv')
test_model['Sex']=np.where(test_model['Sex'] == 'male', '1', test_model['Sex'])
test_model['Sex']=np.where(test_model['Sex'] == 'female', '0', test_model['Sex'])
test_model['Embarked']=np.where(test_model['Embarked'] == 'C', '1', test_model['Embarked'])
test_model['Embarked']=np.where(test_model['Embarked'] == 'Q', '2', test_model['Embarked'])
test_model['Embarked']=np.where(test_model['Embarked'] == 'S', '3', test_model['Embarked'])
write_data = test_model.to_csv('testing_data.csv')
'''

# only using the features that have integers (remove names, ect)
features = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']

test_model = pd.read_csv('testing_data.csv')
test_model = test_model.dropna()
test_X = test_model[features]
# y variable (survival) is not known, and is what we are looking for

training_model = pd.read_csv('training_data.csv')
training_model = training_model.dropna()
y = training_model.Survived
X = training_model[features]

# This would be used if we were training and testing on one set of data, but we have 2 sets here
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

titanic_model = RandomForestRegressor(random_state=1)
titanic_model.fit(X, y)
tm_val_pred = titanic_model.predict(test_X)
# use MAE to assess how good/poor our model is
# tm_val_mae = mean_absolute_error(val_y, tm_val_pred)
rand_forest_score = titanic_model.score(X, y)
acc_titanic_model = round(rand_forest_score * 100, 2)
print("Rand Forest: ", acc_titanic_model)

decision_tree_regression = DecisionTreeRegressor(random_state=1)
decision_tree_regression.fit(X, y)
dec_tree_pred = decision_tree_regression.predict(test_X)
dec_tree_score = decision_tree_regression.score(X, y)
acc_dec_tree = round(dec_tree_score * 100, 2)
print("Decision Tree: ", acc_dec_tree)