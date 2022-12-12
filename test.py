import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('train.csv')
train.head()

train.isnull().sum()
train.fillna(train.mean(), inplace=True)
train.isnull().sum()

test=pd.read_csv('test.csv')
test.head()

test.fillna(test.mean(), inplace=True)
sns.heatmap(train.corr(), cmap='Blues', annot=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X=train[['budget','popularity','runtime']]
y=train['revenue']

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.35)

regg=LinearRegression()

regg.fit(X_train, y_train)

pred=regg.predict(X_test)

test_x=test[['budget','popularity','runtime']]

predicted_revenue=regg.predict(test_x)

my_submission=pd.DataFrame({'id':test.id,'revenue':predicted_revenue})

my_submission.to_csv('for_submission_20221112_1.csv', index=False)