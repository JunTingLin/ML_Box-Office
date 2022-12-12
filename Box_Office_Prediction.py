# import modules ----------------------
import pandas as pd
import numpy as np
#資料視覺化套件模組
import matplotlib.pyplot as plt
import seaborn as sns

# 流程1-取得資料-----------------------------------------------
#裡面也可以放網際網路的位置
df = pd.read_csv("train.csv")

# 流程2-觀察處理資料-----------------------------------------------
#查看前五筆資料，因為預設(n=5)
df.head()
df.info()

sns.pairplot(df[['revenue','budget','popularity','runtime']], dropna=True)

# 我們要的runtime欄位有兩個空值
df['runtime'].isnull().value_counts()
# 空值使用平均去填補
df['runtime']=df['runtime'].fillna(df['runtime'].mean())



# 流程3-資料切割-------------------------------------------
X=df[['budget','popularity','runtime']]
y=df['revenue']

#split to training data & testing data-----------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)
# random_state亂數的種子值，隨便設

# 流程4-模型選擇-------------------------------------------
#using Logistic regression model--------------------------
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train) #題目、答案
#get the result
predictions = reg.predict(X_test) #nparray
predictions

# 流程5-結果分析與驗證
from sklearn.metrics import r2_score
r2_score(y_test, predictions) #能不能讓他更靠近1，去調整
plt.scatter(y_test, predictions, color='blue', alpha=0.1)

#Model Export
import joblib
joblib.dump(reg,'Box_Office-LR-20221112.pkl',compress=3)
# compress 0~9
