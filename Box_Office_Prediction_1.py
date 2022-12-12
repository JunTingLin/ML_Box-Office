# 引入部分套件，sklearn套件等需要用到時再引入
import numpy as np # 線性代數
import pandas as pd # 數據處理，CSV文件I/O(例如pd.read_csv)
import matplotlib.pyplot as plt #繪圖套件
import seaborn as sns # 資料視覺化
import datetime as dt # 日期與時間

# 流程1-取得資料-----------------------------------------------
data = pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

# 流程2-資料清理與視覺化---------------------------------------
data.head() #列出前五筆
data.info() #資料集的相關資訊

# 首先刪除與我們預測無關的欄位
data.drop(['imdb_id','poster_path'],axis=1,inplace=True)
test.drop(['imdb_id','poster_path'],axis=1,inplace=True)

# 處理homepage欄位-------------------------------------------
# 將homepage欄位轉成binary(有、無)新欄位has_homepage
data['has_homepage'] = 0
data.loc[data['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1

# Homepage v/s Revenue
sns.catplot(x='has_homepage', y='revenue', data=data).set(title='Revenue for film with and without homepage')

# 原本homepage欄位去除
data=data.drop(['homepage'],axis =1)
test=test.drop(['homepage'],axis =1)

# 處理belongs_to_collection欄位-------------------------------------------
# 將belongs_to_collection欄位轉成binary(有、無)新欄位collection
data['collection'] = 0
data.loc[data['belongs_to_collection'].isnull() == False, 'collection'] = 1
test['collection'] = 0
test.loc[test['belongs_to_collection'].isnull() == False, 'collection'] = 1

# collections v/s Revenue
sns.catplot(x='collection', y='revenue', data=data).set(title='Revenue for film with and without collection')

# 原本belongs_to_collection欄位去除
data=data.drop(['belongs_to_collection'],axis =1)
test=test.drop(['belongs_to_collection'],axis =1)

# 處理Genres欄位---------------------------------------------------
from wordcloud import WordCloud, STOPWORDS
from collections import OrderedDict
genres = {}
for i in data['genres']:
    if(not(pd.isnull(i))):
        if (eval(i)[0]['name']) not in genres:
            genres[eval(i)[0]['name']]=1
        else:
                genres[eval(i)[0]['name']]+=1
                
plt.figure(figsize = (12, 8))
wordcloud = WordCloud(background_color="white",width=1000,height=1000, max_words=10,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(genres)

plt.imshow(wordcloud)
plt.title('Top genres')
plt.axis("off")
plt.show()
genres = OrderedDict(genres)
#Drama, Comedy 和 Action 是最受歡迎的類型
OrderedDict(sorted(genres.items(), key=lambda t: t[1]))

# 為每部電影類型添加數字
genres_count=[]
for i in data['genres']:
    if(not(pd.isnull(i))):
        
        genres_count.append(len(eval(i)))
        
    else:
        genres_count.append(0)
data['num_genres'] = genres_count

#Genres v/s revenue
sns.catplot(x='num_genres', y='revenue', data=data).set(title='Revenue for different number of genres in the film')

# 添加類型數字到測試數據
genres_count_test=[]
for i in test['genres']:
    if(not(pd.isnull(i))):
        
        genres_count_test.append(len(eval(i)))
        
    else:
        genres_count_test.append(0)
test['num_genres'] = genres_count_test

# 原本genres欄位去除
data.drop(['genres'],axis=1, inplace = True)
test.drop(['genres'],axis=1, inplace = True)

# 處理production_companies欄位-------------------------------------------
# 添加production_companies計數數據
prod_comp_count=[]
for i in data['production_companies']:
    if(not(pd.isnull(i))):
        
        prod_comp_count.append(len(eval(i)))
        
    else:
        prod_comp_count.append(0)
data['num_prod_companies'] = prod_comp_count

# number of prod companies vs revenue
sns.catplot(x='num_prod_companies', y='revenue', data=data).set(title='Revenue for different number of production companies in the film')

# 為測試數據添加Production_companies計數數據
prod_comp_count_test=[]
for i in test['production_companies']:
    if(not(pd.isnull(i))):
        
        prod_comp_count_test.append(len(eval(i)))
        
    else:
        prod_comp_count_test.append(0)
test['num_prod_companies'] = prod_comp_count_test

# 原本production_companies欄位去除
data.drop(['production_companies'],axis=1, inplace = True)
test.drop(['production_companies'],axis=1, inplace = True)

# 處理production_countries欄位-------------------------------------------
# 添加production_countsries計數數據
prod_coun_count=[]
for i in data['production_countries']:
    if(not(pd.isnull(i))):
        
        prod_coun_count.append(len(eval(i)))
        
    else:
        prod_coun_count.append(0)
data['num_prod_countries'] = prod_coun_count

# number of prod countries vs revenue
sns.catplot(x='num_prod_countries', y='revenue', data=data).set(title='Revenue for different number of production countries in the film')

# 為測試數據添加production_counties計算數據
prod_coun_count_test=[]
for i in test['production_countries']:
    if(not(pd.isnull(i))):
        
        prod_coun_count_test.append(len(eval(i)))
        
    else:
        prod_coun_count_test.append(0)
test['num_prod_countries'] = prod_coun_count_test

# 原本production_countries欄位去除
data.drop(['production_countries'],axis=1, inplace = True)
test.drop(['production_countries'],axis=1, inplace = True)

# 處理overview欄位-------------------------------------------
# 有評價的為1，將null設為為0
data['overview']=data['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)
test['overview']=test['overview'].apply(lambda x: 0 if pd.isnull(x) else 1)

sns.catplot(x='overview', y='revenue', data=data).set(title='Revenue for film with and without overview')

data= data.drop(['overview'],axis=1)
test= test.drop(['overview'],axis=1)

# 處理cast欄位-------------------------------------------
# 添加cast數據計數
total_cast=[]
for i in data['cast']:
    if(not(pd.isnull(i))):
        
        total_cast.append(len(eval(i)))
        
    else:
        total_cast.append(0)
data['cast_count'] = total_cast

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(data['cast_count'], data['revenue'])
plt.title('Number of cast members vs revenue')

# 為測試數據添加cast計數
total_cast=[]
for i in test['cast']:
    if(not(pd.isnull(i))):
        
        total_cast.append(len(eval(i)))
        
    else:
        total_cast.append(0)
test['cast_count'] = total_cast

# 原本cast欄位去除
data= data.drop(['cast'],axis=1)
test= test.drop(['cast'],axis=1)

# 處理crew欄位-------------------------------------------
total_crew=[]
for i in data['crew']:
    if(not(pd.isnull(i))):
        
        total_crew.append(len(eval(i)))
        
    else:
        total_crew.append(0)
data['crew_count'] = total_crew

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(data['crew_count'], data['revenue'])
plt.title('Number of crew members vs revenue')

# 為測試數據添加crew計數
total_crew=[]
for i in test['crew']:
    if(not(pd.isnull(i))):
        
        total_crew.append(len(eval(i)))
        
    else:
        total_crew.append(0)
test['crew_count'] = total_crew

# 原本crew欄位去除
data= data.drop(['crew'],axis=1)
test= test.drop(['crew'],axis=1)

# 無法量化，刪除 original_title 欄----------------------------------------
data= data.drop(['original_title'],axis=1)
test= test.drop(['original_title'],axis=1)

# language如何影響revenue----------------------------------------
plt.figure(figsize=(15,11)) #figure size

# 這是繪製我們數據的另一種方法。使用包含圖參數的變量
g1 = sns.boxenplot(x='original_language', y='revenue', data=data[(data['original_language'].isin((data['original_language'].sort_values().value_counts()[:10].index.values)))])
g1.set_title("Revenue by language", fontsize=20) # 標題和字體
g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # 當我們使用變量到圖表時，這是旋轉XTICKS的方法
g1.set_xlabel('Language', fontsize=18) # Xlabel
g1.set_ylabel('Revenue', fontsize=18) #Ylabel

plt.show()

# 僅考慮 en 和 zh，因為它們是最高票房
data['original_language'] =data['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))
test['original_language'] =test['original_language'].apply(lambda x: 1 if x=='en' else(2 if x=='zh' else 0))

# 檢查變量之間的相關性-----------------------------------------
col = ['revenue','budget','popularity','runtime']

plt.subplots(figsize=(10, 8))

corr = data[col].corr()

sns.heatmap(corr, xticklabels=col,yticklabels=col, linewidths=.5, cmap="Reds")

# budget和revenue高度相關
sns.regplot(x="budget", y="revenue", data=data)


# 處理release_date欄位-------------------------------------------
# 檢查revenue如何取決於release_date
data['release_date']=pd.to_datetime(data['release_date'])
test['release_date']=pd.to_datetime(data['release_date'])

release_day = data['release_date'].value_counts().sort_index()
release_day_revenue= data.groupby(['release_date'])['revenue'].sum()
release_day_revenue.index=release_day_revenue.index.dayofweek
sns.barplot(x=release_day_revenue.index,y=release_day_revenue, data= data,ci=None)
plt.show()

# 向數據添加星期幾(dayofweek)特徵
data['release_day']=data['release_date'].dt.dayofweek
test['release_day']=test['release_date'].dt.dayofweek

# 測試數據空值填0
test['release_day']=test['release_day'].fillna(0)

# 原本release_date欄位去除
data.drop(['release_date'],axis=1,inplace=True)
test.drop(['release_date'],axis=1,inplace=True)

# 處理status欄位-------------------------------------------
print("train data")
print(data['status'].value_counts())
print("test data")
test['status'].value_counts()

# 特徵無關緊要，刪除 status 欄--------------------------
data.drop(['status'],axis=1,inplace =True)
test.drop(['status'],axis=1,inplace =True)

# 處理Keywords、title欄位-------------------------------------------
Keywords_count=[]
for i in data['Keywords']:
    if(not(pd.isnull(i))):
        
        Keywords_count.append(len(eval(i)))
        
    else:
        Keywords_count.append(0)
data['Keywords_count'] = Keywords_count

# number of prod countries vs revenue
sns.catplot(x='Keywords_count', y='revenue', data=data).set(title='Revenue for different number of Keywords in the film')

Keywords_count=[]
for i in test['Keywords']:
    if(not(pd.isnull(i))):
        
        Keywords_count.append(len(eval(i)))
        
    else:
        Keywords_count.append(0)
test['Keywords_count'] = Keywords_count

# 刪除title和keyword
data=data.drop(['Keywords'],axis=1)
data=data.drop(['title'],axis=1)
test=test.drop(['Keywords'],axis=1)
test=test.drop(['title'],axis=1)

# 處理tagline欄位-------------------------------------------
data['isTaglineNA'] = 0
data.loc[data['tagline'].isnull() == False, 'isTaglineNA'] = 1
test['isTaglineNA'] = 0
test.loc[test['tagline'].isnull() == False, 'isTaglineNA'] = 1

#Homepage v/s Revenue
sns.catplot(x='isTaglineNA', y='revenue', data=data).set(title='Revenue for film with and without tagline')

data.drop(['tagline'],axis=1,inplace =True)
test.drop(['tagline'],axis=1,inplace =True)

# 處理runtime欄位-------------------------------------------
# runtime 有 2 個空值；將其設置為平均值
data['runtime']=data['runtime'].fillna(data['runtime'].mean())
test['runtime']=test['runtime'].fillna(test['runtime'].mean())

# 處理spoken_languages欄位-------------------------------------------
# 為每部電影添加spoken_languages計數
spoken_count=[]
for i in data['spoken_languages']:
    if(not(pd.isnull(i))):
        
        spoken_count.append(len(eval(i)))
        
    else:
        spoken_count.append(0)
data['spoken_count'] = spoken_count


spoken_count_test=[]
for i in test['spoken_languages']:
    if(not(pd.isnull(i))):
        
        spoken_count_test.append(len(eval(i)))
        
    else:
        spoken_count_test.append(0)
test['spoken_count'] = spoken_count_test

# 原本spoken_languages欄位去除
data.drop(['spoken_languages'],axis=1,inplace=True)
test.drop(['spoken_languages'],axis=1,inplace=True)

# 資料處理結束~~來觀察一下
data.info()
data.head()


# 流程3-資料切割---------------------------------------
data['budget'] = np.log1p(data['budget'])
test['budget'] = np.log1p(test['budget'])

y= data['revenue'].values
cols = [col for col in data.columns if col not in ['revenue', 'id']]
X= data[cols].values
y = np.log1p(y)

# 流程4-模型選擇使用----------------------------------
from sklearn.model_selection import cross_val_score #交叉驗證套件
# model 1 - linear Regression 線性回歸
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
scores = cross_val_score(clf, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean()) #2.4213687728137847
#RMSE均方根誤差(離均差平方合的平均開根號)太大

# Model 2 - Random forest regression 隨機森林回歸
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, min_samples_split=5, random_state=0,n_estimators=500)
scores = cross_val_score(regr, X, y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean()) #2.2127539691384657

# 使用model2的regr
cols = [col for col in test.columns if col not in ['id']]
X_test= test[cols].values

regr.fit(X,y)
y_pred = regr.predict(X_test)


y_pred=np.expm1(y_pred)
pd.DataFrame({'id': test.id, 'revenue': y_pred}).to_csv('submission_RF.csv', index=False)

#Export model
import joblib
joblib.dump(regr,'Box_Office-LR-20221212.pkl',compress=3)



result = regr.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
print(f'Result:{result}')



























