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










