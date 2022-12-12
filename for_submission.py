#Model Using
import joblib
model_pretrained = joblib.load('Box_Office-LR-20221112.pkl')

import pandas as pd
# for submission------------------------------------------------
df_test = pd.read_csv("test.csv")

df_test['runtime']=df_test['runtime'].fillna(df_test['runtime'].mean())

df_test = df_test[['budget','popularity','runtime']]

predictions2 = model_pretrained.predict(df_test)
predictions2

forSubmissionDF = pd.DataFrame(columns=['id','revenue'])
forSubmissionDF['id'] = range(3001,7399)
forSubmissionDF['revenue'] = predictions2
forSubmissionDF

forSubmissionDF.to_csv('for_submission_20221112.csv', index=False)
# index=False，否則會多一欄012345...