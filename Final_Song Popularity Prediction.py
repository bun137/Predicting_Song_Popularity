
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/content/dataset.csv")

"""## Data Visualization"""

df.head(10)

df.shape

df.isnull().sum()

df[df["artists"].isnull()]

df.drop([65900], inplace=True)

df.shape

print(df['artists'].unique())
print()
print()
print(df['album_name'].unique())
print()
print()
print(df['track_genre'].unique())

print(df['artists'].value_counts())
print()
print()
print(df['album_name'].value_counts())
print()
print()
print(df['track_genre'].value_counts())

#df['artists'].value_counts().plot(kind='bar',figsize=(16,9),color=(0.2, 0.3, 0.2, 0.8))

genre_popularity_score = df.groupby('track_genre')['popularity'].mean().sort_values(ascending = False).head(12)
genre_popularity_score

genre_popularity_score.plot(kind='bar',figsize=(16,9),color=(0.2, 0.3, 0.2, 0.8))
plt.title("Top 12 Most Popular Genre", size = 15)
plt.xticks(rotation=45)
plt.xlabel("Track Genre")
plt.ylabel("Popularity (0-100)")

artist_popularity_score = df.groupby('artists')['popularity'].mean().sort_values(ascending = False).head(12)
artist_popularity_score

artist_popularity_score.plot(kind='bar',figsize=(16,9),color=(0.2, 0.3, 0.2, 0.8))
plt.title("Top 12 Most Popular Artists", size = 15)
plt.xticks(rotation=45)
plt.xlabel("Track Artist")
plt.ylabel("Popularity (0-100)")

idx = df.groupby('track_genre')['popularity'].transform(max) == df['popularity']
mostPopularTrack = df[idx].drop_duplicates(['track_genre'])
mostPopularTrack = mostPopularTrack.drop_duplicates(['track_name'])
mostPopularTrack.sort_values('popularity', ascending=False, inplace=True)
mostPopularTrack = mostPopularTrack[['track_name','artists', 'track_genre', 'popularity']]
pd.set_option('display.max_rows', None)
mostPopularTrack.style.hide_index()

artist_genre=df.groupby('artists')['track_genre'].value_counts()
artist_genre

df.drop(["track_id","track_name","artists","album_name"], axis = 1, inplace=True)

df.drop(df.columns[0], axis = 1, inplace = True)

sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.show()

plt.scatter(df['loudness'].values, df['popularity'], color='teal')
plt.xlabel('loudness')
plt.ylabel('popularity')

plt.scatter(df['danceability'].values, df['popularity'], color='teal')
plt.xlabel('danceability')
plt.ylabel('popularity')

plt.scatter(df['energy'].values, df['popularity'], color='teal')
plt.xlabel('energy')
plt.ylabel('popularity')

plt.scatter(df['liveness'].values,df['popularity'], color='teal')
plt.xlabel('liveness')
plt.ylabel('popularity')

plt.scatter(df['tempo'].values,df['popularity'], color='teal')
plt.xlabel('tempo')
plt.ylabel('popularity')

"""## Preprocessing"""

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df

cleanup_nums = {"explicit": {"False":0 , "True":1}
               }

df = labelencoder(df)
df=df.replace(cleanup_nums)
df.head()

"""## Models"""

X = df.drop(columns = ['popularity']).values
y = df['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""### Multiple Linear Regression"""

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predictions
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

y_pred

y_test

print(f'Training Acc {round(regressor.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(regressor.score(X_test, y_test) * 100, 2)}%')

"""### Decision Tree"""

from sklearn.tree import DecisionTreeRegressor
dtregressor = DecisionTreeRegressor(random_state = 0)
dtregressor.fit(X_train, y_train)

y_pred_dt = dtregressor.predict(X_test)

y_pred_dt

y_test

print(f'Training Acc {round(dtregressor.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(dtregressor.score(X_test, y_test) * 100, 2)}%')

"""### Random Forests"""

from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfregressor.fit(X, y)

y_pred_rf = rfregressor.predict(X_test)

y_pred_rf

y_test

print(f'Training Acc {round(rfregressor.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(rfregressor.score(X_test, y_test) * 100, 2)}%')

"""### Support Vector Regression"""

from sklearn.svm import SVR
svregressor = SVR(kernel = 'rbf')
svregressor.fit(X, y)

y_pred_svr = svregressor.predict(X_test)

y_pred_svr

y_test

print(f'Training Acc {round(svregressor.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(svregressor.score(X_test, y_test) * 100, 2)}%')

svregressor_lin = SVR(kernel = 'linear')
svregressor_lin.fit(X, y)

y_pred_svr_lin = svregressor_lin.predict(X_test)

print(f'Training Acc {round(svregressor_lin.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(svregressor_lin.score(X_test, y_test) * 100, 2)}%')

svregressor_poly = SVR(kernel = 'poly')
svregressor_poly.fit(X, y)

y_pred_svr_poly = svregressor_poly.predict(X_test)

print(f'Training Acc {round(svregressor_poly.score(X_train, y_train) * 100, 2)}%')
print(f'Testing Acc {round(svregressor_poly.score(X_test, y_test) * 100, 2)}%')