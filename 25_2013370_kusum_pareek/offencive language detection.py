#!/usr/bin/env python
# coding: utf-8

# # Offensive Language Detection

# In[138]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[139]:


dataset = pd.read_csv("HateSpeechData.csv")
dataset.head(10)


# In[140]:


def highlight_col(x):
    r = 'background-color: red'
    b = 'background-color: blue'
    temp=pd.DataFrame('', index=x.index, columns=x.columns)
    temp.iloc[:,2]=r
    temp.iloc[:,3]=b
    return temp

dataset.head(5).style.apply(highlight_col, axis=None)


# In[141]:


dataset.info()


# In[142]:


dataset.offensive_language.value_counts()


# In[143]:


dataset.isnull().sum()


# In[144]:


dataset['class'].hist()


# In[145]:


import matplotlib.pyplot as plt
offensive=sum(dataset['class']==1)+sum(dataset['class']==0)  
Not_Offensive=sum(dataset['class']==2)
total=dataset['class'].count()
print("Offensive tweets:",round(offensive*100/total,2),'%')
print("Non Offensive tweets:",round(Not_Offensive*100/total,2),'%')
y = np.array([round(Not_Offensive*100/total,2),round(offensive*100/total,2)])
mylabels = ["Not Offensive", "Offensive"]
plt.pie(y, labels = mylabels)
plt.show() 


# In[146]:


dataset['text length'] = dataset['tweet'].apply(len)
print(dataset.head())


# In[147]:


import seaborn as sns
import matplotlib.pyplot as plt
graph = sns.FacetGrid(data=dataset, col='class')
graph.map(plt.hist, 'text length', bins=50)


# In[148]:


dataset=dataset.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'], axis = 1)
dataset.head(5)


# # Preprocessing of tweets

# In[149]:


def preprocess(tweet):  
    #@name[mention]
    tweet = tweet.str.replace(r'@[\w\-]+', '')
    #links[https://abc.com]
    tweet = tweet.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
    # removal of punctuations and numbers
    tweet = tweet.str.replace("[^a-zA-Z]", " ")
    # remove whitespace with a single space
    tweet=tweet.str.replace(r'\s+', ' ')
    # removal of capitalization
    tweet = tweet.str.lower()
    # tokenizing
    tweet = tweet.apply(lambda x: x.split())
    for i in range(len(tweet)):
        tweet[i] = ' '.join(tweet[i])
        tweets_p= tweet
    return tweets_p
processed_tweets = preprocess(dataset.tweet)   

dataset['processed_tweets'] = processed_tweets
dataset.head(10)


# # Feature Engineering

# In[150]:


#TF-IDF Features-F1
tfidf_vectorizer = TfidfVectorizer()
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(dataset['processed_tweets'] )


# In[151]:


from sklearn.metrics import classification_report
X = tfidf
y = dataset['class'].astype(int)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
y_preds = model.predict(X_test_tfidf)
print(classification_report(y_test,y_preds))


# In[153]:


matrix = confusion_matrix(y_test,y_preds, labels=[1,0])
print('Confusion matrix : \n',matrix)


# In[154]:


sns.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# In[156]:


labels = ['True Pos','False Pos','False Neg','True Neg']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')


# In[ ]:




