#!/usr/bin/env python
# coding: utf-8

# # Offensive Language Detection

# In[124]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[125]:


dataset = pd.read_csv("HateSpeechData.csv")
dataset.head(10)


# In[126]:


def highlight_col(x):
    r = 'background-color: red'
    b = 'background-color: blue'
    temp=pd.DataFrame('', index=x.index, columns=x.columns)
    temp.iloc[:,2]=r
    temp.iloc[:,3]=b
    return temp

dataset.head(5).style.apply(highlight_col, axis=None)


# In[127]:


dataset.info()


# In[128]:


dataset.offensive_language.value_counts()


# In[129]:


dataset.isnull().sum()


# In[130]:


dataset['class'].hist()


# In[131]:


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


# In[132]:


dataset['text length'] = dataset['tweet'].apply(len)
print(dataset.head())


# In[133]:


import seaborn as sns
import matplotlib.pyplot as plt
graph = sns.FacetGrid(data=dataset, col='class')
graph.map(plt.hist, 'text length', bins=50)


# In[134]:


dataset=dataset.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'], axis = 1)
dataset.head(5)


# # Preprocessing of tweets

# In[135]:


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

# In[136]:


#TF-IDF Features-F1
tfidf_vectorizer = TfidfVectorizer()
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(dataset['processed_tweets'] )


# In[137]:


from sklearn.metrics import classification_report
X = tfidf
y = dataset['class'].astype(int)
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
y_preds = model.predict(X_test_tfidf)
print(classification_report(y_test,y_preds))


# In[ ]:


v

