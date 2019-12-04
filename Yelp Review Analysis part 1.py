#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re


# In[44]:


df = pd.read_csv('yelp_train.csv')
df = df.dropna()
df.head()


# In[45]:


len(df['business_state'].unique())


# In[47]:


"""
Clean Data
"""
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df.drop('date', axis = 1)
cities = pd.get_dummies(df['business_city'])
states = pd.get_dummies(df['business_state'])
df.drop('business_city', axis = 1)
df.drop('business_state', axis = 1)
df.join(cities)
df.join(states)


# In[48]:


"""
Want to predict ratings from the review text
"""
labels = df['is_good_rating'].values
features = df['text'].values


# In[136]:


"""
Clean the text
"""
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


# In[8]:


"""
Use textblob sentiment
"""
from textblob import TextBlob


# In[10]:


yelp_text = np.array([TextBlob(sentence) for sentence in features])


# In[12]:


polarity = np.array([score.polarity for score in yelp_text])


# In[13]:


subjectivity = np.array([score.subjectivity for score in yelp_text])


# In[19]:


blob_sentiment = {'polarity': polarity, 'subjectivity': subjectivity}


# In[20]:


blob_df = pd.DataFrame(data = blob_sentiment)


# In[21]:


blob_df


# In[22]:


from sklearn.model_selection import train_test_split

X_blob_train, X_blob_val, y_blob_train, y_blob_val = train_test_split(blob_df, labels, test_size=0.2, random_state=0)


# In[68]:


"""
Train a random forest based on textblob sentiment
"""
from sklearn.ensemble import RandomForestClassifier
blob_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
blob_classifier.fit(X_blob_train, y_blob_train)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


training_blob_predictions = blob_classifier.predict(X_blob_train)
accuracy_score(y_blob_train, training_blob_predictions)


# In[30]:


val_blob_predictions = blob_classifier.predict(X_blob_val)
accuracy_score(y_blob_val, val_blob_predictions)


# In[137]:


"""
try tfidf vectorization
"""
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=200, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# In[90]:


processed_features


# In[91]:


rf_features = df[['cool', 'useful', 'funny',
                  'user_average_stars', 'user_review_count', 'business_review_count', 
                  'business_average_stars']].join(cities).join(states)


# In[95]:


"""
train a random forest using the tfidf vectors
"""
from sklearn.ensemble import RandomForestClassifier
feature_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
feature_classifier.fit(rf_features, labels)


# In[96]:


feature_predictions = feature_classifier.predict(rf_features)
accuracy_score(labels, feature_predictions)


# In[54]:


processed_features[:5]


# In[56]:


from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)


# In[57]:


from sklearn.metrics import accuracy_score


# In[58]:


training_predictions = text_classifier.predict(X_train)
accuracy_score(y_train, training_predictions)


# In[59]:


val_predictions = text_classifier.predict(X_val)
accuracy_score(y_val, val_predictions)


# In[60]:


df_test = pd.read_csv('yelp_test.csv')
test_lables = df_test.columns
test_features = df_test['text'].values

processed_test_features = []

for sentence in range(0, len(test_features)):
    # Remove all the special characters
    processed_test_feature = re.sub(r'\W', ' ', str(test_features[sentence]))

    # remove all single characters
    processed_test_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_test_feature)

    # Remove single characters from the start
    processed_test_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_test_feature) 

    # Substituting multiple spaces with single space
    processed_test_feature = re.sub(r'\s+', ' ', processed_test_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_test_feature = re.sub(r'^b\s+', '', processed_test_feature)

    # Converting to Lowercase
    processed_test_feature = processed_test_feature.lower()

    processed_test_features.append(processed_test_feature)


# In[61]:


test_vectorizer = TfidfVectorizer (max_features=500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_test_features = vectorizer.fit_transform(processed_test_features).toarray()


# In[62]:


X_test = processed_test_features


# In[63]:


len(X_test)


# In[139]:


"""
train an MLP Classifier using tfidf vectors
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer

net = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=100, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

net.fit(X_train,y_train)

print('Accuracy on training---')
y_pred_train=net.predict(X_train)
print(accuracy_score(y_train,y_pred_train))
print('Accuracy on test---')
y_pred_valid=net.predict(X_val)
print(accuracy_score(y_val,y_pred_valid))


# In[134]:


print('Accuracy on training---')
y_pred_train=net.predict(X_train)
print(accuracy_score(y_train,y_pred_train))
print('Accuracy on test---')
y_pred_valid=net.predict(X_val)
print(accuracy_score(y_val,y_pred_valid))


# In[110]:


"""
train an ensemble model using the previous models
"""
from sklearn.ensemble import VotingClassifier
#create a dictionary of our models
estimators = [('blob', blob_classifier), ('rf', text_classifier), ('net', net), ('features', feature_classifier), ('net2', net2)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting = 'hard')


# In[111]:


#fit model to training data
ensemble.fit(processed_features, labels)
ensemble_pred = ensemble.predict(processed_features)
#test our model on the test data
print(accuracy_score(labels,ensemble_pred))


# In[112]:


test_predictions = ensemble.predict(X_test)


# In[113]:


final_predictions = df_test[['review_id']]


# In[114]:


final_predictions['is_good_rating'] = test_predictions


# In[115]:


final_predictions.to_csv(r'yelp_final_results.csv', index=False)


# In[98]:


polars = pd.read_csv('yelp_train_polar.csv')


# In[100]:


polars


# In[102]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer

net2 = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=100, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

net2.fit(polars[['polarity', 'cool', 'useful', 'funny',
                  'user_average_stars', 'user_review_count', 'business_review_count', 
                  'business_average_stars']],polars[['is_good_rating']])


# In[105]:


net2_y_pred_train=net2.predict(polars[['polarity', 'cool', 'useful', 'funny',
                  'user_average_stars', 'user_review_count', 'business_review_count', 
                  'business_average_stars']])
print(accuracy_score(polars[['is_good_rating']],net2_y_pred_train))


# In[128]:


"""
train an AdaBoost classifier
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier(criterion='entropy',max_depth=1)
#AdaBoost = AdaBoostClassifier(base_estimator= model,n_estimators=400,learning_rate=1)
AdaBoost = AdaBoostClassifier(n_estimators=400,learning_rate=1,algorithm='SAMME')
AdaBoost.fit(processed_features,labels)
prediction = AdaBoost.score(processed_features,labels)
print('The accuracy is: ',prediction*100,'%')


# In[135]:


test_predictions = net.predict(X_test)
final_predictions = df_test[['review_id']]
final_predictions['is_good_rating'] = test_predictions
final_predictions.to_csv(r'yelp_final_results.csv', index=False)

