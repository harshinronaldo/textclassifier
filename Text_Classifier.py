# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:50:59 2023

@author: ROHAN
"""

import streamlit as st
import pandas as pd
# Reading the dataset using read_csv() in pandas 
bbc_text = pd.read_csv('bbc-text.txt')
bbc_text=bbc_text.rename(columns = {'text': 'News_Headline'}, inplace = False)
bbc_text.head()
bbc_text.category = bbc_text.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
from sklearn.model_selection import train_test_split
X = bbc_text.News_Headline
y = bbc_text.category
#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words = 'english',lowercase=False)
# fit the vectorizer on the training data
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
# for test data
X_test_transformed = vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)
#For saving the model
import pickle
saved_model = pickle.dumps(naivebayes)
#For loading the saved model
saved = pickle.loads(saved_model)
st.header('TEXT CLASSIFIER WITH NAIVE BAYES')
input = st.text_area("Please enter the text", value="")
vec = vector.transform([input]).toarray()
if st.button("Predict"):
    st.write((str(list(saved.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS')))