import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB

def user_input():
  Review = st.text_input("enter your review")
  data={"Sentiment": Review}
  features= pd.DataFrame(data,index=[0])
  return features

st.title("ML Major Project Deployment")
st.subheader("SENTIMENT ANALYSIS OF REVIEWS")
dframe = user_input()
#st.write(dframe)

df=pd.read_csv("twitterdata.csv",encoding='latin-1').drop(["Unnamed: 0"], axis = 1)
df['Score'] = df['Score'].replace([4],"Positive")
df['Score'] = df['Score'].replace([0],"Negative")




x =df.iloc[:,-1].values
y =df.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)


test_model=Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
test_model.fit(X_train, y_train)

y_pred = test_model.predict(dframe)

st.write(y_pred)
