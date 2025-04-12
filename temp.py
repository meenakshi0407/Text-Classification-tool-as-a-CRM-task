from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


df = pd.read_csv("C:/Users/Meenakshi/Downloads/crm_messages_100.csv")
texts = df["message"]
labels = df["label"]

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

model.fit(texts, labels)

def classify_message(msg):
    return model.predict([msg])[0]

import streamlit as st

st.title("CRM Message Classifier 📬")

st.write("Automatically Classifies Customer Messages into Labels")

#input Box
userInput=st.text_area("Please Enter the User Message ")

if st.button("Classify"):
    if userInput.strip():
        category=classify_message(userInput)
        st.success(f"Predicted category: {category.upper()}")
    else:
        st.warning("Please enter a message to classify.")




