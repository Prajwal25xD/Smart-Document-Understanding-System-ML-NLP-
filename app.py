import streamlit as st
import joblib
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download("punkt")
nltk.download("wordnet")

# Load Model & Vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)



def predict_sentiment(text):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)
    return prediction[0]


st.title("🎬 Movie Review Sentiment Analyzer")

st.write("Enter a movie review and click Predict")

user_input = st.text_area("Movie Review")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result.upper()}")
