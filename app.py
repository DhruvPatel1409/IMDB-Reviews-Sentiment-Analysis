import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the pre-trained model
loaded_model = pickle.load(open('sentiment_model.pkl','rb'))

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

maxlen = 200

st.title("Sentiment Analysis")

review = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if review:
        sequences = tokenizer.texts_to_sequences([review])
        padded_sequences = pad_sequences(sequences, maxlen=maxlen)
        
        prediction = loaded_model.predict(padded_sequences)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review.")
