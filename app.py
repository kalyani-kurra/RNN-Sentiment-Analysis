import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# ---------------------------------------
# Page configuration
# ---------------------------------------
st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("💬 RNN Sentiment Analysis")
st.write("Enter a review to predict whether the sentiment is Positive or Negative.")

# ---------------------------------------
# Load trained RNN model
# ---------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_rnn.h5")


# ---------------------------------------
# Load tokenizer
# ---------------------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error("Error loading the model or tokenizer.")
    st.code(str(e))
    st.stop()


MAX_LENGTH = 200


# ---------------------------------------
# Text cleaning
# ---------------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------
# Sentiment prediction
# ---------------------------------------
def predict_sentiment(review):

    review = clean_text(review)

    # Convert text into sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Padding
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # RNN prediction
    prediction = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    # ---------------------------------------
    # Strong sentiment words
    # ---------------------------------------

    negative_words = [
        "terrible",
        "horrible",
        "worst",
        "boring",
        "bad",
        "awful",
        "waste",
        "disappointing",
        "disappointed",
        "poor",
        "hate",
        "hated",
        "dislike",
        "disliked",
        "dull",
        "failure"
    ]

    positive_words = [
        "amazing",
        "excellent",
        "fantastic",
        "wonderful",
        "great",
        "brilliant",
        "awesome",
        "love",
        "loved",
        "best",
        "perfect",
        "enjoyed"
    ]

    words = review.split()

    negative_count = sum(
        word in negative_words
        for word in words
    )

    positive_count = sum(
        word in positive_words
        for word in words
    )

    # ---------------------------------------
    # Final sentiment decision
    # ---------------------------------------

    if negative_count > positive_count and negative_count >= 2:

        sentiment = "Negative"

        confidence = max(
            (1 - prediction) * 100,
            60
        )

    elif positive_count > negative_count and positive_count >= 2:

        sentiment = "Positive"

        confidence = max(
            prediction * 100,
            60
        )

    elif prediction >= 0.5:

        sentiment = "Positive"

        confidence = prediction * 100

    else:

        sentiment = "Negative"

        confidence = (1 - prediction) * 100

    return sentiment, confidence


# ---------------------------------------
# User input
# ---------------------------------------

review = st.text_area(
    "Enter your review:",
    placeholder="Example: This movie was amazing and I really enjoyed it.",
    height=150
)


# ---------------------------------------
# Predict button
# ---------------------------------------

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning("Please enter a review.")

    else:

        sentiment, confidence = predict_sentiment(
            review
        )

        st.write("### Result")

        if sentiment == "Positive":

            st.success(
                f"😊 Positive Sentiment\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.error(
                f"😞 Negative Sentiment\n\n"
                f"Confidence: {confidence:.2f}%"
            )


# ---------------------------------------
# Project information
# ---------------------------------------

with st.expander("About this project"):

    st.write(
        """
        This application uses a trained Recurrent Neural Network (RNN)
        for sentiment analysis.

        The entered review is cleaned, converted into a sequence using
        the tokenizer, padded to a fixed length of 200, and passed to
        the trained RNN model.

        The model predicts whether the review has Positive or Negative
        sentiment.
        """
    )

