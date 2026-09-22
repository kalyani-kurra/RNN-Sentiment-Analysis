from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = Flask(__name__)

# ---------------------------------------
# Load trained RNN model
# ---------------------------------------
model = tf.keras.models.load_model("sentiment_rnn.h5")

# ---------------------------------------
# Load tokenizer
# ---------------------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = 200


# ---------------------------------------
# Text cleaning
# ---------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------
# Sentiment prediction
# ---------------------------------------
def predict_sentiment(review):

    original_review = review

    review = clean_text(review)

    # Convert text to sequence
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
        model.predict(padded, verbose=0)[0][0]
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
        word in negative_words for word in words
    )

    positive_count = sum(
        word in positive_words for word in words
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
# Home page
# ---------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    review = ""

    if request.method == "POST":

        review = request.form.get(
            "review",
            ""
        ).strip()

        if review:

            sentiment, confidence = predict_sentiment(
                review
            )

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


# ---------------------------------------
# API endpoint
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "review" not in data:

        return {
            "error": "Please provide a review"
        }, 400

    review = data["review"]

    sentiment, confidence = predict_sentiment(
        review
    )

    return {
        "review": review,
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }


# ---------------------------------------
# Run Flask
# ---------------------------------------
if __name__ == "__main__":

    app.run(debug=True)

