"""
Disaster Tweet Classifier -- Flask Web Application
====================================================
Part 4 of the project brief: "Deployment with Web Interface".

Loads the trained scikit-learn pipeline (TF-IDF + engineered features + Logistic
Regression, tuned via GridSearchCV in the notebook) and exposes:

  GET  /            -> HTML form where a user pastes a tweet
  POST /predict      -> HTML form submission, renders the result on the page
  POST /api/predict -> JSON API: {"text": "..."} -> {"label": ..., "confidence": ...}

Run locally with:
    pip install -r requirements.txt
    python app.py

Then open:
    http://127.0.0.1:5000
"""

import json
import os
import re

import joblib
import nltk
import pandas as pd
from flask import Flask, jsonify, render_template, request


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIXED PATHS
# app.py and disaster_tweet_pipeline.joblib are both inside C:\nlp
MODEL_PATH = os.path.join(
    BASE_DIR,
    "disaster_tweet_pipeline.joblib"
)

# model_card.json should also be inside C:\nlp
MODEL_CARD_PATH = os.path.join(
    BASE_DIR,
    "model_card.json"
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Check model file before loading
# ---------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found!\n"
        f"Expected location:\n{MODEL_PATH}\n\n"
        f"Please make sure 'disaster_tweet_pipeline.joblib' "
        f"is inside the same folder as app.py."
    )


# ---------------------------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------------------------

for resource in [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "punkt",
    "punkt_tab"
]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


from nltk.corpus import stopwords          # noqa: E402
from nltk.stem import WordNetLemmatizer    # noqa: E402
from nltk.tokenize import word_tokenize    # noqa: E402


# ---------------------------------------------------------------------------
# NLP Setup
# ---------------------------------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"&[a-z]+;")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Identical cleaning logic used during training."""

    text = text.lower()

    text = URL_RE.sub(" ", text)

    text = HTML_RE.sub(" ", text)

    text = MENTION_RE.sub(" ", text)

    text = text.replace("#", " ")

    text = NON_ALPHA_RE.sub(" ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


MODEL = joblib.load(MODEL_PATH)

try:
    with open(MODEL_CARD_PATH, encoding="utf-8") as model_card_file:
        MODEL_CARD = json.load(model_card_file)
except (OSError, json.JSONDecodeError):
    MODEL_CARD = {}


def build_features(text: str) -> pd.DataFrame:
    """Build the same feature columns used by the serialized pipeline."""
    cleaned = clean_text(text)
    return pd.DataFrame([{
        "clean_joined": cleaned,
        "has_hashtag": int("#" in text),
        "has_mention": int(bool(MENTION_RE.search(text))),
        "has_url": int(bool(URL_RE.search(text))),
        "exclaim_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_word_count": sum(
            1 for word in text.split() if len(word) > 1 and word.isupper()
        ),
        "word_count": len(text.split()),
    }])


def predict_text(text: str) -> dict:
    features = build_features(text)
    prediction = int(MODEL.predict(features)[0])
    probabilities = MODEL.predict_proba(features)[0]
    confidence = float(max(probabilities))
    return {
        "label": prediction,
        "is_disaster": bool(prediction),
        "confidence": confidence,
    }


@app.get("/")
def index():
    return render_template("index.html", model_card=MODEL_CARD)


@app.post("/predict")
def predict_form():
    submitted_text = request.form.get("tweet_text", "").strip()
    if not submitted_text:
        return render_template(
            "index.html",
            model_card=MODEL_CARD,
            error="Please enter some tweet text.",
            submitted_text=submitted_text,
        )

    try:
        result = predict_text(submitted_text)
    except Exception as error:
        return render_template(
            "index.html",
            model_card=MODEL_CARD,
            error=str(error),
            submitted_text=submitted_text,
        ), 500

    return render_template(
        "index.html",
        model_card=MODEL_CARD,
        result=result,
        submitted_text=submitted_text,
    )


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "JSON body must include non-empty string 'text'."}), 400

    try:
        return jsonify(predict_text(text.strip()))
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


# -------------------------------------------------------------------
