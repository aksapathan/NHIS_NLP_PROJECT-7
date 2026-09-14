"""
Disaster Tweet Classifier -- Flask Web Application
====================================================
Part 4 of the project brief: "Deployment with Web Interface".

Loads the trained scikit-learn pipeline (TF-IDF + engineered features + Logistic
Regression, tuned via GridSearchCV in the notebook) and exposes:

  GET  /            -> HTML form where a user pastes a tweet
  POST /predict      -> HTML form submission, renders the result on the page
  POST /api/predict  -> JSON API: {"text": "..."} -> {"label": ..., "confidence": ...}

Run locally with:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
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
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "disaster_tweet_pipeline.joblib")
MODEL_CARD_PATH = os.path.join(BASE_DIR, "..", "models", "model_card.json")

app = Flask(__name__)

# Make sure the NLTK resources used during training are available at inference time too.
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
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

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"&[a-z]+;")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Identical cleaning logic used during training (see the notebook, Part 1)."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str):
    tokens = word_tokenize(text)
    return [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]


def build_feature_row(raw_text: str) -> pd.DataFrame:
    """Reproduce the exact feature set the pipeline was trained on."""
    cleaned = clean_text(raw_text)
    tokens = tokenize_and_lemmatize(cleaned)
    clean_joined = " ".join(tokens)

    return pd.DataFrame(
        [
            {
                "clean_joined": clean_joined,
                "has_hashtag": int("#" in raw_text),
                "has_mention": int("@" in raw_text),
                "has_url": int(bool(URL_RE.search(raw_text))),
                "exclaim_count": raw_text.count("!"),
                "question_count": raw_text.count("?"),
                "uppercase_word_count": sum(
                    1 for w in raw_text.split() if w.isupper() and len(w) > 1
                ),
                "word_count": len(raw_text.split()),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("Loading model pipeline from:", MODEL_PATH)
MODEL = joblib.load(MODEL_PATH)

MODEL_CARD = {}
if os.path.exists(MODEL_CARD_PATH):
    with open(MODEL_CARD_PATH) as f:
        MODEL_CARD = json.load(f)


def predict(raw_text: str) -> dict:
    row = build_feature_row(raw_text)

    pred = int(MODEL.predict(row)[0])

    # Confidence: use predict_proba when available, otherwise a normalized decision_function.
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(row)[0]
        confidence = float(proba[pred])
    elif hasattr(MODEL, "decision_function"):
        score = float(MODEL.decision_function(row)[0])
        confidence = 1 / (1 + pow(2.71828, -score))  # sigmoid squash for display
        if pred == 0:
            confidence = 1 - confidence
    else:
        confidence = None

    return {
        "label": "Disaster" if pred == 1 else "Not a Disaster",
        "is_disaster": bool(pred),
        "confidence": round(confidence, 4) if confidence is not None else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_card=MODEL_CARD)


@app.route("/predict", methods=["POST"])
def predict_form():
    tweet_text = request.form.get("tweet_text", "").strip()
    result = None
    error = None

    if not tweet_text:
        error = "Please enter some tweet text before submitting."
    else:
        result = predict(tweet_text)

    return render_template(
        "index.html",
        model_card=MODEL_CARD,
        result=result,
        error=error,
        submitted_text=tweet_text,
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True) or {}
    tweet_text = (data.get("text") or "").strip()

    if not tweet_text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    result = predict(tweet_text)
    result["input_text"] = tweet_text
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
