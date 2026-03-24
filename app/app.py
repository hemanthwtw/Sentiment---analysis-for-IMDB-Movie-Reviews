from flask import Flask, render_template, request, jsonify, send_file, abort
import joblib, re, os, nltk, html
import pandas as pd
import io
import uuid
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
BULK_RESULTS_CACHE = {}

MODEL_DIR = os.path.join("..", "models")
tfidf    = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
MODELS   = {
    "ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble.pkl")),
    "nb"      : joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "lr"      : joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "svm"     : joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
}

lemmatizer = WordNetLemmatizer()
NEGATION_WORDS = {"not", "no", "nor", "never", "none", "cannot"}
stop_words = set(stopwords.words("english")) - NEGATION_WORDS
CONTRACTION_PATTERNS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "am not"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'s\b", " is"),
    (r"'d\b", " would"),
    (r"'ll\b", " will"),
    (r"'ve\b", " have"),
    (r"'m\b", " am"),
]
EMOTICON_PATTERNS = [
    (r"(:-?\)|:d|=\))", " smile "),
    (r"(:-?\(|=\()", " sad "),
    (r";-?\)", " wink "),
]
STRONG_NEGATIVE_PATTERNS = [
    r"\bdid not like\b",
    r"\bdo not like\b",
    r"\bnot like\b",
    r"\bnot recommend\b",
    r"\bnever again\b",
    r"\bwaste of time\b",
    r"\bnot worth\b",
]

def preprocess(text: str) -> str:
    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in EMOTICON_PATTERNS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and (len(t) > 2 or t in NEGATION_WORDS)]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


def apply_rule_based_override(raw_text: str, cleaned_text: str, pred: int, proba) -> tuple[int, float, float, float]:
    joined = f"{str(raw_text).lower()} {cleaned_text}"
    raw_positive = float(proba[1])
    raw_negative = float(proba[0])

    if any(re.search(pattern, joined) for pattern in STRONG_NEGATIVE_PATTERNS):
        # Keep a floor confidence when explicit negative phrasing is detected.
        adjusted_negative = max(raw_negative, 0.85)
        adjusted_positive = 1.0 - adjusted_negative
        return 0, adjusted_negative, adjusted_positive, adjusted_negative

    return pred, float(proba[pred]), raw_positive, raw_negative


def get_movie_verdict(positive_pct: float) -> str:
    if positive_pct >= 60:
        return "Hit"
    if positive_pct >= 40:
        return "Mixed"
    return "Flop"


def build_overall_review(verdict: str, positive_pct: float, total: int) -> str:
    if total == 0:
        return "No reviews were available to analyze."

    if verdict == "Hit":
        return f"Audience response is strongly positive with {positive_pct}% favorable reviews. This looks like a crowd-pleasing movie."
    if verdict == "Mixed":
        return f"Audience response is divided with {positive_pct}% favorable reviews. The movie has both clear strengths and weaknesses."
    return f"Audience response is mostly negative with only {positive_pct}% favorable reviews. This title may not satisfy most viewers."


def predict_bulk_reviews(reviews):
    cleaned_reviews = [preprocess(review) for review in reviews]
    feats = tfidf.transform(cleaned_reviews)
    model = MODELS["ensemble"]
    preds = model.predict(feats)
    probas = model.predict_proba(feats)

    rows = []
    for i, review in enumerate(reviews):
        pred = int(preds[i])
        pred, confidence_score, _, _ = apply_rule_based_override(review, cleaned_reviews[i], pred, probas[i])
        rows.append({
            "review": review,
            "sentiment": "positive" if pred == 1 else "negative",
            "confidence": round(confidence_score * 100, 2),
        })

    total = len(rows)
    positive_count = sum(1 for row in rows if row["sentiment"] == "positive")
    positive_pct = round((positive_count / total) * 100, 2) if total else 0.0

    # Calculate rating based on positive percentage (0-5 scale)
    rating = round((positive_pct / 100.0) * 5, 1)

    verdict = get_movie_verdict(positive_pct)
    return {
        "verdict": verdict,
        "positive_pct": positive_pct,
        "rating": rating,
        "total_reviews": total,
        "positive_count": positive_count,
        "negative_count": total - positive_count,
        "overall_review": build_overall_review(verdict, positive_pct, total),
        "rows": rows,
    }


def cache_bulk_results(dashboard):
    token = str(uuid.uuid4())
    BULK_RESULTS_CACHE[token] = dashboard
    # Keep memory bounded by removing old entries.
    if len(BULK_RESULTS_CACHE) > 30:
        oldest_key = next(iter(BULK_RESULTS_CACHE))
        BULK_RESULTS_CACHE.pop(oldest_key, None)
    return token

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/analyzer")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    movie_name = request.form.get("movie_name", "").strip()
    review    = request.form.get("review", "").strip()
    model_key = request.form.get("model", "ensemble")
    
    if not movie_name:
        return render_template("index.html", error="Please enter a movie name.")
    if not review:
        return render_template("index.html", error="Please enter a review.")
    
    model   = MODELS.get(model_key, MODELS["ensemble"])
    cleaned = preprocess(review)
    feats   = tfidf.transform([cleaned])
    pred    = model.predict(feats)[0]
    proba   = model.predict_proba(feats)[0]
    pred, confidence_score, prob_positive, prob_negative = apply_rule_based_override(review, cleaned, int(pred), proba)
    
    # Calculate rating based on positive probability (0-5 scale)
    # prob_positive is already a decimal (0-1), not a percentage
    rating = round(prob_positive * 5, 1)
    
    result  = {
        "movie_name"    : movie_name,
        "rating"        : rating,
        "review"        : review,
        "sentiment"     : "positive" if pred == 1 else "negative",
        "confidence"    : round(confidence_score * 100, 2),
        "prob_positive" : round(prob_positive * 100, 2),
        "prob_negative" : round(prob_negative * 100, 2),
        "model_used"    : model_key.replace("_", " ").title(),
        "cleaned_text"  : cleaned[:300],
    }
    return render_template("result.html", result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data      = request.get_json(force=True)
    review    = data.get("review", "")
    model_key = data.get("model", "ensemble")
    cleaned   = preprocess(review)
    feats     = tfidf.transform([cleaned])
    model     = MODELS.get(model_key, MODELS["ensemble"])
    pred      = model.predict(feats)[0]
    proba     = model.predict_proba(feats)[0]
    pred, confidence_score, prob_positive, prob_negative = apply_rule_based_override(review, cleaned, int(pred), proba)
    return jsonify({
        "sentiment"     : "positive" if pred == 1 else "negative",
        "confidence"    : round(confidence_score * 100, 2),
        "prob_positive" : round(prob_positive * 100, 2),
        "prob_negative" : round(prob_negative * 100, 2),
    })


@app.route("/predict_bulk_text", methods=["POST"])
def predict_bulk_text():
    movie_name = request.form.get("movie_name", "").strip()
    bulk_text = request.form.get("bulk_reviews", "")
    reviews = [line.strip() for line in bulk_text.splitlines() if line.strip()]

    if not movie_name:
        return render_template("index.html", error="Please enter a movie name for bulk analysis.")
    if not reviews:
        return render_template("index.html", error="Please enter at least one review line for bulk text analysis.")

    dashboard = predict_bulk_reviews(reviews)
    dashboard["movie_name"] = movie_name
    download_token = cache_bulk_results(dashboard)
    return render_template("bulk_result.html", dashboard=dashboard, download_token=download_token)


@app.route("/predict_bulk_file", methods=["POST"])
def predict_bulk_file():
    movie_name = request.form.get("movie_name", "").strip()
    uploaded_file = request.files.get("reviews_file")

    if not movie_name:
        return render_template("index.html", error="Please enter a movie name for bulk analysis.")
    if not uploaded_file or not uploaded_file.filename:
        return render_template("index.html", error="Please upload a CSV or XLSX file.")

    filename = uploaded_file.filename.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            return render_template("index.html", error="Unsupported file type. Upload .csv or .xlsx only.")
    except Exception:
        return render_template("index.html", error="Could not read the uploaded file. Please check the format.")

    matching_col = None
    for col in df.columns:
        col_name = str(col).lower()
        if "review" in col_name or "text" in col_name:
            matching_col = col
            break

    if matching_col is None:
        return render_template("index.html", error="No review/text column found. Include a column name containing 'review' or 'text'.")

    reviews = [str(val).strip() for val in df[matching_col].dropna().tolist() if str(val).strip()]
    if not reviews:
        return render_template("index.html", error="No valid review rows found in the selected file column.")

    dashboard = predict_bulk_reviews(reviews)
    dashboard["movie_name"] = movie_name
    download_token = cache_bulk_results(dashboard)
    return render_template("bulk_result.html", dashboard=dashboard, download_token=download_token)


@app.route("/download_bulk_results/<token>", methods=["GET"])
def download_bulk_results(token):
    dashboard = BULK_RESULTS_CACHE.get(token)
    if dashboard is None:
        abort(404)

    df = pd.DataFrame(dashboard["rows"])

    # Remove control characters that openpyxl cannot write to cells.
    def sanitize_for_excel(value):
        if isinstance(value, str):
            return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)
        return value

    if not df.empty:
        df = df.map(sanitize_for_excel)

    if not df.empty:
        df.insert(0, "index", range(1, len(df) + 1))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="bulk_predictions")

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="bulk_sentiment_predictions.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)