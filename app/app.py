from flask import Flask, render_template, request, jsonify
import joblib, re, os, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

MODEL_DIR = os.path.join("..", "models")
tfidf    = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
MODELS   = {
    "ensemble": joblib.load(os.path.join(MODEL_DIR, "ensemble.pkl")),
    "nb"      : joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "lr"      : joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "svm"     : joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"no","not","nor","never","nothing","n't","cannot"}

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z']", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review    = request.form.get("review", "").strip()
    model_key = request.form.get("model", "ensemble")
    if not review:
        return render_template("index.html", error="Please enter a review.")
    model   = MODELS.get(model_key, MODELS["ensemble"])
    cleaned = preprocess(review)
    feats   = tfidf.transform([cleaned])
    pred    = model.predict(feats)[0]
    proba   = model.predict_proba(feats)[0]
    result  = {
        "review"        : review,
        "sentiment"     : "positive" if pred == 1 else "negative",
        "confidence"    : round(float(proba[pred]) * 100, 2),
        "prob_positive" : round(float(proba[1]) * 100, 2),
        "prob_negative" : round(float(proba[0]) * 100, 2),
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
    return jsonify({
        "sentiment"     : "positive" if pred == 1 else "negative",
        "confidence"    : round(float(proba[pred]) * 100, 2),
        "prob_positive" : round(float(proba[1]) * 100, 2),
        "prob_negative" : round(float(proba[0]) * 100, 2),
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)