import re
import html
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for p in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(p, quiet=True)

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

vec = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/ensemble.pkl")
text = "i did not like the movie"
cleaned = preprocess(text)
X = vec.transform([cleaned])
pred = int(model.predict(X)[0])
proba = model.predict_proba(X)[0]
print("cleaned:", cleaned)
print("pred:", pred)
print("positive:", round(float(proba[1]), 4))
print("negative:", round(float(proba[0]), 4))
