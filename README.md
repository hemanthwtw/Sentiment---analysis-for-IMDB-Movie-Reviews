Here is a professional, high-quality `README.md` file tailored specifically to your project's workflow, code, and tech stack. It is designed to look great on GitHub and impress anyone reviewing your repository.

---

# 🎬 IMDB Movie Sentiment Analysis Pipeline

An end-to-end Machine Learning project that classifies movie reviews as **Positive** or **Negative** using Natural Language Processing (NLP). This project features a comparative analysis of multiple algorithms and a real-time web interface built with **Flask**.

## 📌 Project Overview

The goal of this project is to build a robust sentiment classifier trained on the **IMDB 50k dataset**. It addresses the challenges of text noise (HTML tags, punctuation) and preserves linguistic context (negation handling) to provide accurate predictions.

### **Key Features**

* **Data Cleaning:** Custom NLP pipeline with Lemmatization and selective Stop-word removal.
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams (1,2).
* **Ensemble Learning:** A "Soft Voting" classifier combining Naive Bayes, Logistic Regression, and SVM.
* **Web Deployment:** Real-time prediction interface with a confidence score dashboard.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** * `Scikit-Learn`: Machine Learning models & TF-IDF
* `NLTK`: Text preprocessing & Lemmatization
* `Pandas` & `NumPy`: Data manipulation
* `Matplotlib` & `Seaborn`: EDA and Evaluation plots
* `Flask`: Web framework
* `Joblib`: Model serialization



---

## 🚀 Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

```

### **2. Setup Virtual Environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\Activate.ps1
# On Mac/Linux
source venv/bin/activate

```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt

```

### **4. Run the Application**

```bash
python app.py

```

Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📊 Workflow & Methodology

### **1. Exploratory Data Analysis (EDA)**

Before modeling, we analyzed the dataset to ensure a 50/50 balanced split of sentiments and visualized word frequencies using **Word Clouds**.

### **2. Preprocessing**

* Converted text to lowercase.
* Removed HTML tags and special characters.
* **Preserved Negations:** Kept words like "not" and "no" as they are critical for sentiment.
* **Lemmatization:** Reduced words to their base form (e.g., "watched" → "watch").

### **3. Model Comparison**

We evaluated four different strategies to find the best performance:

1. **Multinomial Naive Bayes:** Baseline frequency-based model.
2. **Logistic Regression:** Strong linear classification using word weights.
3. **Support Vector Machine (LinearSVC):** High-dimensional boundary separation.
4. **Ensemble (Voting):** Combined the above to reduce variance and improve accuracy.

---

## 📈 Evaluation

The models were evaluated using **Confusion Matrices** and **F1-Scores** to ensure balanced performance between identifying Positive and Negative reviews.

| Model | Accuracy | F1-Score |
| --- | --- | --- |
| Naive Bayes | ~86% | 0.85 |
| Logistic Regression | ~89% | 0.89 |
| SVM | ~88% | 0.88 |
| **Ensemble Voting** | **~90%** | **0.90** |

---

## 📂 Project Structure

```text
├── app/
│   ├── app.py              # Flask Application
│   ├── templates/          # HTML files
│   └── static/             # CSS and Images
├── models/
│   ├── ensemble.pkl        # Saved Ensemble Model
│   └── tfidf_vectorizer.pkl # Saved Vectorizer
├── notebooks/
│   └── sentiment_analysis.ipynb # Data Analysis & Training
├── requirements.txt        # Project Dependencies
└── README.md

```

---

## 👨‍💻 Author

**Hemanth Venkatesh** 

---

**Would you like me to help you create a specific "About the Author" section or a more detailed "Future Work" section to add to this README?**
