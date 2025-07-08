import pandas as pd
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import json

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# NLTK setup
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Prepare training data
def prepare_training_data():
    docs = [(list(movie_reviews.words(fileid)), cat)
            for cat in movie_reviews.categories()
            for fileid in movie_reviews.fileids(cat)]
    df = pd.DataFrame({
        "text": [" ".join(doc) for doc, _ in docs],
        "sentiment": ["positive" if cat == "pos" else "negative" for _, cat in docs]
    })
    return df

train_df = prepare_training_data()
train_df["cleaned_text"] = train_df["text"].apply(preprocess_text)

# Vectorizer and models
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df["cleaned_text"])
y = train_df["sentiment"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel="linear", probability=True).fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Model metrics
svm_report = classification_report(y_val, svm_model.predict(X_val), output_dict=True)
rf_report = classification_report(y_val, rf_model.predict(X_val), output_dict=True)

# Process uploaded data
def process_data(df):
    opinion_cols = [col for col in df.columns if "My school" in col]
    df["text"] = df[opinion_cols].fillna("").apply(" ".join, axis=1)
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    X_test = vectorizer.transform(df["cleaned_text"])
    df["sentiment_svm"] = svm_model.predict(X_test)
    df["sentiment_rf"] = rf_model.predict(X_test)
    return df

# Store latest result
latest_result = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global latest_result
    file = request.files.get("file")
    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        processed_df = process_data(df)
        result_file = f"result_{uuid.uuid4()}.xlsx"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_file)
        processed_df.to_excel(result_path, index=False)
        latest_result = processed_df
        return jsonify({"status": "success", "redirect": "/dashboard"})
    return jsonify({"status": "error", "message": "Invalid file"})

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", svm_report=svm_report, rf_report=rf_report)

@app.route("/api/summary")
def api_summary():
    if latest_result is None:
        return jsonify({"error": "No data"})
    summary = latest_result.groupby("Kind of University")["sentiment_rf"].value_counts(normalize=True).unstack().fillna(0) * 100
    return summary.round(2).to_json()

@app.route("/api/trends")
def api_trends():
    if latest_result is None:
        return jsonify({"error": "No data"})
    latest_result["date"] = pd.to_datetime(latest_result["Timestamp"])
    trend = latest_result.groupby([latest_result["date"].dt.to_period("M"), "sentiment_rf"]).size().unstack().fillna(0)
    return trend.to_json()

@app.route("/api/posts")
def api_posts():
    sentiment = request.args.get("sentiment", "positive")
    posts = latest_result[latest_result["sentiment_rf"] == sentiment]["text"].sample(5).tolist()
    return jsonify(posts)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
