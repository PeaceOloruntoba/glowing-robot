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
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import io
import base64

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Download NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

# Prepare training data
def prepare_training_data():
    docs = [(list(movie_reviews.words(fileid)), cat)
            for cat in movie_reviews.categories()
            for fileid in movie_reviews.fileids(cat)]
    return pd.DataFrame({
        "text": [" ".join(doc) for doc, _ in docs],
        "sentiment": ["positive" if cat == "pos" else "negative" for _, cat in docs]
    })

# Train models
train_df = prepare_training_data()
train_df["cleaned_text"] = train_df["text"].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["cleaned_text"])
y_train = train_df["sentiment"]
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train_split, y_train_split)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)

# Model metrics
svm_accuracy = accuracy_score(y_val, svm_model.predict(X_val))
rf_accuracy = accuracy_score(y_val, rf_model.predict(X_val))

# Process uploaded data
def process_data(df):
    opinion_cols = [col for col in df.columns if "My school" in col]
    df["text"] = df[opinion_cols].fillna("").apply(lambda x: " ".join(x), axis=1)
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    X_test = vectorizer.transform(df["cleaned_text"])
    df["sentiment_rf"] = rf_model.predict(X_test)
    df["sentiment_svm"] = svm_model.predict(X_test)
    return df

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", svm_accuracy=svm_accuracy, rf_accuracy=rf_accuracy)

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file and file.filename.endswith(".csv"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        df_processed = process_data(df)
        result_file = f"results_{uuid.uuid4()}.xlsx"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_file)
        df_processed.to_excel(result_path, index=False)
        return jsonify({"status": "success", "file": result_file})
    return jsonify({"status": "error", "message": "Invalid file format"})

@app.route("/api/sentiment_summary")
def api_sentiment_summary():
    df = pd.read_excel(os.path.join(app.config['RESULT_FOLDER'], os.listdir(app.config['RESULT_FOLDER'])[-1]))
    summary = df.groupby("Kind of University")["sentiment_rf"].value_counts(normalize=True).unstack().fillna(0) * 100
    return summary.round(2).to_json()

@app.route("/api/sample_posts")
def api_sample_posts():
    sentiment = request.args.get("sentiment", "positive")
    df = pd.read_excel(os.path.join(app.config['RESULT_FOLDER'], os.listdir(app.config['RESULT_FOLDER'])[-1]))
    posts = df[df["sentiment_rf"] == sentiment]["text"].sample(5).tolist()
    return jsonify(posts)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
