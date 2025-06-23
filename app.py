import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Function to infer university type from text
def infer_university_type(text):
    text = text.lower()
    public_keywords = ['public university', 'state university', 'public college']
    private_keywords = ['private university', 'private college', 'foundation university']
    for keyword in public_keywords:
        if keyword in text:
            return 'public'
    for keyword in private_keywords:
        if keyword in text:
            return 'private'
    return 'unknown'

# Prepare training data from NLTK movie_reviews corpus
def prepare_training_data():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]
    data = []
    for doc, category in documents:
        text = ' '.join(doc)
        label = 'positive' if category == 'pos' else 'negative'
        data.append({'text': text, 'sentiment': label})
    return pd.DataFrame(data)

# Sample university dataset
sample_data = {
    'text': [
        "The lecture halls in this private university are modern and well-equipped.",
        "Public university infrastructure is outdated and needs repair.",
        "Private uni has amazing libraries and study spaces!",
        "The public uni's buildings are falling apart, very disappointing.",
        "Great facilities at this private institution, love the new tech labs.",
        "Public university lacks proper maintenance for classrooms.",
        "State university classrooms are spacious but old.",
        "Private college campus is stunning, great investment in facilities.",
        "Public uni needs better funding for infrastructure upgrades.",
        "The private university's labs are top-notch, best I've seen.",
        "Public university buildings are crumbling, it's embarrassing.",
        "Private uni's new lecture halls are state-of-the-art!",
        "State college facilities are underfunded and outdated.",
        "Private institution has beautiful campus and modern amenities."
    ],
    'date': [pd.Timestamp.now()] * 14,
    'username': [f'user{i}' for i in range(1, 15)],
    'university_type': ['private', 'public', 'private', 'public', 'private', 'public', 'public', 'private', 'public', 'private', 'public', 'private', 'public', 'private']
}

# Train models
train_df = prepare_training_data()
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
y_train = train_df['sentiment']
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_split, y_train_split)
svm_val_predictions = svm_model.predict(X_val)
svm_metrics = {
    'accuracy': accuracy_score(y_val, svm_val_predictions),
    'report': classification_report(y_val, svm_val_predictions, output_dict=True)
}

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_val_predictions = rf_model.predict(X_val)
rf_metrics = {
    'accuracy': accuracy_score(y_val, rf_val_predictions),
    'report': classification_report(y_val, rf_val_predictions, output_dict=True)
}

# Function to process data and generate results
def process_data(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df = df[df['university_type'] != 'unknown']
    X_test = vectorizer.transform(df['cleaned_text'])
    df['sentiment_svm'] = svm_model.predict(X_test)
    df['sentiment_rf'] = rf_model.predict(X_test)
    
    # Aggregate sentiment by university type (Random Forest)
    sentiment_summary = df.groupby(['university_type', 'sentiment_rf']).size().unstack fill_value=0)
    sentiment_summary['total'] = sentiment_summary.sum(axis=1)
    for sentiment in ['positive', 'negative']:
        sentiment_summary[f'{sentiment}_percent'] = (sentiment_summary.get(sentiment, 0) / sentiment_summary['total'] * 100).round(2)
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    sentiment_summary[['positive_percent', 'negative_percent']].plot(kind='bar', stacked=True, color=['#28a745', '#dc3545'])
    plt.title('Sentiment Distribution by University Type (Random Forest)')
    plt.xlabel('University Type')
    plt.ylabel('Percentage')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    
    # Save plot to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return df, sentiment_summary, plot_url

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith(('.csv', '.xlsx')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                if 'text' not in df.columns or 'university_type' not in df.columns:
                    return render_template('results.html', error="File must contain 'text' and 'university_type' columns")
                
                df, sentiment_summary, plot_url = process_data(df)
            except Exception as e:
                return render_template('results.html', error=f"Error processing file: {str(e)}")
        else:
            return render_template('results.html', error="Please upload a valid CSV or XLSX file")
    else:
        df = pd.DataFrame(sample_data)
        df.to_csv('raw_sample_data.csv', index=False)
        df, sentiment_summary, plot_url = process_data(df)
    
    # Save results to XLSX
    output_file = f'sentiment_analysis_results_{uuid.uuid4()}.xlsx'
    df.to_excel(output_file, index=False, sheet_name='Raw_Data')
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
        sentiment_summary.to_excel(writer, sheet_name='Summary')
    
    # Prepare data for template
    raw_data = df[['text', 'university_type', 'sentiment_svm', 'sentiment_rf']].to_dict(orient='records')
    summary_data = sentiment_summary.to_dict()
    svm_accuracy = svm_metrics['accuracy']
    rf_accuracy = rf_metrics['accuracy']
    
    return render_template('results.html', raw_data=raw_data, summary_data=summary_data,
                         plot_url=plot_url, svm_accuracy=svm_accuracy, rf_accuracy=rf_accuracy,
                         output_file=output_file)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)