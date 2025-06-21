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

# Function to infer university type
def infer_university_type(text):
    if pd.isna(text):
        return 'unknown'
    text = text.lower()
    if 'private university' in text:
        return 'private'
    if 'public university' in text:
        return 'public'
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

# Sample dataset based on SENTIMENT ANALYSIS.csv structure
sample_data = {
    'Timestamp': [pd.Timestamp('2025-06-03')] * 5,
    'Username': [f'user{i}' for i in range(1, 6)],
    'Kind of University': ['Private University', 'Public University', 'Private University', 'Public University', 'Private University'],
    'My school renovates old building into modern buildings': ['Agree', 'Disagree', 'Strongly Agree', 'Strongly Disagree', 'Neutral'],
    'My school ensures classroom capacity is strictly adhered to': ['Neutral', 'Disagree', 'Agree', 'Disagree', 'Strongly Agree'],
    'Spacious, ventilated and well-lit classroom [How accessible is this infrastructure in your institution?]': [3, 1, 3, 0, 2]
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
    # Identify opinion columns
    opinion_columns = [col for col in df.columns if col.startswith('My school')]
    
    # Create text column by concatenating opinion responses
    def create_text(row):
        return ' '.join([f"{col}: {row[col]}" for col in opinion_columns if pd.notna(row[col])])
    
    df['text'] = df.apply(create_text, axis=1)
    df['university_type'] = df['Kind of University'].apply(infer_university_type)
    df = df[df['university_type'] != 'unknown']
    
    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Predict sentiment
    X_test = vectorizer.transform(df['cleaned_text'])
    df['sentiment_svm'] = svm_model.predict(X_test)
    df['sentiment_rf'] = rf_model.predict(X_test)
    
    # Aggregate sentiment by university type (Random Forest)
    sentiment_summary = df.groupby(['university_type', 'sentiment_rf']).size().unstack(fill_value=0)
    sentiment_summary['total'] = sentiment_summary.sum(axis=1)
    for sentiment in ['positive', 'negative']:
        sentiment_summary[f'{sentiment}_percent'] = (sentiment_summary.get(sentiment, 0) / sentiment_summary['total'] * 100).round(2)
    
    # Aggregate infrastructure ratings
    infra_columns = [
        col for col in df.columns if 'How accessible is this infrastructure' in col
    ]
    infra_summary = df.groupby('university_type')[infra_columns].mean().round(2)
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sentiment_summary[['positive_percent', 'negative_percent']].reset_index(),
                   x='university_type', y='value', hue='variable')
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
    
    return df, sentiment_summary, infra_summary, plot_url

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                df = pd.read_csv(file_path)
                
                required_columns = ['Timestamp', 'Kind of University']
                if not all(col in df.columns for col in required_columns):
                    return render_template('results.html', error="File must contain 'Timestamp' and 'Kind of University' columns")
                
                df, sentiment_summary, infra_summary, plot_url = process_data(df)
            except Exception as e:
                return render_template('results.html', error=f"Error processing file: {str(e)}")
        else:
            return render_template('results.html', error="Please upload a valid CSV file")
    else:
        df = pd.DataFrame(sample_data)
        df.to_csv('raw_sample_data.csv', index=False)
        df, sentiment_summary, infra_summary, plot_url = process_data(df)
    
    # Save results to XLSX
    output_file = f'sentiment_analysis_results_{uuid.uuid4()}.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df[['Timestamp', 'Username', 'Kind of University', 'text', 'university_type', 'sentiment_svm', 'sentiment_rf']].to_excel(writer, sheet_name='Raw_Data', index=False)
        sentiment_summary.to_excel(writer, sheet_name='Sentiment_Summary')
        infra_summary.to_excel(writer, sheet_name='Infrastructure_Summary')
    
    # Prepare data for template
    raw_data = df[['Timestamp', 'Kind of University', 'text', 'university_type', 'sentiment_svm', 'sentiment_rf']].to_dict(orient='records')
    summary_data = sentiment_summary.to_dict()
    infra_data = infra_summary.to_dict()
    svm_accuracy = svm_metrics['accuracy']
    rf_accuracy = rf_metrics['accuracy']
    
    return render_template('results.html', raw_data=raw_data, summary_data=summary_data,
                         infra_data=infra_data, plot_url=plot_url, svm_accuracy=svm_accuracy,
                         rf_accuracy=rf_accuracy, output_file=output_file)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    