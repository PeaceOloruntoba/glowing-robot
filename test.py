import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import uuid
from datetime import datetime

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')  # Fix for missing punkt_tab resource
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
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

# Sample dataset
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
        "The private university's labs are top-notch, best I've seen."
    ],
    'date': [datetime.now()] * 10,
    'username': [f'user{i}' for i in range(1, 11)],
    'university_type': ['private', 'public', 'private', 'public', 'private', 'public', 'public', 'private', 'public', 'private']
}

# Create DataFrame
df = pd.DataFrame(sample_data)

# Save raw data
df.to_csv('raw_sample_data.csv', index=False)

# Preprocess text
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Filter out rows where university type is unknown (not needed for sample data, but kept for generality)
df = df[df['university_type'] != 'unknown']

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)
df['compound_score'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Aggregate sentiment by university type
sentiment_summary = df.groupby(['university_type', 'sentiment']).size().unstack(fill_value=0)
sentiment_summary['total'] = sentiment_summary.sum(axis=1)
for sentiment in ['positive', 'negative', 'neutral']:
    sentiment_summary[f'{sentiment}_percent'] = (sentiment_summary[sentiment] / sentiment_summary['total'] * 100).round(2)

# Export results to XLSX
output_file = f'sentiment_analysis_results_{uuid.uuid4()}.xlsx'
df.to_excel(output_file, index=False, sheet_name='Raw_Data')
with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
    sentiment_summary.to_excel(writer, sheet_name='Summary')

# Create a chart for sentiment distribution by university type
plt.figure(figsize=(10, 6))
sentiment_summary[['positive_percent', 'negative_percent', 'neutral_percent']].plot(kind='bar', stacked=True, color=['#28a745', '#dc3545', '#ffc107'])
plt.title('Sentiment Distribution by University Type')
plt.xlabel('University Type')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')

print(f"Raw data saved to 'raw_sample_data.csv'")
print(f"Results exported to {output_file}")
print(f"Sentiment distribution plot saved as 'sentiment_distribution.png'")