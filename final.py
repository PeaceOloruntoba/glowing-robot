import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tweepy
import uuid
from datetime import datetime, timedelta
import certifi
import os

# Set SSL certificate path
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# X API credentials (replace with your own)
API_KEY = "KzZ0NJVAKjdKHvxmkkEUlPIsl"
API_SECRET = "JO8D3iQUgCBVTdNPf7s3fmuJCfUzOkyhEdhEBzTK8dJqBRk9CU"
ACCESS_TOKEN = "1615778868824096768-HXrAl7O4Fen3FVDcIloDBlm3QF2kwr"
ACCESS_TOKEN_SECRET = "1QAU5nNkDig0dkvAep2HWQcyLfKCte9ZKDWCHVhJ2biyQ"

# Authenticate with X API
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

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

# Scrape X posts using tweepy
def scrape_x_posts(query, max_tweets=50, days_back=7):
    tweets = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    try:
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(max_tweets):
            text = tweet.full_text if hasattr(tweet, 'full_text') else tweet.text
            tweets.append({
                'text': text,
                'date': tweet.created_at,
                'username': tweet.user.screen_name
            })
    except Exception as e:
        print(f"Error scraping for query '{query}': {e}")
    return pd.DataFrame(tweets)

# Fallback sample data if scraping fails
sample_data = {
    'text': [
        "The lecture halls in this private university are modern and well-equipped.",
        "Public university infrastructure is outdated and needs repair.",
        "Private uni has amazing libraries and study spaces!",
        "The public uni's buildings are falling apart, very disappointing.",
        "Great facilities at this private institution, love the new tech labs.",
        "Public university lacks proper maintenance for classrooms."
    ],
    'date': [datetime.now()] * 6,
    'username': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6'],
    'university_type': ['private', 'public', 'private', 'public', 'private', 'public']
}

# Queries for scraping
queries = [
    '"public university infrastructure"',
    '"private university infrastructure"',
    '"state university facilities"',
    '"private college facilities"',
    '"university campus maintenance"'
]

# Scrape data
data_frames = []
for query in queries:
    df_temp = scrape_x_posts(query, max_tweets=50, days_back=7)
    if not df_temp.empty:
        data_frames.append(df_temp)

# Combine data or use sample data if scraping fails
if data_frames:
    df = pd.concat(data_frames, ignore_index=True)
    df.drop_duplicates(subset='text', inplace=True)
else:
    print("No data scraped. Using sample dataset.")
    df = pd.DataFrame(sample_data)

# Save raw scraped data
df.to_csv('raw_scraped_data.csv', index=False)

# Infer university type
df['university_type'] = df['text'].apply(infer_university_type)

# Filter out rows where university type is unknown
df = df[df['university_type'] != 'unknown']

# Preprocess text
df['cleaned_text'] = df['text'].apply(preprocess_text)

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
sentiment_summary[['positive_percent', 'negative_percent', 'neutral_percent']].plot(kind='bar', stacked=True)
plt.title('Sentiment Distribution by University Type')
plt.xlabel('University Type')
plt.ylabel('Percentage')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')

print(f"Raw data saved to 'raw_scraped_data.csv'")
print(f"Results exported to {output_file}")
print(f"Sentiment distribution plot saved as 'sentiment_distribution.png'")
