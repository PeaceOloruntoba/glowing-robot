# full_sentiment_analysis_script.py

import tweepy
import pandas as pd
import datetime
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os # For environment variables

# --- NLTK Downloads (Run once) ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('corpora/stopwords.zip')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt/PY3/english.pickle')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- Class for Sentiment Analysis ---
class SentimentAnalyzerForInfrastructure:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [w for w in tokens if not w in self.stop_words]
            return " ".join(tokens)
        return ""

    def analyze_sentiment(self, text):
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        vs = self.analyzer.polarity_scores(text)
        return vs

    def categorize_sentiment(self, compound_score):
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def analyze_dataframe(self, df, text_column):
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        df['sentiment_scores'] = df['processed_text'].apply(self.analyze_sentiment)
        df = pd.concat([df, pd.json_normalize(df['sentiment_scores'])], axis=1)
        df['sentiment_category'] = df['compound'].apply(self.categorize_sentiment)
        return df

# --- Class for Twitter Data Collection ---
class TwitterScraper:
    def __init__(self, bearer_token):
        self.client = tweepy.Client(bearer_token)

    def search_tweets(self, query, max_results=100, start_time=None, end_time=None):
        all_tweets = []
        try:
            # You might need to implement pagination for more tweets
            response = self.client.search_recent_tweets(
                query,
                tweet_fields=["created_at", "text", "public_metrics"],
                max_results=max_results,
                start_time=start_time,
                end_time=end_time
            )

            if response.data:
                for tweet in response.data:
                    tweet_info = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                        'reply_count': tweet.public_metrics.get('reply_count', 0),
                        'like_count': tweet.public_metrics.get('like_count', 0),
                        'quote_count': tweet.public_metrics.get('quote_count', 0),
                    }
                    all_tweets.append(tweet_info)
            return all_tweets
        except tweepy.TweepyException as e:
            print(f"Error fetching tweets for '{query}': {e}")
            return []

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Set your Twitter Bearer Token as an environment variable
    # e.g., export TWITTER_BEARER_TOKEN="your_token_here"
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        print("WARNING: TWITTER_BEARER_TOKEN environment variable not set.")
        print("Please set it or replace with your token for actual data collection.")
        # Using a dummy token for local testing if not set, but this won't work for actual API calls
        bearer_token = "DUMMY_TOKEN_REPLACE_ME" # This will cause an error if used for actual API calls

    # Define keywords for Public and Private Universities (expand this list significantly)
    public_uni_keywords = [
        "University of Ibadan infrastructure", "UI facilities", "UNILAG infrastructure",
        "Ahmadu Bello University facilities", "OAU infrastructure",
        "Federal University of Technology Akure facilities", "FUTA infrastructure",
        "Nigerian public university buildings", "public university hostels Nigeria"
    ]
    private_uni_keywords = [
        "Covenant University infrastructure", "CU facilities", "Redeemer's University infrastructure",
        "Babcock University facilities", "American University of Nigeria infrastructure",
        "private university facilities Nigeria", "Nigerian private university hostels"
    ]
    MAX_TWEETS_PER_QUERY = 50 # Adjust based on your needs and API limits
    # You might want to define a time range for tweets, e.g., last 30 days
    # end_time = datetime.datetime.now(datetime.timezone.utc)
    # start_time = end_time - datetime.timedelta(days=30)


    # --- Phase 1: Data Collection ---
    print("--- Phase 1: Data Collection (from Twitter) ---")
    scraper = TwitterScraper(bearer_token)
    all_raw_tweets = []

    for keyword in public_uni_keywords:
        tweets = scraper.search_tweets(keyword, max_results=MAX_TWEETS_PER_QUERY)
        for tweet in tweets:
            tweet['university_type'] = 'Public'
            all_raw_tweets.append(tweet)

    for keyword in private_uni_keywords:
        tweets = scraper.search_tweets(keyword, max_results=MAX_TWEETS_PER_QUERY)
        for tweet in tweets:
            tweet['university_type'] = 'Private'
            all_raw_tweets.append(tweet)

    raw_df = pd.DataFrame(all_raw_tweets)
    print(f"Collected {len(raw_df)} raw tweets.")
    if raw_df.empty:
        print("No data collected. Exiting.")
        exit() # Exit if no data is collected

    # --- Phase 2: Data Preprocessing and Structuring ---
    print("\n--- Phase 2: Data Preprocessing and Structuring ---")
    processed_df = raw_df.drop_duplicates(subset=['id']).copy()
    processed_df = processed_df[['text', 'university_type']].copy() # Keep only necessary columns
    processed_df.rename(columns={'text': 'infrastructure_feedback'}, inplace=True)

    # Social media specific cleaning
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'@\w+', '', x))
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'#\w+', '', x))
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    print(f"Processed {len(processed_df)} unique tweets.")
    print("Sample of processed data:")
    print(processed_df.head())


    # --- Phase 3: Sentiment Analysis ---
    print("\n--- Phase 3: Performing Sentiment Analysis ---")
    sentiment_analyzer = SentimentAnalyzerForInfrastructure()
    df_with_sentiment = sentiment_analyzer.analyze_dataframe(processed_df.copy(), 'infrastructure_feedback')

    print("Sentiment analysis complete. Sample results:")
    print(df_with_sentiment[['university_type', 'infrastructure_feedback', 'compound', 'sentiment_category']].head())


    # --- Phase 4: Result Aggregation and Presentation ---
    print("\n--- Phase 4: Aggregating and Presenting Results ---")

    # Overall sentiment distribution
    overall_sentiment_counts = df_with_sentiment['sentiment_category'].value_counts(normalize=True) * 100
    print("\nOverall Sentiment Distribution:")
    print(overall_sentiment_counts.round(2))

    # Sentiment distribution by university type
    sentiment_by_type = df_with_sentiment.groupby('university_type')['sentiment_category'].value_counts(normalize=True) * 100
    print("\nSentiment Distribution by University Type:")
    print(sentiment_by_type.unstack().round(2))

    # Average compound score by university type
    avg_compound_by_type = df_with_sentiment.groupby('university_type')['compound'].mean()
    print("\nAverage Compound Sentiment Score by University Type:")
    print(avg_compound_by_type.round(3))

    # Top/Bottom Tweets
    print("\nTop 3 Most Positive Infrastructure Feedback (Overall):")
    print(df_with_sentiment.sort_values(by='compound', ascending=False).head(3)[['university_type', 'infrastructure_feedback', 'compound']])
    print("\nTop 3 Most Negative Infrastructure Feedback (Overall):")
    print(df_with_sentiment.sort_values(by='compound', ascending=True).head(3)[['university_type', 'infrastructure_feedback', 'compound']])

    # --- Visualizations ---
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_with_sentiment, x='university_type', hue='sentiment_category', palette='viridis', order=['Public', 'Private'])
    plt.title('Sentiment Towards Infrastructure: Public vs. Private Universities (Nigeria)')
    plt.xlabel('University Type')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_with_sentiment, x='university_type', y='compound', palette='coolwarm', errorbar=None)
    plt.title('Average Compound Sentiment Score by University Type')
    plt.xlabel('University Type')
    plt.ylabel('Average Compound Score')
    plt.ylim(-0.5, 0.5)
    plt.tight_layout()
    plt.show()

    print("\nAll phases complete. Check your console output and generated plots for results.")