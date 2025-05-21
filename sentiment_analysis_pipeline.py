# full_sentiment_analysis_script.py (modified for specific universities and output)

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
import os

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
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAL2B1wEAAAAALdWbczUI4%2B38yDeb9ztZ27XlD8k%3DHHlZOJVEmzt1mjYtALN7OGotmWMXnGIvxv4Nxtg3iULdA6ful3"
    if not bearer_token:
        print("WARNING: TWITTER_BEARER_TOKEN environment variable not set.")
        print("Please set it or replace with your token for actual data collection.")
        bearer_token = "DUMMY_TOKEN_REPLACE_ME" # This will cause an error if used for actual API calls

    # --- UPDATED KEYWORDS FOR UNILAG AND ANCHOR UNIVERSITY ---
    public_uni_keywords = [
        "UNILAG infrastructure", "University of Lagos facilities", "UNILAG hostels",
        "UNILAG lecture halls", "UNILAG library", "UNILAG power supply", "UNILAG maintenance",
        "UNILAG roads", "UNILAG environment", "UNILAG structures", "UNILAG buildings",
        "UNILAG lecture theatre", "UNILAG laboratory", "UNILAG student facilities"
    ]
    private_uni_keywords = [
        "Anchor University infrastructure", "Anchor University Lagos facilities", "Anchor University hostels",
        "Anchor University lecture halls", "Anchor University library", "Anchor University power supply",
        "Anchor University maintenance", "Anchor University roads", "Anchor University environment",
        "Anchor University structures", "Anchor University buildings", "Anchor University lecture theatre",
        "Anchor University laboratory", "Anchor University student facilities", "AUL infrastructure",
        "AUL facilities", "AUL hostels"
    ]
    MAX_TWEETS_PER_QUERY = 100 # Maximum per API call for recent search

    # --- Phase 1: Data Collection ---
    print("--- Phase 1: Data Collection (from Twitter) ---")
    scraper = TwitterScraper(bearer_token)
    all_raw_tweets = []

    # Collect for UNILAG
    for keyword in public_uni_keywords:
        tweets = scraper.search_tweets(keyword, max_results=MAX_TWEETS_PER_QUERY)
        for tweet in tweets:
            tweet['university_type'] = 'Public'
            tweet['university_name'] = 'UNILAG' # Add specific university name
            all_raw_tweets.append(tweet)
    print(f"Collected {len([t for t in all_raw_tweets if t.get('university_name') == 'UNILAG'])} tweets for UNILAG.")

    # Collect for Anchor University
    for keyword in private_uni_keywords:
        tweets = scraper.search_tweets(keyword, max_results=MAX_TWEETS_PER_QUERY)
        for tweet in tweets:
            tweet['university_type'] = 'Private'
            tweet['university_name'] = 'Anchor University' # Add specific university name
            all_raw_tweets.append(tweet)
    print(f"Collected {len([t for t in all_raw_tweets if t.get('university_name') == 'Anchor University'])} tweets for Anchor University.")

    raw_df = pd.DataFrame(all_raw_tweets)
    print(f"Total raw tweets collected: {len(raw_df)}")
    if raw_df.empty:
        print("No data collected. Exiting.")
        exit()

    # --- Phase 2: Data Preprocessing and Structuring ---
    print("\n--- Phase 2: Data Preprocessing and Structuring ---")
    processed_df = raw_df.drop_duplicates(subset=['id']).copy()
    # Keep university_name column
    processed_df = processed_df[['text', 'university_type', 'university_name']].copy()
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
    print(df_with_sentiment[['university_name', 'university_type', 'infrastructure_feedback', 'compound', 'sentiment_category']].head())


    # --- Phase 4: Result Aggregation and Presentation ---
    # This entire block should be within the 'if not df_with_sentiment.empty:' check
    if not df_with_sentiment.empty:
        print("\n--- Phase 4: Aggregating and Presenting Results ---")

        # --- Aggregation and Display (as before) ---
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

        # Sentiment distribution by specific university
        sentiment_by_uni = df_with_sentiment.groupby('university_name')['sentiment_category'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution by Specific University:")
        print(sentiment_by_uni.unstack().round(2))

        # Average compound score by specific university
        avg_compound_by_uni = df_with_sentiment.groupby('university_name')['compound'].mean()
        print("\nAverage Compound Sentiment Score by Specific University:")
        print(avg_compound_by_uni.round(3))


        # Top/Bottom Tweets
        print("\nTop 3 Most Positive Infrastructure Feedback (UNILAG):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'UNILAG'].sort_values(by='compound', ascending=False).head(3)[['infrastructure_feedback', 'compound']])
        print("\nTop 3 Most Negative Infrastructure Feedback (UNILAG):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'UNILAG'].sort_values(by='compound', ascending=True).head(3)[['infrastructure_feedback', 'compound']])

        print("\nTop 3 Most Positive Infrastructure Feedback (Anchor University):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'Anchor University'].sort_values(by='compound', ascending=False).head(3)[['infrastructure_feedback', 'compound']])
        print("\nTop 3 Most Negative Infrastructure Feedback (Anchor University):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'Anchor University'].sort_values(by='compound', ascending=True).head(3)[['infrastructure_feedback', 'compound']])


        # --- Saving Results to Files ---
        output_folder = "sentiment_results"
        os.makedirs(output_folder, exist_ok=True) # Create folder if it doesn't exist

        # 1. Save the detailed DataFrame (df_with_sentiment)
        # As CSV:
        output_csv_path = os.path.join(output_folder, "unilag_anchor_university_sentiment_analysis_detailed.csv")
        df_with_sentiment.to_csv(output_csv_path, index=False)
        print(f"\nDetailed sentiment analysis results saved to: {output_csv_path}")

        # As XLSX (requires openpyxl)
        # You might need to install openpyxl if you haven't already: pip install openpyxl
        output_xlsx_path = os.path.join(output_folder, "unilag_anchor_university_sentiment_analysis_detailed.xlsx")
        df_with_sentiment.to_excel(output_xlsx_path, index=False, sheet_name="Detailed Sentiment")
        print(f"Detailed sentiment analysis results saved to: {output_xlsx_path}")

        # 2. Save Summary Statistics to a separate Excel file with multiple sheets
        summary_xlsx_path = os.path.join(output_folder, "unilag_anchor_university_sentiment_summary.xlsx")
        with pd.ExcelWriter(summary_xlsx_path) as writer:
            overall_sentiment_counts.to_excel(writer, sheet_name='Overall Sentiment Distribution')
            sentiment_by_type.to_excel(writer, sheet_name='Sentiment by Type')
            avg_compound_by_type.to_excel(writer, sheet_name='Avg Compound by Type')
            sentiment_by_uni.to_excel(writer, sheet_name='Sentiment by University Name')
            avg_compound_by_uni.to_excel(writer, sheet_name='Avg Compound by University Name')
        print(f"Summary sentiment statistics saved to: {summary_xlsx_path}")


        # --- Visualizations (as before) ---
        sns.set_style("whitegrid")

        # Plot 1: Sentiment Distribution by University Type
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_with_sentiment, x='university_type', hue='sentiment_category', palette='viridis', order=['Public', 'Private'])
        plt.title('Sentiment Towards Infrastructure: Public (UNILAG) vs. Private (Anchor University)')
        plt.xlabel('University Type')
        plt.ylabel('Number of Tweets')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_folder, "sentiment_by_university_type_plot.png"))
        plt.show()

        # Plot 2: Average Compound Sentiment Score by University Type
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_with_sentiment, x='university_type', y='compound', palette='coolwarm', errorbar=None)
        plt.title('Average Compound Sentiment Score by University Type')
        plt.xlabel('University Type')
        plt.ylabel('Average Compound Score')
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_folder, "avg_compound_by_university_type_plot.png"))
        plt.show()

        # Plot 3: Sentiment Distribution by Specific University
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_with_sentiment, x='university_name', hue='sentiment_category', palette='viridis', order=['UNILAG', 'Anchor University'])
        plt.title('Sentiment Towards Infrastructure: UNILAG vs. Anchor University')
        plt.xlabel('University Name')
        plt.ylabel('Number of Tweets')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_folder, "sentiment_by_specific_university_plot.png"))
        plt.show()

        # Plot 4: Average Compound Sentiment Score by Specific University
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_with_sentiment, x='university_name', y='compound', palette='coolwarm', errorbar=None, order=['UNILAG', 'Anchor University'])
        plt.title('Average Compound Sentiment Score by Specific University')
        plt.xlabel('University Name')
        plt.ylabel('Average Compound Score')
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()
        # Save the plot
        plt.savefig(os.path.join(output_folder, "avg_compound_by_specific_university_plot.png"))
        plt.show()

        print("\nAll phases complete. Check your console output, generated plots, and 'sentiment_results' folder for output files.")
    else: # This 'else' correctly belongs to the 'if not df_with_sentiment.empty:' check for Phase 4
        print("No data to aggregate, visualize, or save. Please check previous phases.")
        