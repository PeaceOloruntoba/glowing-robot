import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# --- Sentiment Analyzer Class ---
class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        return " ".join(tokens)

    def analyze_sentiment(self, text):
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        return self.analyzer.polarity_scores(text)

    def categorize(self, compound):
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def analyze_df(self, df, col):
        df['cleaned'] = df[col].apply(self.preprocess_text)
        df['scores'] = df['cleaned'].apply(self.analyze_sentiment)
        df = pd.concat([df, pd.json_normalize(df['scores'])], axis=1)
        df['category'] = df['compound'].apply(self.categorize)
        return df

# --- Web Scraping Function ---
def scrape_tweets(keyword, max_tweets=100):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword).get_items()):
        if i >= max_tweets:
            break
        tweets.append({'text': tweet.content})
    return pd.DataFrame(tweets)

# --- Keywords ---
unilag_keywords = [
    "UNILAG infrastructure", "University of Lagos hostels", "UNILAG facilities"
]
anchor_keywords = [
    "Anchor University infrastructure", "AUL hostels", "Anchor University Lagos"
]

def collect_all():
    rows = []
    for kw in unilag_keywords:
        df = scrape_tweets(kw, 100)
        df['university'] = 'UNILAG'
        df['type'] = 'Public'
        rows.append(df)

    for kw in anchor_keywords:
        df = scrape_tweets(kw, 100)
        df['university'] = 'Anchor University'
        df['type'] = 'Private'
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

# --- Main Pipeline ---
if __name__ == "__main__":
    print("Collecting tweets...")
    raw_df = collect_all()
    raw_df.drop_duplicates(subset='text', inplace=True)
    print(f"Total tweets collected: {len(raw_df)}")

    analyzer = SentimentAnalyzer()
    result_df = analyzer.analyze_df(raw_df.copy(), 'text')

    # Summary Tables
    summary_counts = result_df.groupby(['university', 'category']).size().unstack(fill_value=0)
    avg_sentiment = result_df.groupby('university')['compound'].mean().round(3)

    # Save folder
    output_dir = "sentiment_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Save Chart ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=result_df, x='university', hue='category', palette='Set2')
    plt.title("Sentiment Towards Infrastructure")
    chart_path = os.path.join(output_dir, "sentiment_plot.png")
    plt.savefig(chart_path)
    plt.close()

    # --- Save Excel ---
    excel_path = os.path.join(output_dir, "sentiment_analysis.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name="Tweets", index=False)
        summary_counts.to_excel(writer, sheet_name="Sentiment Count")
        avg_sentiment.to_frame("Average Compound Score").to_excel(writer, sheet_name="Average Sentiment")

        # Insert chart into Excel (optional but cool!)
        workbook = writer.book
        worksheet = writer.sheets['Sentiment Count']
        worksheet.insert_image('G2', chart_path)

    print(f"\n✅ Sentiment analysis saved to: {excel_path}")
    print(f"📊 Chart saved to: {chart_path}")
