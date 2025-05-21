import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests # For static website scraping
from bs4 import BeautifulSoup # For parsing HTML
# from selenium import webdriver # Uncomment if you need Selenium
# from selenium.webdriver.common.by import By # Uncomment if you need Selenium
# from selenium.webdriver.chrome.service import Service # Uncomment if you need Selenium
# from webdriver_manager.chrome import ChromeDriverManager # Uncomment if you need Selenium

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

# --- Class for Sentiment Analysis (remains the same) ---
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

# --- NEW Class for Web Data Collection ---
class WebScraper:
    def __init__(self):
        # Initialize Selenium if needed, otherwise no special init for requests/BeautifulSoup
        # self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())) # Uncomment for Selenium
        pass

    def fetch_data_from_url(self, url, university_name, university_type, use_selenium=False):
        """
        Fetches text content from a given URL.
        You'll need to customize the parsing logic for each website.
        """
        data_entries = []
        print(f"Attempting to fetch data from: {url}")

        try:
            if use_selenium:
                # Example with Selenium (requires customization per site)
                # print("Using Selenium...")
                # self.driver.get(url)
                # time.sleep(5) # Wait for page to load dynamic content
                # soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                # # Example: Find all paragraph tags that might contain feedback
                # comments = soup.find_all('p', class_='comment-text') # Adjust this selector!
                # for comment in comments:
                #     text = comment.get_text(strip=True)
                #     if text:
                #         data_entries.append({
                #             'text': text,
                #             'university_type': university_type,
                #             'university_name': university_name
                #         })
                pass # Placeholder if not using Selenium right now
            else:
                # Example with requests and BeautifulSoup (for static content)
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}) # Add a User-Agent
                response.raise_for_status() # Raise an exception for HTTP errors
                soup = BeautifulSoup(response.text, 'html.parser')

                # !!! VERY IMPORTANT: Customize these selectors based on the website you are scraping !!!
                # You'll need to inspect the target website's HTML to find the correct
                # CSS selectors or HTML tags/classes for comments/feedback.

                # Example 1: Scraping comments from a hypothetical blog
                # comments = soup.find_all('div', class_='comment-content')
                # for comment in comments:
                #     text = comment.get_text(separator=' ', strip=True)
                #     if text:
                #         data_entries.append({
                #             'text': text,
                #             'university_type': university_type,
                #             'university_name': university_name
                #         })

                # Example 2: Scraping paragraphs or list items on a news page
                # This is a generic example, you'll need to make it more specific
                # to avoid irrelevant text.
                main_content_div = soup.find('div', class_='article-body') # Or a similar container
                if main_content_div:
                    paragraphs = main_content_div.find_all('p')
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        # Filter for reasonable length and content related to universities
                        if len(text) > 50 and any(kw in text.lower() for kw in ['unilag', 'anchor university', 'infrastructure', 'facilities', 'hostels']):
                            data_entries.append({
                                'text': text,
                                'university_type': university_type,
                                'university_name': university_name
                            })
                
                # Example 3: Searching for specific elements that contain reviews/feedback
                # You'll need to adapt this greatly based on where the data is.
                # For instance, if you find a specific section for student feedback, target that.
                
                # IMPORTANT: You'll likely need to scrape *multiple pages* for each source
                # (e.g., next pages of search results, next pages of comments).
                # This requires identifying pagination links or "load more" buttons.

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
        # except Exception as e: # For Selenium errors
        #     print(f"Selenium error on {url}: {e}")

        return data_entries

    def close(self):
        # if hasattr(self, 'driver') and self.driver: # Uncomment for Selenium
        #     self.driver.quit() # Uncomment for Selenium
        pass

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Configuration (No bearer token needed!) ---
    # We are no longer using Twitter API, so no bearer token is required.
    print("--- Running Sentiment Analysis without Twitter API ---")

    # --- UPDATED DATA SOURCES (Replace with actual URLs you find) ---
    # This is the most crucial part: you need to find actual websites
    # that contain discussions about UNILAG and Anchor University infrastructure.

    # Example placeholders - YOU MUST FIND REAL URLs AND THEIR CSS SELECTORS
    # A good strategy is to perform Google searches like:
    # "UNILAG infrastructure student forum"
    # "Anchor University facilities review"
    # "Nairaland UNILAG hostel discussion"
    # "news about UNILAG campus conditions"

    data_sources = [
        # UNILAG Sources
        {'url': 'https://www.nairaland.com/forum/topic/about-unilag-hostels', 'uni_name': 'UNILAG', 'uni_type': 'Public', 'selector_hint': 'div.body'}, # Example Nairaland thread
        {'url': 'https://some-edu-blog.com/tag/unilag-campus-life-review', 'uni_name': 'UNILAG', 'uni_type': 'Public', 'selector_hint': 'p.post-content'}, # Example blog
        # {'url': 'https://unilag-student-forum.org/infrastructure-feedback', 'uni_name': 'UNILAG', 'uni_type': 'Public', 'selector_hint': 'div.user-comment'}, # Hypothetical forum

        # Anchor University Sources
        {'url': 'https://www.nairaland.com/forum/topic/anchor-university-facilities', 'uni_name': 'Anchor University', 'uni_type': 'Private', 'selector_hint': 'div.body'}, # Example Nairaland thread
        {'url': 'https://another-edu-blog.net/aul-student-experience', 'uni_name': 'Anchor University', 'uni_type': 'Private', 'selector_hint': 'div.entry-content'}, # Example blog
        # {'url': 'https://anchoruni-alumni.com/campus-upgrades', 'uni_name': 'Anchor University', 'uni_type': 'Private', 'selector_hint': 'span.review-text'}, # Hypothetical alumni site
    ]

    # --- Phase 1: Data Collection (from Web) ---
    print("--- Phase 1: Data Collection (from Web) ---")
    scraper = WebScraper()
    all_raw_data = []

    for source in data_sources:
        # You'll need to implement the actual parsing logic INSIDE fetch_data_from_url
        # based on the 'selector_hint' or by directly modifying the method.
        # This loop just calls the fetcher.
        fetched_entries = scraper.fetch_data_from_url(
            url=source['url'],
            university_name=source['uni_name'],
            university_type=source['uni_type'],
            use_selenium=False # Set to True if the site requires JavaScript rendering
        )
        all_raw_data.extend(fetched_entries)
    
    # Clean up Selenium driver if used
    scraper.close()

    raw_df = pd.DataFrame(all_raw_data)
    print(f"Total raw text entries collected: {len(raw_df)}")
    if raw_df.empty:
        print("No data collected. Please ensure your URLs are correct and your scraping logic (CSS selectors) is accurate for the target websites.")
        exit()

    # --- Phase 2: Data Preprocessing and Structuring ---
    print("\n--- Phase 2: Data Preprocessing and Structuring ---")
    processed_df = raw_df.drop_duplicates(subset=['text']).copy()
    processed_df.rename(columns={'text': 'infrastructure_feedback'}, inplace=True)

    # General text cleaning (social media specific cleaning like @ and # might be less relevant now)
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    # You might remove @ and # if the source occasionally uses them, or keep them if they are meaningful in context.
    # processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'@\w+', '', x))
    # processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'#\w+', '', x))
    processed_df['infrastructure_feedback'] = processed_df['infrastructure_feedback'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    print(f"Processed {len(processed_df)} unique text entries.")
    print("Sample of processed data:")
    print(processed_df.head())

    # --- Phase 3: Sentiment Analysis (remains the same) ---
    print("\n--- Phase 3: Performing Sentiment Analysis ---")
    sentiment_analyzer = SentimentAnalyzerForInfrastructure()
    df_with_sentiment = sentiment_analyzer.analyze_dataframe(processed_df.copy(), 'infrastructure_feedback')

    print("Sentiment analysis complete. Sample results:")
    print(df_with_sentiment[['university_name', 'university_type', 'infrastructure_feedback', 'compound', 'sentiment_category']].head())

    # --- Phase 4: Result Aggregation and Presentation (remains the same) ---
    if not df_with_sentiment.empty:
        print("\n--- Phase 4: Aggregating and Presenting Results ---")

        overall_sentiment_counts = df_with_sentiment['sentiment_category'].value_counts(normalize=True) * 100
        print("\nOverall Sentiment Distribution:")
        print(overall_sentiment_counts.round(2))

        sentiment_by_type = df_with_sentiment.groupby('university_type')['sentiment_category'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution by University Type:")
        print(sentiment_by_type.unstack().round(2))

        avg_compound_by_type = df_with_sentiment.groupby('university_type')['compound'].mean()
        print("\nAverage Compound Sentiment Score by University Type:")
        print(avg_compound_by_type.round(3))

        sentiment_by_uni = df_with_sentiment.groupby('university_name')['sentiment_category'].value_counts(normalize=True) * 100
        print("\nSentiment Distribution by Specific University:")
        print(sentiment_by_uni.unstack().round(2))

        avg_compound_by_uni = df_with_sentiment.groupby('university_name')['compound'].mean()
        print("\nAverage Compound Sentiment Score by Specific University:")
        print(avg_compound_by_uni.round(3))

        print("\nTop 3 Most Positive Infrastructure Feedback (UNILAG):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'UNILAG'].sort_values(by='compound', ascending=False).head(3)[['infrastructure_feedback', 'compound']])
        print("\nTop 3 Most Negative Infrastructure Feedback (UNILAG):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'UNILAG'].sort_values(by='compound', ascending=True).head(3)[['infrastructure_feedback', 'compound']])

        print("\nTop 3 Most Positive Infrastructure Feedback (Anchor University):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'Anchor University'].sort_values(by='compound', ascending=False).head(3)[['infrastructure_feedback', 'compound']])
        print("\nTop 3 Most Negative Infrastructure Feedback (Anchor University):")
        print(df_with_sentiment[df_with_sentiment['university_name'] == 'Anchor University'].sort_values(by='compound', ascending=True).head(3)[['infrastructure_feedback', 'compound']])

        output_folder = "sentiment_results_web_scrape" # Changed folder name
        os.makedirs(output_folder, exist_ok=True)

        output_csv_path = os.path.join(output_folder, "unilag_anchor_university_web_sentiment_detailed.csv")
        df_with_sentiment.to_csv(output_csv_path, index=False)
        print(f"\nDetailed sentiment analysis results saved to: {output_csv_path}")

        output_xlsx_path = os.path.join(output_folder, "unilag_anchor_university_web_sentiment_detailed.xlsx")
        df_with_sentiment.to_excel(output_xlsx_path, index=False, sheet_name="Detailed Sentiment")
        print(f"Detailed sentiment analysis results saved to: {output_xlsx_path}")

        summary_xlsx_path = os.path.join(output_folder, "unilag_anchor_university_web_sentiment_summary.xlsx")
        with pd.ExcelWriter(summary_xlsx_path) as writer:
            overall_sentiment_counts.to_excel(writer, sheet_name='Overall Sentiment Distribution')
            sentiment_by_type.to_excel(writer, sheet_name='Sentiment by Type')
            avg_compound_by_type.to_excel(writer, sheet_name='Avg Compound by Type')
            sentiment_by_uni.to_excel(writer, sheet_name='Sentiment by University Name')
            avg_compound_by_uni.to_excel(writer, sheet_name='Avg Compound by University Name')
        print(f"Summary sentiment statistics saved to: {summary_xlsx_path}")

        sns.set_style("whitegrid")

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_with_sentiment, x='university_type', hue='sentiment_category', palette='viridis', order=['Public', 'Private'])
        plt.title('Sentiment Towards Infrastructure: Public (UNILAG) vs. Private (Anchor University) [Web Scraped]') # Updated title
        plt.xlabel('University Type')
        plt.ylabel('Number of Text Entries') # Updated label
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "sentiment_by_university_type_web_plot.png"))
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_with_sentiment, x='university_type', y='compound', palette='coolwarm', errorbar=None)
        plt.title('Average Compound Sentiment Score by University Type [Web Scraped]') # Updated title
        plt.xlabel('University Type')
        plt.ylabel('Average Compound Score')
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "avg_compound_by_university_type_web_plot.png"))
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_with_sentiment, x='university_name', hue='sentiment_category', palette='viridis', order=['UNILAG', 'Anchor University'])
        plt.title('Sentiment Towards Infrastructure: UNILAG vs. Anchor University [Web Scraped]') # Updated title
        plt.xlabel('University Name')
        plt.ylabel('Number of Text Entries') # Updated label
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "sentiment_by_specific_university_web_plot.png"))
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_with_sentiment, x='university_name', y='compound', palette='coolwarm', errorbar=None, order=['UNILAG', 'Anchor University'])
        plt.title('Average Compound Sentiment Score by Specific University [Web Scraped]') # Updated title
        plt.xlabel('University Name')
        plt.ylabel('Average Compound Score')
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "avg_compound_by_specific_university_web_plot.png"))
        plt.show()

        print("\nAll phases complete. Check your console output, generated plots, and 'sentiment_results_web_scrape' folder for output files.")
    else:
        print("No data to aggregate, visualize, or save. Please check previous phases.")
