import httpx
from bs4 import BeautifulSoup
import json
import asyncio
import aiohttp
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
from typing import List, Dict, Any

class SocialMediaScraper:
    def __init__(self):
        self.http_client = httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "*/*",
            },
            timeout=httpx.Timeout(seconds=30)
        )
            
    async def scrape_instagram_profile(self, username: str) -> List[Dict]:
        """Scrape Instagram profile posts"""
        url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
        response = await self.http_client.get(url)
        if response.status_code == 200:
            data = json.loads(response.content)
            posts_data = []
            # Extract first 12 posts (available in profile info)
            for post in data["data"]["user"]["edge_owner_to_timeline_media"]["edges"]:
                post_data = {
                    'text': post['node'].get('accessibility_caption', ''),
                    'platform': 'Instagram',
                    'timestamp': datetime.datetime.now(),
                    'engagement': {
                        'likes': post['node']['edge_liked_by']['count'],
                        'comments': post['node']['edge_media_to_comment']['count']
                    }
                }
                posts_data.append(post_data)
            return posts_data
        return []

    async def scrape_facebook_page(self, page_url: str) -> List[Dict]:
        """Scrape Facebook page posts"""
        response = await self.http_client.get(page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            posts_data = []
            # Facebook uses dynamic content loading, this is simplified
            # In production, consider using Selenium for better results
            posts = soup.find_all('div', {'data-testid': 'post_message'})
            for post in posts:
                post_data = {
                    'text': post.get_text(strip=True),
                    'platform': 'Facebook',
                    'timestamp': datetime.datetime.now(),
                    'engagement': {
                        'likes': 0,
                        'comments': 0
                    }
                }
                posts_data.append(post_data)
            return posts_data
        return []

    async def scrape_social_media(self, instagram_usernames: List[str], facebook_urls: List[str]) -> List[Dict]:
        """Scrape data from both Instagram and Facebook"""
        all_posts = []
        
        # Scrape Instagram
        for username in instagram_usernames:
            posts = await self.scrape_instagram_profile(username)
            all_posts.extend(posts)
            
        # Scrape Facebook
        for url in facebook_urls:
            posts = await self.scrape_facebook_page(url)
            all_posts.extend(posts)
            
        return all_posts

class SentimentAnalyzerForInfrastructure:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [w for w in tokens if not w in self.stop_words]
            return " ".join(tokens)
        return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not text:
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        return self.analyzer.polarity_scores(text)
    
    def categorize_sentiment(self, compound_score: float) -> str:
        """Categorize sentiment based on compound score"""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        return "Neutral"
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Analyze sentiment for entire dataframe"""
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        df['sentiment_scores'] = df['processed_text'].apply(self.analyze_sentiment)
        df = pd.concat([df, pd.json_normalize(df['sentiment_scores'])], axis=1)
        df['sentiment_category'] = df['compound'].apply(self.categorize_sentiment)
        return df

async def main():
    # Initialize scraper and analyzer
    scraper = SocialMediaScraper()
    sentiment_analyzer = SentimentAnalyzerForInfrastructure()
    
    # Define universities and their social media handles
    universities = {
        'UNILAG': {
            'instagram': ['unilagofficial'],
            'facebook': ['https://www.facebook.com/unilagofficial']
        },
        'Anchor University': {
            'instagram': ['anchoruniversity'],
            'facebook': ['https://www.facebook.com/anchoruniversity']
        }
    }
    
    # Scrape data
    print("--- Phase 1: Data Collection ---")
    all_posts = []
    for uni_name, handles in universities.items():
        print(f"\nCollecting data for {uni_name}")
        
        # Collect Instagram posts
        instagram_posts = await scraper.scrape_social_media(
            handles['instagram'], 
            []  # Facebook URLs will be scraped separately
        )
        for post in instagram_posts:
            post['university_name'] = uni_name
            post['university_type'] = 'Public' if uni_name == 'UNILAG' else 'Private'
        all_posts.extend(instagram_posts)
        
        # Collect Facebook posts
        facebook_posts = await scraper.scrape_social_media(
            [],  # Instagram usernames
            handles['facebook']
        )
        for post in facebook_posts:
            post['university_name'] = uni_name
            post['university_type'] = 'Public' if uni_name == 'UNILAG' else 'Private'
        all_posts.extend(facebook_posts)
    
    print(f"\nTotal posts collected: {len(all_posts)}")
    
    if not all_posts:
        print("No data collected. Exiting.")
        return
    
    # Convert to DataFrame
    raw_df = pd.DataFrame(all_posts)
    
    # --- Phase 2: Data Preprocessing ---
    print("\n--- Phase 2: Data Preprocessing ---")
    processed_df = raw_df.drop_duplicates(subset=['text']).copy()
    processed_df = processed_df[['text', 'university_type', 'university_name']].copy()
    processed_df.rename(columns={'text': 'infrastructure_feedback'}, inplace=True)
    
    # Clean social media specific content
    processed_df['infrastructure_feedback'] = (
        processed_df['infrastructure_feedback']
        .apply(lambda x: re.sub(r'http\S+|www\S+', '', str(x)))
        .apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    )
    
    print(f"Processed {len(processed_df)} unique posts.")
    
    # --- Phase 3: Sentiment Analysis ---
    print("\n--- Phase 3: Performing Sentiment Analysis ---")
    df_with_sentiment = sentiment_analyzer.analyze_dataframe(
        processed_df.copy(), 
        'infrastructure_feedback'
    )
    
    print("Sentiment analysis complete. Sample results:")
    print(df_with_sentiment[
        ['university_name', 'university_type', 'infrastructure_feedback', 'compound', 'sentiment_category']
    ].head())
    
    # Rest of your visualization and reporting code remains the same...
    await scraper.http_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
    