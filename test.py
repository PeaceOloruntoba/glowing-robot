import snscrape.modules.twitter as sntwitter

for tweet in sntwitter.TwitterSearchScraper('Anchor University Lagos since:2024-01-01').get_items():
    print(tweet.content)
    break
