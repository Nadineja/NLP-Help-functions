import snscrape.modules.twitter as sntwitter
import pandas as pd

# The scraper automatically scrapes local data.

tweetstrend = []
# Using TwitterSearchScraper to scrape data and append tweets to list
for i, tweet in enumerate(sntwitter.TwitterTrendsScraper().get_items()):
    if i > 100:
        break
    tweetstrend.append([tweet])

# Creating a dataframe from the tweets list above
tweetstrends_df1 = pd.DataFrame(tweetstrend, columns=['Trends'])
print(tweetstrends_df1)
tweetstrends_df1.to_csv(r'C:/Users/jamme/Desktop/trends_tweets.csv', encoding='utf-8')




