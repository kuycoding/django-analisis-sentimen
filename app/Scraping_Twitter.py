import tweepy
import io
import csv
from tweepy import streaming 

consumer_key = "x9zuaTU1HbDErb9amFHGyY2kY"
consumer_secret = "6tZlXskc8QD9f8j9FthqZzLFHN6E2aH1nINfMlqJu2pWg3MsEc"
access_token = "1278541122512408576-ClQSTF1hM173Maz8Sy6n109zJxiTS4"
access_token_secret = "JqDkBY5iamljZPcrs0ayv0I26UoKa3TEnxZMGZT7EhXoz"

def authenticate(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    global api
    api = tweepy.API(auth, wait_on_rate_limit=True)

def search_twitter(keyword, count):
    filterKey = " -filter:retweets"
    tweets = tweepy.Cursor(api.search, q=keyword + filterKey, lang="in", tweet_mode='extended').items(int(count))
    return tweets
