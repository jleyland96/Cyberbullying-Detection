import tweepy
import csv
import re
import time

CONSUMER_KEY = ""
CONSUMER_SECRET = ""
OAUTH_TOKEN = ""
OAUTH_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

count = 0

with open('tweets_7K_raw.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    with open('twitter_7K.csv', mode='w') as csv_write_file:
        csv_writer = csv.writer(csv_write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["tweet_id", "tweet", "label", "role"])

        for row in csv_reader:
            count += 1
            print(count)
            tweet_id = row[0]
            label = row[2]
            role = row[6]

            # WRITE THE TWEET IF WE GET A HIT
            try:
                tweet = api.get_status(tweet_id)
                tweet_text = tweet.text
                tweet_text = re.sub('[\\n]', '', tweet_text)
                print(tweet_text)
                csv_writer.writerow([tweet_id, tweet_text, label, role])
            except tweepy.TweepError as e:
                print(e)
                continue
