import tweepy

CONSUMER_KEY = "Dpk7aD4LQh3UV1lYQ9VqbOx3n"
CONSUMER_SECRET = "UIkY1GGMlZtU6gGtGdKdpKEIWmgg9kf9U4RFe3ZbmpHkb0Upod"
OAUTH_TOKEN = "308075281-G4ei6R7t9pCe4TNPKnnc4Ye68phLmYjfPeI35SZZ"
OAUTH_TOKEN_SECRET = "3gCcDa9BnxlTDzqewRBGRtREAfKSriKJPMPUZjk9tMBkV"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)

tweet = api.get_status(572342978255048705)
print(tweet.text)