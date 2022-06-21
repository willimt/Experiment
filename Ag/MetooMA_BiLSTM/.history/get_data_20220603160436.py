import tweepy

consumer_key = "willimt_app_for_api"

consumer_secret = " "

access_key = " "

access_secret = " "

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
