import tweepy

consumer_key = "nAXDj5HBmywFoJmqEsFH7WOvJ"

consumer_secret = "n06aEvR4fVVz5xb9VwlWyEmHxhEKfGU1UDMfhgKbiAEuCN6oJj"

access_key = "1125761673581580288-m3K4VchnaDpxlJYfpCHa2CJv9Mncd3"

access_secret = "POxGHB86ZAf3DBOn1OgSHO4vOBnLFVNz23sr5AvMAZe1O"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
api.get_st
data = api.get_status(id=1015249783408164864)

print(data)