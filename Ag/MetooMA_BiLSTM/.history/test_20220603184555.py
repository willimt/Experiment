from sys import flags
import requests
import tweepy
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

url = "https://api.twitter.com/2/tweets?ids=1050864924367433729"

payload={}
headers = {
  'Authorization': 'OAuth oauth_consumer_key="nAXDj5HBmywFoJmqEsFH7WOvJ",oauth_token="1125761673581580288-WRETzx2EVDe36lsxmrFbvZa8HLWfJT",oauth_signature_method="HMAC-SHA1",oauth_timestamp="1654252993",oauth_nonce="7eGxixCop9l",oauth_version="1.0",oauth_signature="rmg9GywivEDvc%2B3r%2BUZgXABVGOc%3D"',
  
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)


consumer_key = "nAXDj5HBmywFoJmqEsFH7WOvJ"

consumer_secret = "n06aEvR4fVVz5xb9VwlWyEmHxhEKfGU1UDMfhgKbiAEuCN6oJj"

access_key = "1125761673581580288-WRETzx2EVDe36lsxmrFbvZa8HLWfJT"

access_secret = "yjYPqpOaiAwaRt1hvIOfJNs2nJxY3bRiSIbidMkyNV2Mc"

# api = tweepy.Client(consumer_key= consumer_key,consumer_secret= consumer_secret,access_token= access_key,access_token_secret= access_secret)

# data = api.get_tweet(id=1015249783408164864)
# print(data)

# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)

# api = tweepy.API(auth)

# public_tweets = api.get_status(id=1015249783408164864)
