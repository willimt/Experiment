from sys import flags
import requests
import tweepy
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# url = "https://api.twitter.com/2/tweets?ids=1050864924367433729"

# payload={}
# headers = {
#   'Authorization': 'OAuth oauth_consumer_key="nAXDj5HBmywFoJmqEsFH7WOvJ",oauth_token="1125761673581580288-m3K4VchnaDpxlJYfpCHa2CJv9Mncd3",oauth_signature_method="HMAC-SHA1",oauth_timestamp="1654250221",oauth_nonce="iccoNKpyWwD",oauth_version="1.0",oauth_signature="NhO972rQ%2Bm7gfKegKSckq9KPTxw%3D"',
#   'Cookie': 'guest_id=v1%3A165424583562303245'
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)

# consumer_key = "nAXDj5HBmywFoJmqEsFH7WOvJ"

# consumer_secret = "n06aEvR4fVVz5xb9VwlWyEmHxhEKfGU1UDMfhgKbiAEuCN6oJj"

# access_key = "1125761673581580288-m3K4VchnaDpxlJYfpCHa2CJv9Mncd3"

# access_secret = "POxGHB86ZAf3DBOn1OgSHO4vOBnLFVNz23sr5AvMAZe1O"

# api = tweepy.Client(consumer_key= consumer_key,consumer_secret= consumer_secret,access_token= access_key,access_token_secret= access_secret)

# data = api.get_tweet(id=1015249783408164864)
# print(data)