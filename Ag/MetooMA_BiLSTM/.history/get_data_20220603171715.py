import tweepy
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterPager
from requests_oauthlib import OAuth1Session
import json
from twython import Twython

consumer_key = "nAXDj5HBmywFoJmqEsFH7WOvJ"

consumer_secret = "n06aEvR4fVVz5xb9VwlWyEmHxhEKfGU1UDMfhgKbiAEuCN6oJj"

access_key = "1125761673581580288-m3K4VchnaDpxlJYfpCHa2CJv9Mncd3"

access_secret = "POxGHB86ZAf3DBOn1OgSHO4vOBnLFVNz23sr5AvMAZe1O"

proxyUrl = "https://127.0.0.1:7890"
twitter = Twython(consumer_key, consumer_secret,
    access_key, access_secret)
tweet = twitter.show_status(id=1015249783408164864)
# api = TwitterAPI(consumer_key=consumer_key,
#                      consumer_secret=consumer_secret,
#                      access_token_key=access_key,
#                      access_token_secret=access_secret,
#                      proxy_url=proxyUrl)
# params = {"ids": "1015249783408164864"}
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)

# api = tweepy.API(auth)
# api.get_status(id=1015249783408164864)
# oauth = OAuth1Session(
#     consumer_key,
#     client_secret=consumer_secret,
#     resource_owner_key=access_key,
#     resource_owner_secret=access_secret,
# )

# response = oauth.get(
#     "https://api.twitter.com/2/tweets", params=params
# )
# if response.status_code != 200:
#     raise Exception(
#         "Request returned an error: {} {}".format(response.status_code, response.text)
#     )

# print("Response code: {}".format(response.status_code))
# json_response = response.json()
# print(json.dumps(json_response, indent=4, sort_keys=True))

def search_tweets(the_consumer_key, the_consumer_secret, the_access_token_key,
                  the_access_token_secret, the_proxy_url):
    """
    搜索含有特定“内容”的推文
    :param the_consumer_key: 已有的consumer_key
    :param the_consumer_secret: 已有的consumer_secret
    :param the_access_token_key: 已有的access_token_key
    :param the_access_token_secret: 已有的access_token_secret
    :param the_proxy_url: 代理及端口号
    :return:
    """
    api = TwitterAPI(consumer_key=the_consumer_key,
                     consumer_secret=the_consumer_secret,
                     access_token_key=the_access_token_key,
                     access_token_secret=the_access_token_secret,
                     proxy_url=the_proxy_url)
    r = TwitterPager(api, 'search/tweets', {'q': 'pizza', 'count': 10})
    for item in r.get_iterator():
        if 'text' in item:
            print item['text']
        elif 'message' in item and item['code'] == 88:
            print 'SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message']
            break
