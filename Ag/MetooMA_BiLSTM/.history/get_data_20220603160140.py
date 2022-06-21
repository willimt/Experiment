import tweepy

consumer_key = " "

consumer_secret = " "

access_key = " "

access_secret = " "

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

api = tweepy.API(auth)
————————————————
版权声明：本文为CSDN博主「小张奔小康」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_42294077/article/details/120996473