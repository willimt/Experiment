import requests

url = "https://api.twitter.com/2/tweets?ids=1015249783408164864"

payload={}
headers = {
  'Authorization': 'OAuth oauth_consumer_key="nAXDj5HBmywFoJmqEsFH7WOvJ",oauth_token="1125761673581580288-m3K4VchnaDpxlJYfpCHa2CJv9Mncd3",oauth_signature_method="HMAC-SHA256",oauth_timestamp="1654248661",oauth_nonce="3nCkFSbvhA0",oauth_version="1.0",oauth_signature="5P8oBFKzck9LBtz1rtaWDlmySDwjgekegUZ8xE3D0Ls%3D"',
  'Cookie': 'guest_id=v1%3A165424583562303245'
}

response = requests.request("HEAD", url, headers=headers, data=payload)

print(response.text)
