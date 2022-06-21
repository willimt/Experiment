from Scweet.scweet import scrape

data = scrape(words=["nijisanji"], since="2020-01-01", until="2021-01-01", from_account=None,
              interval=1,
              headless=False, display_type="Latest", save_images=False, proxy="127.0.0.1:7890", save_dir='outputs',
              resume=False, filter_replies=True, proximity=False)
