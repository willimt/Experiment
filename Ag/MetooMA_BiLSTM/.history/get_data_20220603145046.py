from Scweet.scweet import scrape

data = scrape(id from_account=None,
              interval=1,
              headless=False, display_type="Latest", save_images=False, proxy="127.0.0.1:7890", save_dir='outputs',
              resume=False, filter_replies=True, proximity=False)
