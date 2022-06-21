from Scweet.scweet import scrape

data = scrape(
              headless=False, proxy="127.0.0.1:7890", save_dir='outputs',
              resume=False, filter_replies=True, proximity=False)
