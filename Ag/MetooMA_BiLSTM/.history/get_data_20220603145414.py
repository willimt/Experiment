import stweet as st


def try_tweet_by_id_scrap():
    id_task = st.TweetsByIdTask('1447348840164564994')
    output_json = st.JsonLineFileRawOutput('output_raw_id.jl')
    output_print = st.PrintRawOutput()
    st.TweetsByIdRunner(tweets_by_id_task=id_task,
                        raw_data_outputs=[output_print, output_json]).run()


if __name__ == '__main__':
    try_search()
    try_user_scrap()
    try_tweet_by_id_scrap()