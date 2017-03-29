#!/usr/bin/env python
# -*- coding: utf-8 -*-
from twython import Twython, TwythonError
import json
import time


if __name__ == "__main__":
    with open('./conf/twitter-keys.json', 'r') as app_file:
        app = json.load(app_file)

    with open('./data/input/Twitter-DS/MERCE/2015/tweets.txt') as in_file:
        tweet_ids = in_file.readlines()

    twitter = Twython(app["CONSUMER_KEY"], app["CONSUMER_SECRET"], oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()

    twitter = Twython(app["CONSUMER_KEY"], access_token=ACCESS_TOKEN)

    tweets = []
    n = 0
    recovered = 0
    req_x_sec = 1
    print("Twitter limits this end point to " + str(req_x_sec) + " request per second")
    print("Approximate time to complete "+str((len(tweet_ids)/req_x_sec)/3600.)+"hours")
    for tweet_id in tweet_ids:
        start_time = time.time()
        id = tweet_id[:-1]
        print(str(n) + "/" + str(len(tweet_ids)))
        print("Querying Tweet ID " + id)
        try:
            tweet = twitter.show_status(id=id)
            print(tweet)
            tweets.append(tweet)
            recovered += 1
        except TwythonError:
            print(" Oops! Something went wrong with this tweet!")
        n += 1
        elapsed_time = time.time() - start_time
        if (elapsed_time < 1):
            time.sleep(1-elapsed_time)

    print(str(recovered) + "/" + str(n) + " tweets recovered")
    with open('./data/input/MERCE_2015.json', 'w') as out_file:
        json.dump(tweets, out_file)