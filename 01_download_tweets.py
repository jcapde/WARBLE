#!/usr/bin/env python
# -*- coding: utf-8 -*-

from twython import Twython, TwythonError
import json
import time
import argparse
from pandas import DataFrame
from datetime import datetime


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download tweets from Twitter')
    parser.add_argument('-filePath', metavar='filePath', type=str, default='data/input/Twitter-DS/MERCE/2015/tweets.txt')
    args = parser.parse_args()

    filepath = args.filePath
    fileoutput = "/".join(filepath.split("/")[2:]).replace("/","_").replace(".txt",".json")

    with open('./conf/twitter-keys.json', 'r') as app_file:
        app = json.load(app_file)

    with open(filepath) as in_file:
        tweet_ids = in_file.readlines()

    twitter = Twython(app["CONSUMER_KEY"], app["CONSUMER_SECRET"], oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()

    twitter = Twython(app["CONSUMER_KEY"], access_token=ACCESS_TOKEN)

    tweets = []
    n = 0
    recovered = 0
    processing_errors = 0

    req_x_sec = 1
    print("Twitter limits this end point to " + str(req_x_sec) + " request per second.")
    print("Approximate time to complete "+str((len(tweet_ids)/req_x_sec)/3600.)+"hours")
    print("NOTE that some tweets might have been deleted by users and will not be retrieved.")

    tweet = []
    coordx = []
    coordy = []
    user = []
    timestamp = []
    text = []
    hashtags = []

    for tweet_id in tweet_ids:
        start_time = time.time()
        id = tweet_id[:-1]
        print(str(n) + "/" + str(len(tweet_ids)))
        print("Querying Tweet ID " + id)
        try:
            tweet_obj = twitter.show_status(id=id)
            print(tweet_obj)
            try:
                thts = ' '.join([ht["text"] for ht in tweet_obj["entities"]['hashtags'] if(tweet_obj["entities"]['hashtags'])>0])
                tmstmp = datetime.strptime(tweet_obj["created_at"], '%a %b %d %H:%M:%S +0000 %Y')
                tid = tweet_obj["id"]
                tcoordx = tweet_obj['geo']['coordinates'][0]
                tcoordy = tweet_obj['geo']['coordinates'][1]
                tuser = tweet_obj["user"]["id"]
                ttext = tweet_obj["text"]
                tweet.append(tid)
                coordx.append(tcoordx)
                coordy.append(tcoordy)
                user.append(tuser)
                timestamp.append(tmstmp)
                text.append(ttext)
                hashtags.append(thts)
                tweets.append(tweet_obj)
                recovered += 1
            except TypeError as te:
                processing_errors += 1
                print(te)
                print(" Oops! Something went wrong when processing this tweet!")
        except TwythonError as e:
            print(e)
            print(" Oops! This tweet is no longer available in Twitter!")
        n += 1
        elapsed_time = time.time() - start_time
        if (elapsed_time < 1):
            time.sleep(1-elapsed_time)

    df = DataFrame({
     "tweet_id": tweet,
     "user_id": user,
     "time": timestamp,
     "text": text,
     "hashtags": hashtags,
     "coordx": coordx,
     "coordy": coordy
    })

    print(str(recovered) + "/" + str(n) + " tweets recovered.")
    print("There were "+ str(processing_errors) + " processing errors.")
    df.to_pickle('./data/input/'+fileoutput.replace(".json",".pkl"))
    with open('./data/input/'+fileoutput, 'w') as out_file:
        json.dump(tweets, out_file)