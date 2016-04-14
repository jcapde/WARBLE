# The Warble model for event detection in Twitter

This repository contains all scripts to reproduce the article titled *Tuning up probabilistic models for event detection in
social networks*. Next, we summarize the main scripts.

* *preprocess_tweets.py* takes raw input datasets from data/input/ folder and generates a series of processed files in data/input/ folder
that will be used by *train_all.py* file.

* *train_all.py* applies variational inference to learn the model parameters from the processed tweets. This script executes learning
for *McInerney&Blei model*, *WARBLE without simulatenous topic learning*, *WARBLE without background*, *WARBLE model* and *Tweet-SCAN*. Files
containing the learnt parameters are stored in data/output.

* *evaluate_all.py* evaluates all abovementioned models in terms of Purity, Inverse Purity and F-measure.

* *plot_results.py* plots comparative results with the above metrics and visually show the event detection results.

* functions/ folder contains functions used by the models and the evaluation.


Note that input data is empty given that sharing of datasets is prohibited under Twitter’s API Terms of Service [https://dev.twitter.com/overview/terms/agreement-and-policy]
