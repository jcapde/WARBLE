# WARBLE: A Probabilistic Model for Retrospective Event Detection in Twitter.

This repository contains the WARBLE code, which implements the probabilistic model and learning scheme presented in:

```
@article{Capdevila-DAMI-2017,
  author = {Joan Capdevila and Jes{\'u}s Cerquides and Jordi Torres},
  title = {Mining Urban Events in the Tweet Stream through a Probabilistic Mixture Model},
  booktitle = {Data Mining and Knowledge Discovery (DAMI)},
  year = 2017,
  pages = {x--x}
}
```

* *download_tweets.py* Given a list of tweet IDs, this script call the Twitter API and downloads the whole metadata for each tweet and
stores in *data/input/ folder*.

* *preprocess_tweets.py* takes raw input datasets from data/input/ folder and generates a series of processed files in data/input/ folder
that will be used by *train_all.py* file.

* *train_all.py* applies variational inference to learn the model parameters from the processed tweets. This script executes learning
for *McInerney&Blei model*, *WARBLE without simulatenous topic learning*, *WARBLE without background*, *WARBLE model* and *Tweet-SCAN*. Files
containing the learnt parameters are stored in data/output.

* *evaluate_all.py* evaluates all abovementioned models in terms of Purity, Inverse Purity and F-measure.

* *plot_results.py* plots comparative results with the above metrics and visually show the event detection results.

* functions/ folder contains functions used by the models and the evaluation.


Note that input data is empty given that sharing of datasets is prohibited under Twitter’s API Terms of Service [https://dev.twitter.com/overview/terms/agreement-and-policy]
