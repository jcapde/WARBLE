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

1-  Install Python requirements

```
pip install -r requirements.txt

```

2- *01_download_tweets.py* crawls the Twitter API with a list of tweet IDs and stores their content in *data/input/ folder*.

3- *02_preprocess_tweets.py* takes the raw input datasets from data/input/ folder and generates a series of processed files in data/input/ folder
that will be used by *train_all.py* file.

4- *03_create_backgrounds.py* creates the spatio-temporal backgrounds from the original dataset for the specified dates.

5- *04_learn_topics.py* learns topic-word and topic-document distributions to be used for Tweet-SCAN and models without joint topic-event learning scheme.

6- *05_event_detection.py* perform event detection with the 5 models described in the paper: *McInerney&Blei model*, *WARBLE without simulatenous topic learning*, *WARBLE without background*, *WARBLE model* and *Tweet-SCAN*

7- *06_evaluate_all.py* evaluates all abovementioned models in terms of set matching metrics (Purity, Inverse Purity and F-measure) and BCubed metrics (BCubed Precision, BCubed Recall and BCubed F-measure).

8- *07_plot_results.py* plots the above metrics and shows the results in space-time dimensions.

