 WARBLE: A Probabilistic Event Detection model for Twitter.
======


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

Set up the environment
-----

1-  Install Python requirements:

```
pip install -r requirements.txt

```

2- Register an App to the [Twitter Application Management portal](https://apps.twitter.com/) and obtain your own credentials. 


3- Copy the configuration file ```conf/twitter-key.sample.json``` to ```conf/twitter-key.json``` and fill in the configuration variables with your credentials.


Preparing the datasets
-----

4- Query the Twitter API with a set of tweet IDs given in the ``-filePath`` parameter through ``.txt`` file containing the IDs. 

```
python -filePath data/input/tweets.txt 01_download_tweets.py

```
By default this script downloads tweets from [La Mercè dataset 2014](https://github.com/jcapde/Twitter-DS/tree/master/MERCE/2014). Make sure that you have pulled this repository and placed it under ``data/input`` folder.

The script creates two output files ``data/input/tweets.json`` and ``data/input/tweets.pkl`` which contain all tweets with their corresponding metadata in json format and a pandas Dataframe object with the required fields to run these experiments. 

Note that Twitter users might have deleted some of the tweets or they might have changed their privacy settings, causing some tweets to not be anymore public. The script will notify about tweets that cannot be retrieved. 

The script also limits the number of petitions to the Twitter API to 1 query per second in order to satisfy Twitter limitations. 


Preprocessing datatasets 
----

3- Preprocesses raw tweets from ``data/input/tweets.pkl`` and stores the cleaned dataset in ``data/tmp/dataset.pkl`` 


```
python -fileName data/input/tweets.pkl -labelTweets data/input/label_tweets.csv -day 24/09/2014 02_preprocess_tweets.py

```

- select tweets from a specific day ``-day``.

- remove tweets from users tweetting constinuously ('bots').

- transform and normalise spatial coordinates.
 
- transform and normalise timestamps. 

- stopword removal (spanish, catalan and english)

- remove urls, emojii, numbers and punctuation. 

- assign event labels to tweets for evaluation purposes ``-labelTweets``. The ``data/input/label_tweets.csv`` contains a table with tweet IDs and event classes. 

This script also stores other preprocessed data in``data/tmp/`` used in these experiments. 
- `data/tmp/w.pkl` contains a tweet-word matrix.
- `data/tmp/spacetime_stats.pkl` contains spatiotemporal statistics used to create the backgrounds and to unconvert data. 
- `data/tmp/vocabulary.pkl` and `data/tmp/corpus.pkl` contains the corpus and vocabulary to learn LDA topics for those models that do not perform joint event-topic learning.


Background creation
----

4- Creates the spatio-temporal backgrounds from the raw tweets `data/input/tweets.pkl`.

```
python -fileName data/input/tweets.pkl -day 20/09/2014 -ndays 4 -plot false 03_create_backgroounds.py

```

- build the temporal background from tweets within days (`-ndays`) after the specified date (`-day`).

- save spatiotemporal histogram plots (`-plot false`) of the backgrounds in `data/pics`

- save spatiotemporal backgrounds in `data/tmp/background.pkl`

<table style="width:100%" align="center">
<tr>
<td>Temporal Background</td>
<td>Spatial Background</td>
</tr>

<tr>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/sp.png" align="left" height="250" width="250" ></td>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/tmp.png" align="right" height="250" width="250" ></td>
</tr>
</table>

Learning topics
----

5- *04_learn_topics.py* learns topic-word and topic-document distributions to be used for Tweet-SCAN and models without joint topic-event learning scheme.

6- *05_event_detection.py* perform event detection with the 5 models described in the paper: *McInerney&Blei model*, *WARBLE without simulatenous topic learning*, *WARBLE without background*, *WARBLE model* and *Tweet-SCAN*

7- *06_evaluate_all.py* evaluates all abovementioned models in terms of set matching metrics (Purity, Inverse Purity and F-measure) and BCubed metrics (BCubed Precision, BCubed Recall and BCubed F-measure).

8- *07_plot_results.py* plots the above metrics and shows the results in space-time dimensions.

