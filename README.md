 WARBLE: A Probabilistic Event Detection model for Twitter.
======


This repository contains the WARBLE code, which implements the probabilistic model and learning scheme presented [here](https://drive.google.com/file/d/0B8Dg3PBX90KNcUhFbUhxYWthamJUM2h1aXhfUEZ4OWU5ZDd3/view).



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

5- Preprocesses raw tweets from ``data/input/tweets.pkl`` and stores the cleaned dataset in ``data/tmp/dataset.pkl`` 


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

6- Creates the spatio-temporal backgrounds from the raw tweets `data/input/tweets.pkl`.

```
python -fileName data/input/tweets.pkl -day 20/09/2014 -ndays 4 -plot false 03_create_backgroounds.py

```

- build the temporal background from tweets within days (`-ndays`) after the specified date (`-day`).

- save spatiotemporal histogram plots (`-plot false`) of the backgrounds in `data/pics`

- save spatiotemporal backgrounds in `data/tmp/background.pkl`

<table style="width:100%" align="center">
<tr>
<td>Spatial Background</td>
<td>Temporal Background</td>
</tr>

<tr>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/sp.png" align="left" height="250" width="250" ></td>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/tmp.png" align="right" height="250" width="250" ></td>
</tr>
</table>


Learning topics
----

7- learns topic-word and topic-document distributions used by Tweet-SCAN and other models without joint topic-event learning scheme.

```
python -T 30 04_learn_topics.py

```

This script uses `data/tmp/vocabulary.pkl` and `data/tmp/corpus.pkl`  and stores topic-word and topic-document distributions in `data/tmp/Phi.pkl` and `data/tmp/Theta.pkl`,
respectively.


WARBLE event detection
----

8- performs event detection in WARBLE with `-T` topics `-K` events `-maxIter` maximum iterations on day `-day`.

```
python -T 30 -K 8 -day -maxIter 50 -day 24/09/2014 05_WARBLE.py

```

and outputs extrinsic clustering measures (purity, inverse purity and F-measure) as well as Recalls and location summaries 
for the uncovered events.

```
WARBLE model -  Purity:  0.448412698413  Inv. Purity:  0.689655172414  F-measure: 0.543465191776

Human towers Recall 0.388888888889(0.218106995885) tweets: 7 out of 18
Location: 41.3856448288 ± 0.00631487974915 2.18791884745 ± 0.0157190909997
Time: 2014-09-24 15:03:44.208664 ± 0:06:47.367456
 --- 
Concert revival Recall 0.857142857143(0.75) tweets: 24 out of 28
Location: 41.3856448288 ± 0.00631487974915 2.18791884745 ± 0.0157190909997
Time: 2014-09-24 15:03:44.208664 ± 0:06:47.367456
 --- 
Fireworks Recall 0.947368421053(0.9) tweets: 54 out of 57
Location: 41.3729935849 ± 0.00151711856168 2.1492366885 ± 0.00220343635435
Time: 2014-09-24 23:47:11.350112 ± 0:05:36.217038
 --- 
Concert Recall 1.0(1.0) tweets: 21 out of 21
Location: 41.392775551 ± 0.00156595867765 2.20563928718 ± 0.00192998521161
Time: 2014-09-24 04:22:42.107117 ± 0:13:28.215183
 --- 
Museums  0.736842105263(0.583333333333) tweets: 14 out of 19
Location: 41.3833337281 ± 0.00128782807803 2.17193377282 ± 0.00487644844136
Time: 2014-09-24 20:09:42.862351 ± 0:06:22.760313
 --- 
```

The script also plots the the spatio-temporal features of tweets and colours them depending to the event or background (grey color) that they were assigned by WARBLE. 


<table style="width:100%" align="center">
<tr>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/warble.png" align="right" height="300" width="300" ></td>
</tr>
</table>


Evaluation
----

9 - compares a probabilistic baseline model presented [here](http://ailab.ijs.si/~blazf/NewsKDD2014/submissions/newskdd2014_submission_9.pdf), the 
two WARBLEs version w/o simultaneous event-topic learning and w/o background, the state-of-the-art model *Tweet-SCAN* presented [here](http://www.sciencedirect.com/science/article/pii/S0167865516302124) and the complete WARBLE. 

```
python -T 30 -K 8 -day -maxIter 50 06_compare_models.py

```

The script outputs the file `data/output/event_assignments.npy` which will be used for assement. 


10- evaluates the abovementioned models in terms of set matching metrics (Purity, Inverse Purity and F-measure) or BCubed metrics (BCubed Precision, BCubed Recall and BCubed F-measure).


```
python -BCubed False 07_evaluate.py

```

It stores the results in `data/output/purity.txt`, `data/output/inv_purity.txt` and `data/output/f_measure.txt`.


11- plots the above metrics and shows the results in space-time dimensions.


```
python 08_plot_results.py

```

<table style="width:100%" align="center">
<tr>
<td>Set matching metrics</td>
<td>BCubed metrics</td>
</tr>

<tr>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/setmatch.png" align="left" height="250" width="250" ></td>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/bcubed.png" align="right" height="250" width="250" ></td>
</tr>
</table>

Model A: McInerney&Blei model;
Model B: WARBLE w/o simultaneous topic-event learning;
Model C: WARBLE w/o background model;
Model D: Complete WARBLE;
Model E: Teet-SCAN;
(Below) F: Labeled Events 

<table style="width:100%" align="center">
<tr>
<td><img src="https://github.com/jcapde/WARBLE/blob/master/data/pics/spacetime.png" align="right" height="300" width="300" ></td>
</tr>
</table>


When using this repository, please cite: 

```
@article{Capdevila-DAMI-2017,
  author = {Joan Capdevila and Jes{\'u}s Cerquides and Jordi Torres},
  title = {Mining Urban Events in the Tweet Stream through a Probabilistic Mixture Model},
  booktitle = {Data Mining and Knowledge Discovery (DAMI)},
  year = 2017,
  pages = {x--x}
}
@article{Capdevila-ICML-2016,
  author = {Joan Capdevila and Jes{\'u}s Cerquides and Jordi Torres},
  title = {Recognizing warblers: a probabilistic model for event detection in Twitter},
  booktitle = {ICML Anomaly Detection Workshop},
  year = 2016,
}
```