# music-tweet-ner
Project which performs Named Entity Recognition on raw tweet text from Million Musical Tweets dataset, using supervised models

## Dependencies

Install via pip or conda - see below for a list of package install locations, followed by a list of pip install commands (as an example):

[note all packages, modules etc. here]

* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Numpy](https://numpy.org/install/)
* [spaCy](https://spacy.io/usage)*
* [spaCy model - en_core_web_sm](https://space.io/usage)*
* [srsly](https://pypi.org/project/srsly/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [Tweet Preprocessor](https://pypi.org/project/tweet-preprocessor/)
* [sklearn CRFsuite](https://pypi.org/project/sklearn-crfsuite/)
* Tensorflow v2.2
* Keras v2.3.1
* [plot keras history](https://pypi.org/project/plot-keras-history/)

* In terms of installing spaCy & en_core_web_sm, select your operating system, appropriate platofmr, package manager, hardware, highlight 'train models', select pipeline for 'efficiency'.

List of example pip commands relating to the list above:
```
pip install pandas
pip install numpy
pip install -U pip setuptools wheel
pip install -U spacy
pip install srsly
pip install -U scikit-learn
pip install tweet-preprocessor
pip install sklearn-crfsuite
pip install tensorflow==2.2
pip install keras==2.3.1
pip install plot_keras_history
```

Seperately, install the spaCy model via python:

```
python -m spacy download en_core_web_sm
```

## Hydrator

[Hydrator](https://github.com/DocNow/hydrator)

Hydrator *was* used to rehydrate tweet information from tweet IDs. Unfortunately, recent changes to the Twitter API (see below) mean that this is no longer possible. Given the proximity of the timeframe to the hand-in date of this project (01 May 2023), finding another method is no longer feasible, apart from manually copying tweet IDs into Twitter web addresses within browsers to get the required text one-by-one.

UPDATE (as of 18Apr2023):

"Twitter's changes to their API which greatly reduce the amount of read-only access means that the Hydrator is no longer a useful application. The application keys, which functioned for the last 7 years, have been rescinded by Twitter."

## Updated sklearn-CRFsuite files

As the TeamHG-Memex sklearn-CRFsuite Github repository is not being maintained anymore, there are some issues that have cropped up in recent years regarding outdated functionality. MeMartijn on Github has updated some of the scripts, which have been included within this repository. The link for the MeMartijn repository is [here](https://github.com/MeMartijn/updated-sklearn-crfsuite#egg=sklearn_crfsuite).

The two updated scripts referred in this current repository are:

```
updatedmetrics.py
updatedestimator.py
```

## Installing/running

## Guide

[step-by-step guide to running the project]

## Limitations

[Hydrator - no longer working]
[Manual cleaning required for the rehydrated dataset]
[Computational limitations]

## Index of files created

