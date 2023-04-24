# music-tweet-ner
This project will seek to measure the effectiveness of various methods of supervised machine learning and deep learning when undertaking Named Entity Recognition (NER) on a dataset consisting of microblog posts (tweets) which contain named entities in the music domain. 

Windows or Mac compatability.

# Contents
### - File list
### - Dependencies
#### - Hydrator
#### - Updated sklearn-CRFsuite files
### - Installing & running
### - Limitations
### - Index of files created
### - Future work

# File list

```
cleaned_rehydrated_tweets_text.csv
datacleaning_preprocessing_macos.py
datacleaning_preprocessing_windows.py
extract_clean_rehydrated_dataset_macos.py
extract_clean_rehydrated_dataset_windows.py
initial_processing_macos.py
initial_processing_windows.py
models_artist.py
models_track.py
rehydrated_tweets.csv
updated_estimator.py
updated_metrics.py
```

Also required are the below files, both found at the Million Musical Tweets Dataset project page (last accessed 23 Apr 2023, 18:21):

* *mmtd.txt* (found in [mmtd.zip](http://www.cp.jku.at/datasets/MMTD/))
* *tweet.txt* (found in [tweet.zip](http://www.cp.jku.at/datasets/MMTD/))

The below files are also included in case you *just* want to run the models - they are both the outputs of running *initial_processing.py, extract_clean_rehydrated_dataset.py and datacleaning_preprocessing.py*:

```
data_for_models_artist.csv
data_for_models_track.csv
```


# Dependencies

Firstly, install [Python 3.7.4 64-bit]([https://protobuf.dev/](https://www.python.org/downloads/release/python-374/) and [Git](https://git-scm.com/downloads).

Installing all of the below packages within a virtual environment is not essential, but is recommended. On Windows, the packages below will not all correctly and fully install *unless* Python 3.7.4, so ensure that you either amend your PATH to point towards the Python 3.7.4 installation, or set the Python used by your virtual environment as 3.7.4 on setup (e.g. python3.7 -m venv path\to\your\dir).

Install via pip or conda - see below for a list of package install locations, followed by a list of pip install commands (as an example):

* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
* [Numpy](https://numpy.org/install/)
* [spaCy](https://spacy.io/usage)**
* [spaCy model - en_core_web_sm](https://space.io/usage)**
* [srsly](https://pypi.org/project/srsly/)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [Tweet Preprocessor](https://pypi.org/project/tweet-preprocessor/)
* [sklearn CRFsuite](https://pypi.org/project/sklearn-crfsuite/)
* [Tensorflow](https://www.tensorflow.org/install/pip) v2.2
* [Keras](https://pypi.org/project/keras/) v2.3.1
* [plot keras history](https://pypi.org/project/plot-keras-history/)
* [Keras community contributions library](https://www.github.com/keras-team/keras-contrib.git)
* [Protobuf](https://protobuf.dev/) v3.20.0

** In terms of installing spaCy & en_core_web_sm, select your operating system, appropriate platform, package manager, hardware, highlight 'train models', select pipeline for 'efficiency'.

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
pip install protobuf==3.20.0
pip install plot_keras_history
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

Separately, install the spaCy model via python:
```
python -m spacy download en_core_web_sm
```

## Hydrator

[Hydrator](https://github.com/DocNow/hydrator)

Hydrator *was* used to rehydrate tweet information from tweet IDs. Unfortunately, recent changes to the Twitter API (see below) mean that this is no longer possible. Given the proximity of the timeframe to the hand-in date of this project (01 May 2023), finding another method is no longer feasible, apart from manually copying tweet IDs into Twitter web addresses within browsers to get the required text one-by-one.

UPDATE (from Github page, as of 18Apr2023):

*"Twitter's changes to their API which greatly reduce the amount of read-only access means that the Hydrator is no longer a useful application. The application keys, which functioned for the last 7 years, have been rescinded by Twitter."*

To circumvent this, the files 'rehydrated_tweets.csv' is included in this repository, which is the output of a Hydrator session which was undertaken mid-way through March 2023. This file is used in the script 'extract_clean_rehydrated_dataset.py'.

## Updated sklearn-CRFsuite files

As the TeamHG-Memex sklearn-CRFsuite Github repository is not being maintained anymore, there are some issues that have cropped up in recent years regarding outdated functionality. MeMartijn on Github has updated some of the scripts, which have been included within this repository. The link for the MeMartijn repository is [here](https://github.com/MeMartijn/updated-sklearn-crfsuite#egg=sklearn_crfsuite).

The two updated scripts referred to in this current (music-tweet-ner) repository are:

```
updated_metrics.py - https://github.com/MeMartijn/updated-sklearn-crfsuite/blob/master/sklearn_crfsuite/metrics.py
updated_estimator.py - https://github.com/MeMartijn/updated-sklearn-crfsuite/blob/master/sklearn_crfsuite/estimator.py
```

# Installing & running

NOTE: Please ensure that you have at least 2.5GB of space free to install and run the project locally.

1. Download the contents of this repository/clone this repository.
2. Unzip/extract contents into a new folder, if not cloned.
3. Download *mmtd.zip* and *tweet.zip* from [here](http://www.cp.jku.at/datasets/MMTD/) and extract *mmtd.txt* and *tweet.txt* into the downloaded/cloned folder.
4. In Terminal (macOS) or the Command Prompt in Windows, cd/move directory to the extracted/cloned folder/repository.
5. Ensure that the required packages/dependencies are installed (see the 'Dependencies' section in this README).

**The running of Python scripts from this point onwards depends on your system. When the instructions demand you to run a .PY script, make sure to run the one that has your system as a suffix e.g. if on Windows, then initialprocessing_windows.py.**
 
7. Run initialprocessing.py (i.e., python initialprocessing.py), which produces files outlined in the 'Index of files created' section
8. *(OPTIONAL)* Run extract_clean_rehydrated_dataset.py, (i.e., python extract_clean_rehydrated_dataset.py), which prepares rehydrated file. Optional due to issues with Hydrator, and with the (already performed) manual cleaning step, which produced the file cleaned_rehydrated_tweets_text.csv.
9. Run datacleaning_preprocessing.py (i.e., python datacleaning_preprocessing.py), which produces files outlined in the 'Index of files created' section
10. Run models_artist.py (i.e, python models_artist.py), which produces a .TXT file which details the running & evaluation relating to 'ARTIST' entites
11. Run models_track.py (i.e., python models_track.py), which produces a .TXT file which details the running & evaluation relating to 'TRACK' entites

# Limitations

* The scripts are split between Windows and Mac due to a frustrating issue regarding newline and additions of '/r' symbols at the end of the final column when using the command .to_csv in pandas on Windows. The code is slightly different between the scripts to reflect the differences required.
* As mentioned in the 'Hydrator' section, the process for rehydrated a mass of tweet IDs quickly has ceased working, due to recent changes in the Twitter API. Given that the eventual samples used is only in the region of 25,000-30,000, it is moderately feasible that they can be hydrated manually, albeit in a time-consuming fashion.
* Manual cleaning was also required for the 'cleaned_rehydrated_tweets_text.csv' dataset, which cannot be automated or replicated with a script. This process included going through the entire rehydrated dataset (620,000+ records, in November 2022) and ensuring that multiple-line tweets were on one line, as well as ensuring that the IDs lined up. Again, with a smaller amount of records, this process is either vastly reduced or much easier in terms of time taken; it took 30+ hours of work to clean that original dataset.
* In the latter stages of the project, it became obvious that it wouldn't be possible to train the models on the full data available up to that point (620,000+ records), due to computational limitations and a lack of funding for using more computation or resources. The decision was taken to then use the limit of what this device (see below) could handle - this was around 27,500 records.
* The two entities 'ARTIST' and 'TRACK' are split at a late stage in the datacleaning_preprocessing.py script. This was primarily due to the script containing *both* of the entities ('ARTIST', then 'TRACK') crashing when attempting to run all of the models in one run. This makes the process more cumbersome than it is, due to the running of two separate scripts when it could realistically be one.

## Device originally used for project
* MacBook Pro (Retina, 13-inch, early 2015), with a 2.7 GHz Dual-Core Intel Core i5 processor, and 8GB 1867 MHz DDR3 memory, running macOS Big Sur

# Index of files created

All of the below files are created in the running of each of these scripts. These files are created mainly for the purpose of being able to illustrate the transformation of the data between each step, as well as allowing others to use aspects for the process for other means, if required.

* initialprocessing.py (both macos and windows)
```
tweet.csv
tweetid_artist_track.csv
just_tweet_ids.csv
```
* extract_clean_rehydrated_dataset.py (both macos and windows)
```
rehydrated_tweets_text.csv
```
* datacleaning_preprocessing.py (both macos and windows)
```
joined_text_artisttrack.csv
master_noindex.csv
master_nodupes_pre.csv
master_cleaned.csv
master_nodupes_post.csv
master_allpunc_removed.csv
spacy_format_artist.json
spacy_format_track.json
data_for_models_artist.csv
data_for_models_track.csv
```
* models_artist.py
```
model_artist_output.txt
```
* models_track.py
```
model_track_output.txt
```

# Future work

The plan is to use more computational power to run a greater number of the converted/rehydrated MMTD records, as well as engaging with other methods of learning e.g., semi-supervised and unsupervised. Also, writing code to run both entities on the same dataset at one time, and see how this affects the performance of the models. Looking at other entities (types of artist, genre etc.) may also yield fertile future research.
