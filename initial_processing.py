'''
File: project.py
Overall purpose: Pull all tweet IDs from Million Musical Tweets dataset (MMTD), for usage with Hydrator app.
Note: The last step involves usage of an external app, Hydrator (see last section in script).
'''

'''
Section: Convert tweet.txt
Purpose: Read tweet.txt (from MMTD collection) in correct format & convert to CSV.
Exports: tweet.csv
'''

import pandas as pd # used for data processing
# importtweets.py (original script name)

tweet_data = pd.read_csv('tweet.txt', sep='\s+') # imports data
print('tweet.csv will be created from tweet.txt') 

tweet_data.to_csv('tweet.csv') # exports/converts data
print('tweet.csv created')

'''
Section: Pull Artist/Track/TweetID from MMTD
Purpose: Creates a new DataFrame/CSV which contains Artist, Track and TweetID information. For usage in the data preprocessing script.
Exports: tweetid_artist_track_test.csv
'''

# master_df.py (original script name)

pd.set_option('float_format','{:f}'.format) # amends float_format within pandas, for usage in reading dataset correctly

mmtd_data = pd.read_csv('mmtd.txt',sep='\t',lineterminator='\r')
tweet_data = pd.read_csv('tweet.csv',dtype={'tweet_id':'Int64'})

mmtd_df = pd.DataFrame(mmtd_data)
tweet_df = pd.DataFrame(tweet_data)

df_small_1 = (mmtd_df[['artist_name','track_title']])
df_small_2 = (tweet_df[['tweet_id']])

required_mmtd_data = pd.concat([df_small_2,df_small_1],axis=1) # produces a new dataset, with columns from two different datasets
print('Creating new CSV - containing Artist, Track and TweetID data...')
# print(required_mmtd_data) - historic print check

required_mmtd_data.to_csv('tweetid_artist_track.csv')
print('tweetid_artist_track.csv created')

'''
Section: Export just Tweet IDs
Purpose: Extract just the tweet IDs for use with Hydrator.
Exports: just_tweet_ids.csv (strict format used within Hydrator app)
'''

# extract_just_tweet_ids.py (original script name)

dataset = pd.read_csv('tweet.csv')
just_tweets_ids_df = pd.DataFrame(dataset)
just_tweets_ids_df = just_tweets_ids_df['tweet_id']
print('Creating CSV containing a column of TweetIDs for use with Hydrator...')
# print(just_tweets_ids_df) - used to check the status of the data

just_tweets_ids_df.to_csv('just_tweet_ids.csv', index=False, header=False)
print('just_tweet_ids.csv created')

'''
The app 'Hydrator' is used as the next step:

https://github.com/DocNow/hydrator

Uploading the 'just_tweet_ids.csv' produces another CSV, which contains the associated details of tweets relating to the available tweet IDs in 'just_tweets.csv'

An example 'rehydrated_tweets.csv' file is included to demonstrate the next steps beyond this.

NOTE: Tweets from suspended accounts, or tweets which have been removed, are not included.

NOTE (UPDATED 11 APR 2023):
Due to the overhaul of Twitter's API, Hydrator does not seem to be working in terms of pulling Tweet information.
Thankfully, tweet data was pulled off for the project well in advance, but will make it difficult to replicate 
the pulling off of fresh raw tweet data in the future without another method.

See: https://github.com/DocNow/hydrator/issues/142
'''