'''
File: extract_clean_rehydrated_dataset.py
Overall purpose: Convert rehydrated_tweets.csv to rehydrated_tweets_text.csv
'''

'''
Section: Extract tweet text and tweet IDs from rehydrated tweets
Purpose: Extract just tweet text and tweet IDs from newly-created CSV via the Hydrator app
Output: rehydrated_tweets_text.csv (columns: tweet text, tweet ID)
'''

import pandas as pd # used for data processing

dataset = pd.read_csv("rehydrated_tweets.csv")
print('Creating new CSV - tweet ID and raw tweet text from rehydrated dataset...')

df = pd.DataFrame(dataset)

df1 = (df[['id', 'text']])

df1.to_csv("rehydrated_tweets_text.csv")

with open("rehydrated_tweets_text.csv", mode='w', newline='\n',encoding="utf-8") as f:
    df1.to_csv(f, sep=",", line_terminator='\n', encoding='utf-8')

print('rehydrated_tweets_text.csv created')

'''
MANUAL CLEANING REQUIRED
Purpose: The rehydrated and processed CSV has a large amount of records (600,000+) where the tweet text is split over multiple lines.
         On reviewing the dataset, it is clear that this process cannot be automated due to the differences in each individual case.
         The issue seems to stem from tweets which included pictures, but this is not comprehensive.
Time taken: 30+ hrs to manually clean.
Output: cleaned_rehydrated_tweets_text.csv (columns: tweet ID, tweet text)
'''
