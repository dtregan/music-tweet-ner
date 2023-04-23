'''
File: datacleaning_preprocessing.py
Overall purpose: Join, clean and process data, with the eventual usage for model training and evaluation on both 'ARTIST' and 'TRACK' entities
'''

'''
Section: Join rehydrated tweet text to corresponding artists and tracks
Purpose: Create a dataset containing the required fields: tweetID (joined on), raw tweet text, artist, tracks
Output: joined_text_artisttrack.csv
'''
# join_artist_track_text.csv (original script name)

import pandas as pd # used for data processing

print('Creating new CSV - joining tweet text with artist & track, using tweet ID as joining ID...')
join_data_1 = pd.read_csv("cleaned_rehydrated_tweets_text.csv",dtype={'id':'Int64'},lineterminator='\n')
join_data_2 = pd.read_csv("tweetid_artist_track.csv",dtype={'tweet_id':'Int64'})

join_df1 = pd.DataFrame(join_data_1)
join_df1.rename(columns={'id': 'tweet_id'}, inplace=True)

join_df2 = pd.DataFrame(join_data_2)

join_df3 = join_df1.merge(join_df2,on='tweet_id',how='left') # merges/joins data - tweet ids, text, artist and track

# print(df3.head()) # historic print check

join_df3.to_csv('joined_text_artisttrack.csv')
print('joined_text_artisttrack.csv created')


'''
Section: Removes extra index columns
Purpose: Code removes index columns created erroneously through previous processing
Output: text_artisttrack_noindex.csv
'''
# clean_up_master.py (original script name)

indexes_data = pd.read_csv("joined_text_artisttrack.csv",dtype={'tweet_id':'Int64'},lineterminator='\n')

indexes_df = pd.DataFrame(indexes_data)

# df = df.head(100000) # provides a decent selection to pick 27500 random records from later, and to use less system resources
# for processing

# print(df.head()) - historic print check
print('Creating new CSV - cleaning data of extra index columns...')
indexes_df_new = indexes_df.drop(columns = ['Unnamed: 0','Unnamed: 0_x','Unnamed: 0_y']) # removes extra index columns

# print(df_new.head()) - historic print check

indexes_df_new.to_csv('master_noindex.csv', index=False)
print('master_noindex.csv created')

'''
Section: Drop duplicates from dataset
Purpose: Drop any duplicates from within dataset
Output: master_nodupes_pre.csv
'''
# drop_dupes.py (orignal script name)

df = pd.read_csv("master_noindex.csv",dtype={'tweet_id':'Int64'},lineterminator='\n')
print('Creating new CSV - cleaning data of duplicate records, pre-processing...')
df.drop_duplicates(subset=None, inplace=True) # remove duplicate rows

# Write the results to a different file
df.to_csv("master_nodupes_pre.csv", index=False)
print('master_nodupes_pre.csv created')

'''
Section: Clean parts of tweet text using Twitter Preprocessor
Purpose: Use Twitter Preprocessor to clean aspects of tweet text
Output: master_cleaned.csv
'''
# cleandata.py

import preprocessor as p # Tweet Preprocessor

preprocessor_df = pd.read_csv("master_nodupes_pre.csv")

# df = df.head(10000) - historic print check

cleaned_preprocessor_df = []

print('Creating new CSV - cleaning data of duplicate records, pre-Tweet Preprocessor...')

p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY) # set Tweet Preprocessor to remove: URLs, mentions (@name), hashtag (#trend), special words (e.g., RT, which is Twitter vernacular, emojis & smilies)

for i in preprocessor_df.index.tolist():
    value = preprocessor_df.iloc[i,1]
    try:
        cleaned_text = p.clean(value)
        preprocessor_df.iloc[i,1] = cleaned_text
        cleaned_preprocessor_df.append(preprocessor_df.iloc[i]) # adds cleaned text to empty list
        print(cleaned_text) # prints ALL text entries to show progress
    except:
        preprocessor_df.drop(i) # removes text entry if doesn't fit within guidelines
        
cleaned_df_toexport = pd.DataFrame(cleaned_preprocessor_df)

# print(df_cleaned) - historic print check

cleaned_df_toexport.to_csv("master_cleaned.csv",index=False)
print('master_cleaned.csv created')

'''
Section: Clean file processed by Tweet Preprocessor
Purpose: Some artifacts are made by Tweet Proprocessor - cleaning required to correct the format
Output: master_nodupes_post.csv
'''

nodupes_post_df = pd.read_csv("master_cleaned.csv")

# REMOVE DUPLICATES
print('Creating new CSV - cleaning data of duplicate records, post-Tweet Preprocessor...')
nodupes_post_df = nodupes_post_df.drop_duplicates()

# REMOVE ERROR RECORDS WHERE TWEET ID IS NOT 18-DIGIT STRING/ID
nodupes_post_df = nodupes_post_df.drop(nodupes_post_df[nodupes_post_df['tweet_id'].str.len()!=18].index) # removes records where tweet ID isn't standard length of 18 numbers

# print(df) - historic print check

nodupes_post_df.to_csv("master_nodupes_post.csv",index=False)
print('master_nodupes_post.csv created')

'''
Section: Further data cleaning (punctuation)
Purpose: Further clean up text of punctuation characters, primarily using the .replace() method.
Output: master_allpunc_removed.csv
'''

# nopunc.py (original script name)

#df = pd.read_csv("master_cleaned_proper.csv")
df = pd.read_csv("master_nodupes_post.csv")

# df = df.head(10000)

df_punc = pd.DataFrame(df)

# FIND NAMES OF COLUMNS: - historic print check
#for col in df_punc.columns:
    #print(col)

print('Cleaning tweet text of any further unneeded punctuation...')

new_text_col = []

for row in df_punc['text']: # removes (or converts) all below punctuation and adds cleaned record to a new list
    row = str(row)
    row = row.replace('&amp', '&').replace('&lt;3','<').replace('&lt; 3','<').replace('&lt;','<').replace('&lt','<').replace('&gt;','>').replace('Black Squirrel (Broken, Beat & Scarred).mp3','Metallica')
    row = row.replace(r'[^\w\s\&\(\)\-]+', '')
    row = row.replace('>','').replace('<','').replace('|','').replace('!','').replace('?','').replace('.','').replace(',','').replace('~','').replace(';','').replace('"','').replace(":'3",'').replace('^','').replace(':','').replace('*','').replace('--','').replace('---','').replace('[','').replace(']','').replace("\\",'').replace('/','').replace('(','').replace(')','').replace('-',' ')
    row = row.replace('  ',' ').replace('   ',' ').replace('null',' ')
    new_text_col.append(row)

# removed punctuation - | . ~ > < ? !

# re: ('Black Squirrel (Broken, Beat & Scarred).mp3','Metallica')
# Note: for some reason, this artist was coded wrong in the MMTD dataset

new_text_col_srs = pd.Series(new_text_col)

df_punc['text'] = new_text_col_srs

# print(df) - historic print check

print('Creating new CSV - post-punctuation removal...')

df_punc.to_csv("master_allpunc_removed.csv",index=False)
print('master_allpunc_removed.csv created')


'''
Section: Convert to spaCy ('ARTIST')
Purpose: Using spaCy module, convert the tweet text dataset to JSON file, which includes extra features (via spaCy's 'en_core_web_sm' algorithm) - POS tags, morphology etc.)
Output: spacy_format_artist.json
'''

import re # imports regex for usage in zipping text and entities
import spacy # imports spaCy, for processing
import srsly # imports srsly for conversion to JSON

from spacy.training import docs_to_json, offsets_to_biluo_tags, offsets_to_biluo_tags, biluo_tags_to_spans # used to map the tokens to entity tags

print('Creating new JSON (using spaCy algorithm) containing ARTIST-entity and associated tags...')
print()
print('NOTE: Misaligned or incorrect entity pairings will be omitted (warning message(s) will fire)')
print()

artist_cleaned_data = pd.read_csv("master_allpunc_removed.csv")

artist_cleaned_data_master = pd.DataFrame(artist_cleaned_data)

artist_cleaned_data_master = artist_cleaned_data_master.sample(n=27500, random_state = 10)

tweet_text = artist_cleaned_data_master['text'].tolist()
all_artists = artist_cleaned_data_master['artist_name'].tolist()

# Identifies location of specific entities and creates a new string which contains the text string and entities locations (as indexes)

artist_results = []
for text, artist in zip(tweet_text, all_artists):
    try:
        text_l = text.lower()
        artist_l = artist.lower()
        for match in re.finditer(artist_l,text_l):
            s_artist = match.start()
            e_artist = match.end()
            # print(f'String match at [{s_artist}:{e_artist}]')

        annotate = (text, {"entities":[(s_artist,e_artist,"ARTIST")]})
        # print(annotate)

        artist_results.append(annotate)

    except:
        continue

nlp = spacy.load('en_core_web_sm')
docs = []

### CONVERT INTO BILOU FORMAT VIA SPACY ###

for text, annot in artist_results:
    try:
        doc = nlp(text)
        tags = offsets_to_biluo_tags(doc, annot['entities'])
        entities = biluo_tags_to_spans(doc, tags)
        doc.ents = entities
        docs.append(doc)
    except:
        continue

srsly.write_json("spacy_format_artist.json", [docs_to_json(docs)])
print('spacy_format_artist.json created.')

'''
Section: Convert to spaCy ('TRACK')
Purpose: Using spaCy module, convert the tweet text dataset to JSON file, which includes extra features (via spaCy's 'en_core_web_sm' algorithm) - POS tags, morphology etc.)
Output: spacy_format_track.json
'''

print('Creating new JSON (using spaCy algorithm) containing TRACK-entity and associated tags...')
print()
print('NOTE: Misaligned or incorrect entity pairings will be omitted (warning message(s) will fire)')
print()

track_cleaned_data = pd.read_csv("master_allpunc_removed.csv")

track_cleaned_data_master = pd.DataFrame(track_cleaned_data)

track_cleaned_data_master = track_cleaned_data_master.sample(n=27500, random_state = 10)

tweet_text = track_cleaned_data_master['text'].tolist()
all_tracks = track_cleaned_data_master['track_title'].tolist()

# Identifies location of specific entities and creates a new string which contains the text string and entities locations (as indexes)

track_results = []
for text, track in zip(tweet_text, all_tracks):
    try:
        text_l = text.lower()
        track_l = track.lower()
        for match in re.finditer(track_l,text_l):
            s_track = match.start()
            e_track = match.end()
            # print(f'String match at [{s_track}:{e_track}]')

        annotate = (text, {"entities":[(s_track,e_track,"TRACK")]})
        # print(annotate)

        track_results.append(annotate)

    except:
        continue

nlp = spacy.load('en_core_web_sm')
docs = []

### CONVERT INTO BILOU FORMAT VIA SPACY ###

for text, annot in track_results:
    try:
        doc = nlp(text)
        tags = offsets_to_biluo_tags(doc, annot['entities'])
        entities = biluo_tags_to_spans(doc, tags)
        doc.ents = entities
        docs.append(doc)
    except:
        continue

srsly.write_json("spacy_format_track.json", [docs_to_json(docs)])
print('spacy_format_track.json created.')

'''
Section: Convert spaCy format JSON to dataset for model training ('ARTIST')
Purpose: Extracts required features from spaCy-converted dataset - sentence/tweet reference, word (orth), POS tag (two forms), entity type (B,I,L,O,U) - into required format for model training/eval
Output: data_for_ml_artist.csv
'''

# convert_spacy_to_train.py (original script name)

artist_df = pd.read_json('spacy_format_artist.json')

print('Creating new CSV file containing ARTIST-entity and specific tag information, to be used for model training/evaluation...')

artist_df_norm = pd.json_normalize(artist_df["paragraphs"][0])
all_artist_records = artist_df_norm['sentences']

tweet_no = []
word = []
tag = []
#pos = [] - not required
ner = []

for index, record in enumerate(all_artist_records):  # extracts the specified information and adds to a series of lists, which are used to build a DataFrame
    for part_of_sentence in record:
        length_of_record = len(part_of_sentence['tokens'])
        for num in range(0,length_of_record):
            tweet_no.append(index)
            word.append(part_of_sentence['tokens'][num]['orth'])
            tag.append(part_of_sentence['tokens'][num]['tag'])
            #pos.append(part_of_sentence['tokens'][num]['pos']) - not required
            ner.append(part_of_sentence['tokens'][num]['ner'])

artist_new_df = pd.DataFrame(
    {'tweet_no': tweet_no,
    'word': word,
     'tag': tag,
     #'pos': pos, - not required
     'ner': ner
 })

#df['ner'] = artist_new_df['ner'].replace(['B-TRACK','I-TRACK','L-TRACK','U-TRACK'],'TRACK')

# print(artist_new_df) - historic print check

artist_new_df.to_csv("data_for_models_artist.csv",index=False)
print('data_for_ml_artist.csv created.')

'''
Section: Convert spaCy format JSON to dataset for model training ('TRACK')
Purpose: Extracts required features from spaCy-converted dataset - sentence/tweet reference, word (orth), POS tag (two forms), entity type (B,I,L,O,U) - into required format for model training/eval
Output: data_for_models_track.csv
'''

track_df = pd.read_json('spacy_format_track.json')

print('Creating new CSV file containing TRACK-entity and specific tag information, to be used for model training/evaluation...')

track_df_norm = pd.json_normalize(track_df["paragraphs"][0])
all_track_records = track_df_norm['sentences']

tweet_no = []
word = []
tag = []
#pos = [] - not required
ner = []

for index, record in enumerate(all_track_records): # extracts the specified information and adds to a series of lists, which are used to build a DataFrame
    for part_of_sentence in record:
        length_of_record = len(part_of_sentence['tokens'])
        for num in range(0,length_of_record):
            tweet_no.append(index)
            word.append(part_of_sentence['tokens'][num]['orth'])
            tag.append(part_of_sentence['tokens'][num]['tag'])
            #pos.append(part_of_sentence['tokens'][num]['pos']) - not required
            ner.append(part_of_sentence['tokens'][num]['ner'])

track_new_df = pd.DataFrame(
    {'tweet_no': tweet_no,
    'word': word,
     'tag': tag,
     #'pos': pos, - not required
     'ner': ner
 })

#df['ner'] = df['ner'].replace(['B-TRACK','I-TRACK','L-TRACK','U-TRACK'],'TRACK')

# print(track_df) - historic print check

track_new_df.to_csv("data_for_models_track.csv",index=False)
print('data_for_models_track.csv created.')