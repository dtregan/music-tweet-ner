'''
File: models_artist.py
Overall purpose: Train, run and evaluate the supervised models using the transformed 'ARTIST' entities dataset.

'''

# Import required packages

import sys
f = open("model_artist_output.txt", 'w')
sys.stdout = f

import numpy as np
import pandas as pd
import sklearn
import sklearn_crfsuite
import updated_metrics # updated_metrics.py # sourced from: https://github.com/MeMartijn/updated-sklearn-crfsuite#egg=sklearn_crfsuite
import scipy.stats
import updated_estimator # updated_estimator.py # sourced from: https://github.com/MeMartijn/updated-sklearn-crfsuite#egg=sklearn_crfsuite)

from math import nan
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report 
from sklearn_crfsuite import scorers
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

##########################
# Data upload & analysis #
##########################

df = pd.read_csv("data_for_models_artist.csv", encoding = "ISO-8859-1")

# print(df) - historic print check

print(df.isnull().sum()) # prints amounts of null values
df = df.fillna(' ') # replaces null values (in tweet text - words) with spaces

print(df.groupby('ner').size().reset_index(name='counts')) # prints counts of entity BILOU tags
print(df.groupby('tag').size().reset_index(name='counts')) # prints counts of POS tags

print(df['tweet_no'].nunique(),' unique tweets') # prints no. of unique tweets
print(df.word.nunique(),' unique words') # prints no. of unique words
print(df.tag.nunique(),' unique POS tags') # prints no. of unique POS tags

word_counts = df.groupby("tweet_no")['word'].agg(['count'])
max_sentence = word_counts.max()[0]
print('Longest sentence is ',max_sentence,' words long.') # prints length of longest sentence

avg_length = word_counts.mean()[0] 
print('Average length of a sentence in the dataset is: ',avg_length) # prints the average length of the tweet text sentences

# DictVectorizer

'''
Transforms the values of the 'ner' (entities) column to values that can be mapped as model training data (vectors/matrices)
'''

X = df.drop('ner', axis=1)
v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))
y = df.ner.values

classes = np.unique(y)
classes = classes.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1) # produces a train and test split on the data

print(X_train.shape) # prints the shape of the X_train aspect of the training data
print(y_train.shape) # prints the shape of the y_train aspect of the training data

############################################
# 1. NB classifier for multinomial models ##
############################################

print()
print('##########')
print('Naive Bayes - ARTIST')
print('##########')
print()

from sklearn.naive_bayes import MultinomialNB # (Multinomial) Naive Bayes classifier

# Mod

nb = MultinomialNB(alpha=0.01) # model
nb.partial_fit(X_train, y_train, classes) # partially/gradually fits the model

# Classification report

print(classification_report(y_pred=nb.predict(X_test), y_true=y_test, labels = classes)) # prints classification report

#conf_matrix = confusion_matrix(y_true=y_test,y_pred=nb.predict(X_test)) - historic
#print(conf_matrix) - historic

# Confusion matrix

#print(multilabel_confusion_matrix(y_test.flatten(), y_pred.flatten())) - historic
print(multilabel_confusion_matrix(y_test, y_pred=nb.predict(X_test), labels=classes)) # prints confusion matrix
print()

########################
# 2. Perceptron model ##
########################

print()
print('##########')
print('Perceptron - ARTIST')
print('##########')
print()

from sklearn.linear_model import Perceptron # Perceptron

per = Perceptron(verbose=10, n_jobs=-1, max_iter=5) # model
per.partial_fit(X_train, y_train, classes) # partially/gradually fits the model

# Classification report

print(classification_report(y_pred=per.predict(X_test),y_true=y_test, labels=classes))

#conf_matrix = confusion_matrix(y_true=y_test,y_pred=per.predict(X_test)) - historic
#print(conf_matrix)

# Confusion matrix

print(multilabel_confusion_matrix(y_test, y_pred=per.predict(X_test), labels=classes))
print()

#####################
# 3. SGDClassifier ##
#####################

print()
print('##########')
print('SQD Classifier - ARTIST')
print('##########')
print()

from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent classifer

sgd = SGDClassifier() # model
sgd.partial_fit(X_train, y_train, classes) # partially/gradually fits the model

# Classification report

print(classification_report(y_pred=sgd.predict(X_test),y_true=y_test, labels=classes))

#conf_matrix = confusion_matrix(y_true=y_test,y_pred=sgd.predict(X_test)) - historic
#print(conf_matrix) - historic

# Confusion matrix

print(multilabel_confusion_matrix(y_test, y_pred=sgd.predict(X_test), labels=classes))
print()

#####################################
# 4. Passive Aggressive Classifier ##
#####################################

print()
print('##########')
print('PA Classifier - ARTIST')
print('##########')
print()

from sklearn.linear_model import PassiveAggressiveClassifier # Passive Aggressive Classifier

pa = PassiveAggressiveClassifier(validation_fraction=0.2) # model
pa.partial_fit(X_train, y_train, classes) # partially/gradually fits the model

# Classification report

print(classification_report(y_pred=pa.predict(X_test), y_true=y_test, labels=classes))

#conf_matrix = confusion_matrix(y_true=y_test,y_pred=pa.predict(X_test)) - historic
#print(conf_matrix) - historic

# Confusion matrix

print(multilabel_confusion_matrix(y_test, y_pred=pa.predict(X_test), labels=classes))
print()

#####################
# 5. Random Forest ##
#####################

print()
print('##########')
print('Random Forest - ARTIST')
print('##########')
print()

from sklearn.ensemble import RandomForestClassifier

def feature_map(word): # maps features
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(), word.isalpha()])
    
words = [feature_map(w) for w in df['word'].values.tolist()]
tags = df['ner'].values.tolist() # converts entity values to tags, usable by the model

print(words[:5])

pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words,y=tags,cv=5)

# Classification report

report = classification_report(y_pred=pred, y_true=tags)
print(report)
print()

#conf_matrix = confusion_matrix(y_true=tags,y_pred=pred) - historic
#print(conf_matrix) - historic

# Confusion matrix

print(multilabel_confusion_matrix(tags, y_pred=pred, labels=classes))

############
# 6. CRFs ##
############

print()
print('##########')
print('CRF - ARTIST')
print('##########')
print()

# Retrieve sentences with their POS and tags

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['word'].values.tolist(), 
                                                           s['tag'].values.tolist(), 
                                                           s['ner'].values.tolist())]
        self.grouped = self.data.groupby('tweet_no').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
getter = SentenceGetter(df)
sentences = getter.sentences

# Feature Extraction

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]
    
# Split test/training

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Train CRF models & cross-validation

crf = updated_estimator.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1':scipy.stats.expon(scale=0.5),
    'c2':scipy.stats.expon(scale=0.05),
}


f1_scorer = make_scorer(updated_metrics.flat_f1_score,
                        average='weighted',labels=classes)
                        
rs = RandomizedSearchCV(crf,params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50, 
                        scoring=f1_scorer)

#crf.fit(X_train, y_train) - historic

try:
    rs.fit(X_train,y_train)
except AttributeError:
    pass
    
y_pred = rs.predict(X_test)

# Print best params/cross-validation/GridSearch

print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# Evaluation - classification report & confusion matrix

crf = rs.best_estimator_
y_pred = crf.predict(X_test)

print(updated_metrics.flat_classification_report(y_test, y_pred, labels = classes))
print()

# Confusion matrix

from itertools import chain

f_y_test = list(chain.from_iterable(y_test))
f_y_pred = list(chain.from_iterable(y_pred))

#conf_matrix = confusion_matrix(f_y_test,f_y_pred,labels=classes) - historic
#print(conf_matrix) - historic

print(multilabel_confusion_matrix(f_y_test, f_y_pred, labels=classes))
print()

##########################
#     7. BiLSTM-CRF     ##
##########################

print()
print('##########')
print('BiLSTM-CRF - ARTIST')
print('##########')
print()

from keras.callbacks import ModelCheckpoint
from keras_contrib.layers import CRF

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["ner"].values.tolist())]
        self.grouped = self.dataset.groupby("tweet_no").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
            
getter = SentenceGetter(df)

sentences = getter.sentences

maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)

words = list(set(df["word"].values))
words.append("ENDPAD")

n_words = len(words)

entities = []
for entity in set(df["ner"].values):
    if entity is nan or isinstance(entity, float):
        entities.append('unk')
    else:
        entities.append(entity)
print(entities)

n_entities = len(entities)

from future.utils import iteritems
word2idx = {w: i for i, w in enumerate(words)}
entity2idx = {t: i for i, t in enumerate(entities)}
idx2entity = {v: k for k, v in iteritems(entity2idx)}

from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=99, sequences=X, padding="post",value=n_words - 1)

y_idx = [[entity2idx[w[1]] for w in s] for s in sentences]

y = pad_sequences(maxlen=99, sequences=y_idx, padding="post", value=entity2idx["O"])

from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_entities) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X = [[word2idx[w[0]] for w in s] for s in sentences]

# Model training

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import keras as k

input = Input(shape=(99,))
word_embedding_size = 300
model = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=99)(input)
model = Bidirectional(LSTM(units=word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.5, 
                           recurrent_dropout=0.5, 
                           kernel_initializer=k.initializers.he_normal()))(model)
model = LSTM(units=word_embedding_size * 2, 
             return_sequences=True, 
             dropout=0.5, 
             recurrent_dropout=0.5, 
             kernel_initializer=k.initializers.he_normal())(model)
model = TimeDistributed(Dense(n_entities, activation="relu"))(model)  # previously softmax output layer

crf = CRF(n_entities)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
#model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

model.summary()

filepath="ner-bi-lstm-td-model-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=5, validation_split=0.2, verbose=1)

history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=5, validation_split=0.2, verbose=1, callbacks=callbacks_list)

# Accumulate metrics by entity

TP = {}
TN = {}
FP = {}
FN = {}
for entity in entity2idx.keys():
    TP[entity] = 0
    TN[entity] = 0    
    FP[entity] = 0    
    FN[entity] = 0    

def accumulate_score_by_entity(gt, pred):
    """
    For each entity keep stats
    """
    if gt == pred:
        TP[gt] += 1
    elif gt != 'O' and pred == 'O':
        FN[gt] +=1
    elif gt == 'O' and pred != 'O':
        FP[gt] += 1
    else:
        TN[gt] += 1
        
p = model.predict(np.array(X_test))  

print(classification_report(np.argmax(y_test, 2).ravel(), np.argmax(p, axis=2).ravel(),labels=list(idx2entity.keys()), target_names=list(idx2entity.values())))

# Accumulate scores by entity

for i, sentence in enumerate(X_test):
    y_hat = np.argmax(p[i], axis=-1)
    gt = np.argmax(y_test[i], axis=-1)
    for idx, (w,pred) in enumerate(zip(sentence,y_hat)):
        accumulate_score_by_entity(idx2entity[gt[idx]],entities[pred])
        

for entity in entity2idx.keys():
    print(f'Entity:{entity}')    
    print('\t TN:{:10}\tFP:{:10}'.format(TN[entity],FP[entity]))
    print('\t FN:{:10}\tTP:{:10}'.format(FN[entity],TP[entity])) 

#########

f.close()