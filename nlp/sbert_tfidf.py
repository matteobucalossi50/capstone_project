## tfidf
import pandas as pd
import math
import numpy as np
import re
import collections

import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords.extend(['https', 'http', 'com', 'tbt', 'ico', 'foodie', 'food', 'please', 'love', 'home'])
from nltk import FreqDist
from textblob import TextBlob as tb
import collections
import plotly.graph_objects as go

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

import matplotlib.pyplot as plt
import preprocessor as p

from transformers import BertModel, BertConfig
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def process_words(raw_texts, stop_words=stopwords):
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
    # remove stopwords, short tokens and letter accents
    texts = []
    for t in raw_texts:
        texts.append(p.clean(t))

    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3)
              if word not in stop_words] for doc in texts]
    texts_out = []

    noun = []
    adj = []

    relevants = []
    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent))

        noun.append([token.lemma_ for token in doc if token.pos_ == 'PROPN' or token.pos_ =='NOUN'])
        adj.append([token.lemma_ for token in doc if token.pos_ == 'ADJ'])
        texts_out.append([token.lemma_ for token in doc])
        relevants.append([token.lemma_ for token in doc if token.pos_ == 'ADJ' or token.pos_ == 'PROPN' or token.pos_ =='NOUN'])

    # remove stopwords and short tokens again after lemmatization
    # texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3)
    #               if word not in stop_words] for doc in texts_out]
    return texts_out, noun, adj, relevants


def tokens_col(filtered_df):
    texts = filtered_df.tweet_text.values.tolist()
    texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

    out_toks, nouns, adjs, relev_tokens = process_words(texts, stopwords)

    filtered_df['tokens'] = relev_tokens

    return filtered_df


## search on transformers
def sbert_search(filtered_df, zip, query):

    model = SentenceTransformer('msmarco-distilbert-base-v4')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    zipped_df = filtered_df[filtered_df['zipcode'] == zip]
    filt_texts = zipped_df.tweet_text.values.tolist()

    ## embed search on zipcodes
    filt_embeds = model.encode(filt_texts, convert_to_tensor=True)


    top_k = min(int(150), len(filt_texts))

    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, filt_embeds, top_k=top_k)
    hits = hits[0]

    cross_inp = [[query, filt_texts[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    rows = []
    for hit in hits:
        rows.append(hit['corpus_id'])
        print("\t{:3f}\t{}".format(hit['cross-score'], filt_texts[hit['corpus_id']].replace("\n", "")))


    zipped_df_searched = zipped_df.iloc[rows]

    return zipped_df_searched


## tfidf on zipcodes
def tfidf_keywords(df):
    zipped_tweets_searched = df.tweet_text.values.tolist()
    zipped_tweets_searched = ' '.join(zipped_tweets_searched)
    bloblist = [tb(zipped_tweets_searched)]

    words = []

    for i, blob in enumerate(bloblist):
            print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:100]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
                words.append(word)

    return words


## trigrams
## grams-association
def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """

    lengths = range(min_length, max_length + 1)
    ngrams = collections.Counter()
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in line:
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def grams_plots(df):

    zipped_toks_searched = df.tokens.values.tolist()

    trigrams = count_ngrams(zipped_toks_searched, 3, 3)

    bigrams = count_ngrams(zipped_toks_searched, 2, 2)

    fig = go.Figure(data=[go.Bar(
        x=list(list(trigrams.values())[0].keys()),
        y=list(list(trigrams.values())[0].values()),
        text=list(list(bigrams.values())[0].values()),
        textposition='auto',
    )])
    fig.update_layout(title_text='Trigrams frequency')
    fig.show()

    fig = go.Figure(data=[go.Bar(
        x=list(list(bigrams.values())[0].keys()),
        y=list(list(bigrams.values())[0].values()),
        text=list(list(bigrams.values())[0].values()),
        textposition='outside',
    )])
    fig.update_layout(title_text='Bigrams frequency')
    fig.show()


if __name__ == '__main__':
    filtered_df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/nlp/20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_tok_embeds.csv')

    zip = str(input())

    query = [input()]

    filt_df = sbert_search(filtered_df, zip, query)

    keywords = tfidf_keywords(filt_df)  # get list of words for top part of dash (and use to search documenu if want to)

    grams_plots(filt_df)  # get 2 plotly bar charts for mid-part of dash
