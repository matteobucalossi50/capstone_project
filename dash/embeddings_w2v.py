#%%
import numpy as np
import pandas as pd
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

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import multiprocessing
from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.manifold import TSNE
#%%

def process_words(texts, stop_words=stopwords):
    # remove stopwords, short tokens and letter accents
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
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3)
                    if word not in stop_words] for doc in texts_out]
    return texts_out, noun, adj, relevants


def tokenization(df):
    texts = df.tweet_text.values.tolist()
    texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

    out_toks, nouns, adjs, relev_tokens = process_words(texts, stopwords)

    flat_tokens = [item for sublist in relev_tokens for item in sublist]

    fdist = FreqDist(flat_tokens)
    top_50k = fdist.most_common(50000)

    common_tokens = []
    for sent in relev_tokens:
        new_sent = []
        for tok in sent:
            if tok in [t[0] for t in top_50k]:
                new_sent.append(tok)
        common_tokens.append(new_sent)

    return common_tokens


def w2v_model(tokens, queries):
    model = gensim.models.Word2Vec(sentences=tokens)

    # functionalize (from df to output to plot)
    embedding_clusters = []
    word_clusters = []
    for similar_word, _ in model.wv.most_similar(queries, topn=50):
        word_clusters.append(similar_word)
        embedding_clusters.append(model.wv[similar_word])

    return embedding_clusters, word_clusters


def tsne_plot(embedding_clusters, word_clusters, queries):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            plt.scatter(x, y, c=color, alpha=a, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=.9, xy=(x[i], y[i]), xytext=(5, 4),
                                textcoords='offset points', ha='right', va='bottom', size=10)
        plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        if filename:
            plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        # plt.show()

    tsne_plot_similar_words('Similar words from Twitter', queries, embeddings_en_2d, word_clusters, 0.7, 'similar_words.png')

if __name__ == '__main__':
    df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv')

    query = [input()]

    tokens = tokenization(df)
    embedding_clusters, word_clusters = w2v_model(tokens, query)

    tsne_plot(embedding_clusters, word_clusters, query)

# %%
