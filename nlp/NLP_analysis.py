import numpy as np
import pandas as pd
import re
# import spacy
# nlp = spacy.load('en_core_web_sm')

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
from gensim.models import LdaModel, CoherenceModel

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from textblob import TextBlob

df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/data/20211011_093034_all_food_tweets_clean.csv')

documenu = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/data/documenu.csv')


###nlp
texts = df.tweet_text.values.tolist()
texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

# tokenized_sents = [word_tokenize(i) for i in texts]
# flat_tokens = [item for sublist in tokenized_sents for item in sublist]

out_toks = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stopwords] for doc in texts]
flat_tokens = [item for sublist in out_toks for item in sublist]

fdist = FreqDist(flat_tokens)
top_100 = fdist.most_common(100)

print(top_100)

plt.gcf().subplots_adjust(bottom=0.35) # to avoid x-ticks cut-off
fdist.plot(30, title='Tokens Frequency Distribution')


bigr = [nltk.bigrams(sent) for sent in out_toks]
flat_bigr = [item for sublist in bigr for item in sublist]

fdist_bigr = nltk.FreqDist(flat_bigr)

plt.gcf().subplots_adjust(bottom=0.55) # to avoid x-ticks cut-off
fdist_bigr.plot(30, title='Bigrams Frequency Distribution')


###LDA
out_toks_lda = [' '.join(tok) for tok in out_toks]

vectorizer = CountVectorizer(analyzer='word', min_df=10, lowercase=True
                             , token_pattern='[a-zA-Z0-9]{3,}')

data_vect = vectorizer.fit_transform(out_toks_lda)

lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50, random_state=0)

lda.fit(data_vect)


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)

        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)

        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


tf_feature_names = vectorizer.get_feature_names()
plot_top_words(lda, tf_feature_names, 10, 'Topics in LDA model')


# #plot
# import pyLDAvis
# import pyLDAvis.sklearn
#
# topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]
# doc_lengths = data_vect.sum(axis=1).getA1()
# term_frequency = data_vect.sum(axis=0).getA1()
# lda_doc_topic_dists = lda.transform(data_vect)
# doc_topic_dists = lda_doc_topic_dists / lda_doc_topic_dists.sum(axis=1)[:, None]
# vocab = vectorizer.get_feature_names()
#
# lda_pyldavis = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
# pyLDAvis.display(lda_pyldavis)

