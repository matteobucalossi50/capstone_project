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
from gensim.models import LdaModel, CoherenceModel

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from textblob import TextBlob

df_stream = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_082018_clean_streaming_custom_hashtags_data.csv')

df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv')

# documenu = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/data/documenu.csv')

# documenu = documenu[documenu['menu_items.description'].notna()]
# menu_items = documenu['menu_items.description'].values.tolist()


def process_words(texts, stop_words=stopwords):
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
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
    # texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3)
    #               if word not in stop_words] for doc in texts_out]
    return texts_out, noun, adj, relevants

#######
### nlp
texts = df.tweet_text.values.tolist()
texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

out_toks, nouns, adjs, relev_tokens = process_words(texts, stopwords)

# tokenized_sents = [word_tokenize(i) for i in texts]

flat_tokens = [item for sublist in relev_tokens for item in sublist]

fdist = FreqDist(flat_tokens)
top_50k = fdist.most_common(50000)

print(top_50k)

plt.gcf().subplots_adjust(bottom=0.55)  # to avoid x-ticks cut-off
fdist.plot(30, title='Tokens Frequency Distribution')


bigr = [nltk.bigrams(sent) for sent in relev_tokens]
flat_bigr = [item for sublist in bigr for item in sublist]

fdist_bigr = nltk.FreqDist(flat_bigr)

plt.gcf().subplots_adjust(bottom=0.45) # to avoid x-ticks cut-off
fdist_bigr.plot(30, title='Bigrams Frequency Distribution')


# frequency
freq_tok = {}
for t in flat_tokens:
    if t in freq_tok:
        freq_tok[t] += 1
    else:
        freq_tok[t] = 1

sorted_freq_tok = sorted(freq_tok.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

# dictionary freq
sorted_freq_tok_dic = {}
for a,b in sorted_freq_tok:
    sorted_freq_tok_dic.setdefault(a, []).append(b)


## grams-association
def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """

    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

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

def print_most_frequent(ngrams, num=150):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

trigrams = count_ngrams(relev_tokens, 3, 3)
print_most_frequent(trigrams)


def word_association(ngrams, target):
    targ_grams = {}
    for keys, values in ngrams.items():
        targ = {}
        for k, v in values.items():
            if target in k:
                targ[k] = v

        targ = collections.Counter(dict(sorted(targ.items(), key=lambda item: item[1], reverse=True)))
        targ_grams[keys] = targ

    return targ_grams

targ_grams = word_association(trigrams, 'chicken')
print_most_frequent(targ_grams)


### LDA
out_toks_lda = [' '.join(tok) for tok in relev_tokens]

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


## plot
import pyLDAvis
import pyLDAvis.sklearn

topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]
doc_lengths = data_vect.sum(axis=1).getA1()
term_frequency = data_vect.sum(axis=0).getA1()
lda_doc_topic_dists = lda.transform(data_vect)
doc_topic_dists = lda_doc_topic_dists / lda_doc_topic_dists.sum(axis=1)[:, None]
vocab = vectorizer.get_feature_names()

lda_pyldavis = pyLDAvis.prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
pyLDAvis.display(lda_pyldavis)


## embeddings-space
import multiprocessing
from gensim.models import Word2Vec

common_tokens = []
for sent in relev_tokens:
    new_sent = []
    for tok in sent:
        if tok in [t[0] for t in top_50k]:
            new_sent.append(tok)
    common_tokens.append(new_sent)

# word2vec model
model = gensim.models.Word2Vec(sentences=common_tokens)

# arbitrary topic
keys = ['coffee', 'chicken', 'meat', 'dessert']

# functionalize(from df to output to plot)

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.wv.most_similar(word, topn=50):
        words.append(similar_word)
        embeddings.append(model.wv[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


from sklearn.manifold import TSNE
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

import matplotlib.cm as cm

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
    plt.show()

tsne_plot_similar_words('Similar words from Twitter', keys, embeddings_en_2d, word_clusters, 0.7, 'similar_words.png')


# 3d plot
import plotly.graph_objs as go
import nbformat

def append_list(sim_words, words):

    list_of_words = []

    for i in range(len(sim_words)):

        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words

result_word = []
for words in keys:

    sim_words = model.wv.most_similar(words, topn=10)
    sim_words = append_list(sim_words, words)

    result_word.extend(sim_words)

similar_word = [word[0] for word in result_word]
similarity = [word[1] for word in result_word]
similar_word.extend(keys)
labels = [word[2] for word in result_word]
label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
color_map = [label_dict[x] for x in labels]


def display_tsne_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=5, sample=10):

    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.wv.key_to_index), sample)
        else:
            words = [word for word in model.wv.key_to_index]

    word_vectors = np.array([model.wv[w] for w in words])
    three_dim = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate, n_iter=iteration).fit_transform(word_vectors)[:, :3]

    # For 2D, change the three_dim variable into something like two_dim like the following:
    # two_dim = TSNE(n_components = 2, random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    for i in range(len(user_input)):
        trace = go.Scatter3d(
            x=three_dim[count:count+topn, 0],
            y=three_dim[count:count+topn, 1],
            z=three_dim[count:count+topn, 2],
            text=words[count:count+topn],
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }
        )

        # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
        # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
        data.append(trace)
        count = count+topn

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=words[count:],
        name='input words',
        textposition="top center",
        textfont_size=20,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'black'
        }
    )

    # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
    # variable three_dim, use the variable that we have declared earlier (e.g two_dim)

    data.append(trace_input)
    # Configure the layout
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()


import plotly.io as pio
pio.renderers.default = "browser"

# display_tsne_scatterplot_3D(model, user_input=None, words=None, label=None, color_map=None, perplexity = 0, learning_rate = 0, iteration = 0, topn=5, sample=10)
display_tsne_scatterplot_3D(model, keys, similar_word, labels, color_map, 5, 500, 10000)



# words_3d = []
# embeddings_3d = []
# for word in list(model.wv.key_to_index):
#     embeddings_3d.append(model.wv[word])
#     words_3d.append(word)
#
# tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
# embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_3d)
#
# from mpl_toolkits.mplot3d import Axes3D
#
# def tsne_plot_3d(title, label, embeddings, a=1):
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     colors = cm.rainbow(np.linspace(0, 1, 1))
#     plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
#     plt.legend(loc=4)
#     plt.title(title)
#     plt.show()
#
#
# tsne_plot_3d('Visualizing Embeddings using t-SNE', 'Tweets', embeddings_wp_3d, a=0.2)



## geo-location engineering
from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()

zipcodes = []
rows = []
for i, r in df.iterrows():
    place = r['location']
    if type(place) == str:
        place = place.split(',')[0]
        zip = zcdb.find_zip(city=place)
        if zip==None:
            pass
        else:
            zip = zip[0].zip
            zipcodes.append(zip)
            rows.append(i)

# filtered_df = df[df['location'].apply(lambda x: isinstance(x, str))]
filtered_df = df.iloc[rows]
filtered_df['zipcode'] = zipcodes

## save df
filtered_df.to_csv('20211026_195518_clean_scraping_custom_hashtags_data_zipcodes.csv')