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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import preprocessor as p
from textblob import TextBlob

df_stream = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_082018_clean_streaming_custom_hashtags_data.csv')

df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv')

# documenu = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/data/documenu.csv')

# documenu = documenu[documenu['menu_items.description'].notna()]
# menu_items = documenu['menu_items.description'].values.tolist()


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
# filtered_df.to_csv('20211026_195518_clean_scraping_custom_hashtags_data_zipcodes.csv')


#######
### nlp
texts = filtered_df.tweet_text.values.tolist()
texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

out_toks, nouns, adjs, relev_tokens = process_words(texts, stopwords)

# tokenized_sents = [word_tokenize(i) for i in texts]

flat_tokens = [item for sublist in relev_tokens for item in sublist]

fdist = FreqDist(flat_tokens)
top_50k = fdist.most_common(50000)

print(top_50k)

plt.gcf().subplots_adjust(bottom=0.35)  # to avoid x-ticks cut-off
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

def print_most_frequent(ngrams, num=150):
    """Print num most common n-grams of each length in n-grams dict."""
    grams = []
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams.most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
            grams.append(' '.join(gram), count)
        print('')
        return grams

trigrams = count_ngrams(relev_tokens, 3, 3)
grs = print_most_frequent(trigrams, 20)


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

lda = LatentDirichletAllocation()

search_params = {'n_components': [10, 15, 20], 'learning_decay': [.9, 1, 1.5]}
# Init Grid Search Class
grid_lda = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
grid_lda.fit(data_vect)

best_lda_model = grid_lda.best_estimator_

# Model Parameters
print("Best Model's Params: ", grid_lda.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", grid_lda.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vect))


n_topics = [10, 15, 20]
log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in grid_lda.cv_results_ if gscore.params['learning_decay']==0.5]
log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in grid_lda.grid_scores_ if gscore.parameters['learning_decay']==0.7]
log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in grid_lda.grid_scores_ if gscore.parameters['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()

# # Show top n keywords for each topic
# def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
#     keywords = np.array(vectorizer.get_feature_names())
#     topic_keywords = []
#     for topic_weights in lda_model.components_:
#         top_keyword_locs = (-topic_weights).argsort()[:n_words]
#         topic_keywords.append(keywords.take(top_keyword_locs))
#     return topic_keywords
#
# topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
#
# # Topic - Keywords Dataframe
# df_topic_keywords = pd.DataFrame(topic_keywords)
# df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
# df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
# df_topic_keywords

# lda.fit(data_vect)


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 10, figsize=(30, 15), sharex=True)
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
plot_top_words(best_lda_model, tf_feature_names, 10, 'Topics in LDA model')


## plot
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
keys = ['coffee', 'chicken']

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
    fig = go.Figure()
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        fig.adgo.Scatter(x, y, c=color, alpha=a, label=label)
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



## tfidf
import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)



## search on transformers
from transformers import BertModel, BertConfig
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

model = SentenceTransformer('msmarco-distilbert-base-v4')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

texts = filtered_df.tweet_text.values.tolist()

sbert_embeddings = model.encode(texts, convert_to_tensor=True)

filtered_df['tweet_embeddings'] = sbert_embeddings.tolist()

filtered_df['tokens'] = relev_tokens

filtered_df.to_csv('20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_tok_embeds.csv')


## zipcodes filter
zipped_df = filtered_df[filtered_df['zipcode'] == '34145']
filt_texts = zipped_df.tweet_text.values.tolist()
# filt_embeds = zipped_df.tweet_embeddings.values.tolist()


## embed search on zipcodes
filt_embeds = model.encode(filt_texts, convert_to_tensor=True)

# filtered_df['tweet_embeddings'] = filt_embeds.tolist()


query = input()
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
    print("\t{:3f}\t{}".format(hit['cross-score'], texts[hit['corpus_id']].replace("\n", "")))


zipped_df_searched = zipped_df.iloc[rows]


## tfidf on zipcodes
zipped_tweets_searched = zipped_df_searched.tweet_text.values.tolist()
zipped_tweets_searched = ' '.join(zipped_tweets_searched)
bloblist = [tb(zipped_tweets_searched)]

for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    words = []
    for word, score in sorted_words[:100]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
        words.append(word)


## bigrams/trigrams
zipped_toks_searched = zipped_df_searched.tokens.values.tolist()

bigr = [nltk.bigrams(sent) for sent in zipped_toks_searched]
flat_bigr = [item for sublist in bigr for item in sublist]

fdist_bigr = nltk.FreqDist(flat_bigr)

plt.gcf().subplots_adjust(bottom=0.45)  # to avoid x-ticks cut-off
fdist_bigr.plot(30, title='Bigrams Frequency Distribution')


trigrams = count_ngrams(zipped_toks_searched, 3, 3)
# print_most_frequent(trigrams)
bigrams = count_ngrams(zipped_toks_searched, 2, 2)

targ_grams = word_association(trigrams, 'apple')
# print_most_frequent(targ_grams)

import plotly.graph_objects as go

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


# cv=CountVectorizer(max_df=0.85)
# word_count_vector=cv.fit_transform(filt_texts)
#
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector)


### tf-idf finds keywords of tweets in the zipcode
### trigrams frequencies shows what's simply more common
### if want to search for something + specific, use SBERT to retrieve closer tweets
### and then give out tfidf/trigrams results
### can still show similairties with w2v model
### try elmo/bert-words to search and for similarity - will fail, document why fail

### do grid search for lda
### do sbert on tweets and query engine
### plots for grams
### code tfidf for filtered (preproc and all for subset)
### tables/clouds for keywords
### ppt
### dash concordance
### find like 3/4 zipcodes and queries for demo

