# Update combined data to contain one location column
# Preprocess zipcodes before pushing

#%%
import pandas as pd
import numpy as np
from pyzipcode import ZipCodeDatabase

df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project.nosync/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv', index_col=0)

zcdb = ZipCodeDatabase()
locations_df = df.dropna(subset=['location']).reset_index(drop=True)

locations = locations_df.location.tolist()
locations = [location for location in locations if ',' in location]

zipcodes = []
city_state_zip = []
errors = []

#%%
for location in locations:
# Grab city and state information from each tweet
    try:
        city, state = location.split(', ')
        zip_codes = zcdb.find_zip(city=city, state=state)
        for zip_code in zip_codes:
            city_state_zip.append(zip_code.zip)
        zipcodes.append(city_state_zip)
    except:
        errors.append(location)
else:
    pass

print(zipcodes)
# %%
## geo-location engineering
from pyzipcode import ZipCodeDatabase
import pandas as pd
import numpy as np
df = pd.read_csv("/Users/test/Desktop/github_mp/capstone_project/tweepy/data/20211202_172249_clean_combined_custom_hashtags_data.csv", index_col=0)

zcdb = ZipCodeDatabase()
locations_df = df.dropna(subset=['location'])

locations = locations_df.location.tolist()
# locations = [location for location in locations if ',' in location]

zipcodes = []
city_state_zip = []
errors = []
i = -1

for location in locations:
    city_state_zip = []
    i += 1
    if i < len(locations_df):
        try:
            city, state = location.split(', ')
            zip_codes = zcdb.find_zip(city=city, state=state)
            for zipcode in zip_codes:
                city_state_zip.append(zipcode.zip)
        except:
            city_state_zip.append(np.nan)
        print(i)
        print(city_state_zip[0])
        zipcodes.append(city_state_zip)

#%%
## save df
zipcodes_series = pd.Series(zipcodes)
locations_df['zipcode_list'] = zipcodes_series
locations_df.reset_index(drop=True, inplace=True)
locations_df.head(20)

# %%
locations_df.to_csv('20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_list.csv')

#%%
# For combined dataset 
from pyzipcode import ZipCodeDatabase
import pandas as pd
import numpy as np
import pandas as pd

df = pd.read_csv("/Users/test/Desktop/github_mp/capstone_project/tweepy/data/20211202_172249_clean_combined_custom_hashtags_data.csv", index_col=0)

def place_to_city_state(x):
    if type(x) == str:
        full_name_i = x.find("full_name='")
        full_name_j = x.find("', country_code")
        return x[full_name_i+11:full_name_j]
df['city_state_from_place'] = df['place'].apply(lambda x: place_to_city_state(x))
df['city_state_from_place'].value_counts()

#%%
from pyzipcode import ZipCodeDatabase
import pandas as pd
import numpy as np

def location_to_zip(x):
    zcdb = ZipCodeDatabase()
    city_state_zip = []
    try:
        city, state = x.split(', ')
        zip_codes = zcdb.find_zip(city=city, state=state)
        for zipcode in zip_codes:
            city_state_zip.append(zipcode.zip)
    except:
        return np.nan
    return city_state_zip

df['all_zips1'] = df['location'].apply(lambda x: location_to_zip(x))
df['all_zips2'] = df['city_state_from_place'].apply(lambda x: location_to_zip(x))

#%%
# For tokens and embeddings
import pandas as pd
from nltk.corpus import stopwords
import re
import preprocessor as p
from gensim.utils import simple_preprocess
import spacy
nlp = spacy.load("en_core_web_sm")
stopwords = stopwords.words('english')
stopwords.extend(['https', 'http', 'com', 'tbt', 'ico', 'foodie', 'food', 'please', 'love', 'home'])

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

texts = df.tweet_text.values.tolist()
texts = [re.sub(r'https?://\S+', '', rev) for rev in texts]

out_toks, nouns, adjs, relev_tokens = process_words(texts, stopwords)

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/nlp/20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_list.csv', index_col=0)

model = SentenceTransformer('msmarco-distilbert-base-v4')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

texts = df.tweet_text.values.tolist()
sbert_embeddings = model.encode(texts, convert_to_tensor=True)

df['tweet_embeddings'] = sbert_embeddings.tolist()
df['tokens'] = relev_tokens

# %%
df.to_csv('/Users/test/Desktop/github_mp/capstone_project/tweepy/data/20211202_172249_clean_combined_custom_hashtags_data.csv')
# %%
