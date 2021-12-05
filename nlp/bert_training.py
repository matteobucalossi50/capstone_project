import numpy as np
import pandas as pd
import re
import collections

df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv')
texts = df.tweet_text.values.tolist()

# model bert
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

word = ["chicken", 'coffee']

inputs = tokenizer(word, return_tensors="pt")
outputs = model(**inputs)
word_vect = outputs.pooler_output.detach().numpy()
from transformers import AutoModel, AutoTokenizer
import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# inputs = tokenizer(word, return_tensors="pt")
# outputs = model(**inputs)
# word_vect = outputs.pooler_output.detach().numpy()

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

word = ["chicken", 'coffee']

embds = []
for tweet in texts:
    input_ids = torch.tensor([tokenizer.encode(tweet)])
    with torch.no_grad():
        features = bertweet(input_ids)
        embds.append(features)


inputs = tokenizer(word, return_tensors="pt")
outputs = model(**inputs)
word_vect = outputs.pooler_output.detach().numpy()