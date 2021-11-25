import numpy as np
import pandas as pd
import re
import collections

df = pd.read_csv('/Users/Matteo/Desktop/repo/capstone_project/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv')

# model bert
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

word = ["chicken", 'coffee']

inputs = tokenizer(word, return_tensors="pt")
outputs = model(**inputs)
word_vect = outputs.pooler_output.detach().numpy()