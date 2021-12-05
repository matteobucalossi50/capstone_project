#%%
import pandas as pd

tok_embed_df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/nlp/20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_tok_embeds.csv')

zipcode_list_df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/nlp/20211026_195518_clean_scraping_custom_hashtags_data_zipcodes_list.csv')

len(tok_embed_df) == len(zipcode_list_df)
# %%
