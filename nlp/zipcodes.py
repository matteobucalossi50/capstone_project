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
df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project.nosync/tweepy/data/20211026_195518_clean_scraping_custom_hashtags_data.csv', index_col=0)

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

# %%
