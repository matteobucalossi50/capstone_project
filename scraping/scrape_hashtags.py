# %%
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import pandas as pd
from datetime import datetime
import os

def get_hashtag_stats(hashtag_list):
    """
    Visits https://ritetag.com/hashtag-comparison/ to get statistics about the supplied hashtags.
    
    Parameters:
    -----------
    hashtag_list (list of strings): The hashtags of interest, without the '#'.
    
    Return:
    -------
    df (DataFrame): A pandas DataFrame of the result.
    """
    
    # Begin Selenium service
    service = Service(r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\scraping\chromedriver_win32\chromedriver')
    service.start()
    driver = webdriver.Remote(service.service_url)

    # Create URL
    hashtag_url_list = [s + '%7C' if i != len(hashtag_list) - 1 else s for i, s in enumerate(hashtag_list)]
    hashtag_url_str = "".join(hashtag_url_list)
    url = 'https://ritetag.com/hashtag-comparison/' + hashtag_url_str
    driver.get(url)
    time.sleep(2)

    # list of hashtags page info
    hashtag_list_from_browser = driver.find_elements_by_class_name('tagstyleBig')
    hashtag_list_from_browser = [s.text.replace('#', '') for s in hashtag_list_from_browser[6:-6]]

    # list of unique tweets per hour
    unique_tweets_per_hour = driver.find_elements_by_class_name('htagUniqueTweets')
    unique_tweets_per_hour = [s.text for s in unique_tweets_per_hour]

    # list of retweets per hour
    retweets_per_hour = driver.find_elements_by_class_name('htagRetweets')
    retweets_per_hour = [s.text for s in retweets_per_hour]

    # hashtag exposure per hour
    hashtag_exposure_per_hour = driver.find_elements_by_class_name('htagViews')
    hashtag_exposure_per_hour = [s.text for s in hashtag_exposure_per_hour]

    # quit driver
    driver.quit()

    data = {'hashtag': hashtag_list_from_browser, 'unique_tweets_per_hour': unique_tweets_per_hour, 'retweets_per_hour': retweets_per_hour, 'views_per_hour': hashtag_exposure_per_hour}

    # build dataframe
    df = pd.DataFrame(data)
    df = df.replace(',', '', regex=True)
    df[['unique_tweets_per_hour', 'retweets_per_hour', 'views_per_hour']] = df[['unique_tweets_per_hour', 'retweets_per_hour', 'views_per_hour']].apply(pd.to_numeric)
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S_')
    path = r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\data'
    filename = path + '\\' + to_csv_timestamp + 'hashtag_stats.csv'
    df.to_csv(filename, index=False)
    
    return df

# %%
