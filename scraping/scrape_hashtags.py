# %%
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime
import os

# Chrome driver
CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
CHROMEDRIVER_PATH = r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\scraping\chromedriver_win32\chromedriver'
WINDOW_SIZE = '1920,1080'

# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.binary_location = CHROME_PATH

# Begin Selenium service
# service = Service(r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\scraping\chromedriver_win32\chromedriver')
# service.start()

# Initialize Driver
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)

#%%
def enchanted_learning_food(driver=driver):
    
    # Navigate to page
    url = 'https://www.enchantedlearning.com/wordlist/food.shtml'
    driver.get(url)
    time.sleep(2)

    # Create list of words from page
    word_list_from_browser = driver.find_elements_by_class_name('wordlist-item')
    raw_word_list = [s.text for s in word_list_from_browser]
    # hashtags_from_word_list = ['#' + s.replace(' ', '') for s in raw_word_list]

    return raw_word_list

#%%

def get_hashtag_stats(driver=driver):
    """
    Visits https://ritetag.com/hashtag-comparison/ to get statistics about the supplied hashtags.
    
    Parameters:
    -----------
    hashtag_list (list of strings): The hashtags of interest, without the '#'.
    
    Return:
    -------
    df (DataFrame): A pandas DataFrame of the result.
    """
    
    hashtag_list = enchanted_learning_food()

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

    data = {'hashtag': hashtag_list_from_browser[:len(unique_tweets_per_hour)], 'unique_tweets_per_hour': unique_tweets_per_hour, 'retweets_per_hour': retweets_per_hour, 'views_per_hour': hashtag_exposure_per_hour}

    # build dataframe
    df = pd.DataFrame(data)
    df = df.replace(',', '', regex=True)
    df[['unique_tweets_per_hour', 'retweets_per_hour', 'views_per_hour']] = df[['unique_tweets_per_hour', 'retweets_per_hour', 'views_per_hour']].apply(pd.to_numeric)
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S_')
    path = r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\data\hashtag_stats'
    filename = path + '\\' + to_csv_timestamp + 'hashtag_stats.csv'
    df.to_csv(filename, index=False)
    
    return df

# %%
# raw_word_list, hashtags_from_word_list = enchanted_learning_food()
# df = get_hashtag_stats(raw_word_list)
# df

# %%
