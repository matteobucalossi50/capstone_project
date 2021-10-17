# %%
# Import libraries
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime
import os
from os.path import join
import sys
sys.path.insert(1, 'C:\\Users\\raide\\OneDrive\\Documents\\GitHub\\capstone_project\\constants')
from constants import chrome_path


#%%
# Setup Selenium

# Chrome driver
CHROME_PATH = chrome_path()
# CHROMEDRIVER_PATH = os.path.join(os.getcwd(), 'chromedriver_win32', 'chromedriver.exe')
CHROMEDRIVER_PATH = r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\scraping\chromedriver_win32\chromedriver.exe'
print(CHROMEDRIVER_PATH)
WINDOW_SIZE = '1920,1080'

# Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.binary_location = CHROME_PATH

# Initialize Driver
driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=chrome_options)

#%%
def enchanted_learning_food(driver=driver):
    """
    Parameters:
    -----------
        driver (selenium.webdriver.chrome.webdriver.WebDriver) = Selenium webdriver.
        
    Returns:
    --------
        raw_word_list (list): List of words from https://www.enchantedlearning.com/wordlist/food.shtml. 
    """
    
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
    Visits https://ritetag.com/hashtag-comparison/ to get statistics about the supplied hashtags and writes the resulting DataFrame to the directory.
    
    Parameters:
    -----------
    driver (selenium.webdriver.chrome.webdriver.WebDriver) = Selenium webdriver.
    
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

    # Write to directory
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S_')
    data_path = os.path.join(os.getcwd(), 'data')
    file_path = os.path.join(data_path, to_csv_timestamp + '_hashtag_stats.csv')
    # df.to_csv(file_path, index=False)
    
    return df
# %%
