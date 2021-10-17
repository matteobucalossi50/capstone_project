#%%
################ Documenu ################
# The API is queried to get restaurant information within a given zipcode. 
# Documentation can be found here(https://documenu.com/docs#get_started).

#%%

# import libraries
import requests
import pandas as pd
import os
<<<<<<< HEAD
from os.path import join
import sys
sys.path.insert(1, 'C:\\Users\\raide\\OneDrive\\Documents\\GitHub\\capstone_project\\constants')
from constants import get_mp_documenu_api_key
=======
import sys
sys.path.insert(1, 'C:\\Users\\raide\\OneDrive\\Documents\\GitHub\\capstone_project\\constants')
from constants import get_mp_api_key
>>>>>>> 386a160a52747da40a10a4156bd42892b1b2eaa7
from ast import literal_eval

#%% 
def restaurants_by_zip(zipcode, size, full_menu=False):
    
    """
    Queries the documenu API to grab available restaurants in the provided zip code.
    
    Parameters:
    -----------
    zipcode (int): A valid zipcode in the United States.
    size (int): The number of restaurants to return.
    full_menu (bool): When True, adds a column containing the menu(s) available for each restaurant. Default is False.
    
    Returns:
    --------
<<<<<<< HEAD
    df (DataFrame): A pandas DataFrame of the results.
    """

    # Access API
    API_KEY = get_mp_documenu_api_key()
=======
    df_final (DataFrame): A pandas DataFrame of the results.
    """

    # Access API
    API_KEY = get_mp_api_key()
>>>>>>> 386a160a52747da40a10a4156bd42892b1b2eaa7
    # if full_menu:
    #     url = f'https://api.documenu.com/v2/restaurants/zip_code/{zipcode}?fullmenu=true&size={size}&key={API_KEY}'
    # else:
    #     url = f'https://api.documenu.com/v2/restaurants/zip_code/{zipcode}?size={size}&fullmenu=false&key={API_KEY}'
    # r = requests.get(url)
    
    url = f'https://api.documenu.com/v2/restaurants/zip_code/{zipcode}'
    
    if full_menu:
        payload = {
            'fullmenu': 'true',
            'key': API_KEY,
            # 'size': str(size) # The size parameter does not work when fullmenu is true
        }
    else:
        payload = {
            'size': str(size),
            'key': API_KEY,
            'fullmenu': 'false'
        }
    r = requests.get(url, params=payload)
    response = r.json()['data']

    # Build table
    df = pd.json_normalize(response)
    split_cuisines = pd.DataFrame(df.cuisines.tolist(), columns=['cuisine_1', 'cuisine_2', 'cuisine_3', 'cuisine_4', 'cuisine_5', 'cuisine_6'])
    df = pd.concat([df, split_cuisines], axis=1)
    
    if full_menu:
        df.drop(columns=['cuisines'], inplace=True)
        df.menus = df.menus.astype(str).apply(literal_eval)
        df = df.explode('menus').reset_index(drop=True)
        df = pd.json_normalize(df.to_dict(orient='records'))
        df = df.explode('menus.menu_sections').reset_index(drop=True)
        df = pd.json_normalize(df.to_dict(orient='records'))
        df = df.explode('menus.menu_sections.menu_items').reset_index(drop=True)
        df = pd.json_normalize(df.to_dict(orient='records'))
        df = df.explode('menus.menu_sections.menu_items.pricing').reset_index(drop=True)
        df = pd.json_normalize(df.to_dict(orient='records'))
        df = df.rename(columns={'menus.menu_name': 'menu_name',
            'menus.menu_sections.section_name': 'section_name',
            'menus.menu_sections.description': 'description',
            'menus.menu_sections.menu_items.name': 'menu_items.name',
            'menus.menu_sections.menu_items.description': 'menu_items.description',
            'menus.menu_sections.menu_items.price': 'menu_items.price'})
        df = df.drop(columns=['menus.menu_sections.menu_items','menus.menu_sections.menu_items.pricing.price','menus.menu_sections.menu_items.pricing.currency','menus.menu_sections.menu_items.pricing.priceString','menus.menu_sections.menu_items.pricing'])
    else:
        df.drop(columns=['cuisines', 'menus'], inplace=True)
<<<<<<< HEAD
        
    # Write to directory
    data_path = os.path.join(os.getcwd(), 'data')
    file_name = os.ath.join(data_path, f'{zipcode}_documenu_results.csv')
    df.to_csv(file_name)
=======
>>>>>>> 386a160a52747da40a10a4156bd42892b1b2eaa7

    return df
# %%
