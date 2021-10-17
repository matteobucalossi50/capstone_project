def get_mp_documenu_api_key():
    # return 'c7f971117b34db96607a989c10f0d985'
    return '53b7426a41102cb6e217635a882c36d1'
    # return 'fb1be0d8f88649210207adab97853f34'
    
def get_matteo_twitter_creds():
    access_token = '2559294522-9beAFYriWwSdFBnpGZMlEuiFmLwWFsU9yDyXlXU'
    # bearer 'AAAAAAAAAAAAAAAAAAAAACKwSAEAAAAAy%2FFd2hCP7rY9j61xMjWhvg1MsHk%3DlMS19IuX2vpYNU7L2hP33P5eNc7MoHvIT5Hc6QagASmEcKYstw'
    access_token_secret = 'x2eNxpidxxqLSbLvL4eYpoEaBTRQM0eYD8keqr3YnZxlL'
    consumer_key = '57HDK4QvI5XCZn4Dq2Oa27vdN'
    consumer_secret = 'bM59AFPQy9jqI8xp7yqBb6zNUGVcdzIsqdTIyuHOLlNvmXzIJ9'
    return access_token, access_token_secret, consumer_key, consumer_secret

# usda api key: ZUx6WsT4DdeWPqibDECIU3I1gYFcC5zNYGIyc4Xc
def get_michael_twitter_creds():
    access_token = '798261927470399489-ZG8gkHvrcPsy5DwVV7ddJA4sZuinIR7'
    # bearer 'AAAAAAAAAAAAAAAAAAAAACKwSAEAAAAAy%2FFd2hCP7rY9j61xMjWhvg1MsHk%3DlMS19IuX2vpYNU7L2hP33P5eNc7MoHvIT5Hc6QagASmEcKYstw'
    access_token_secret = 'gelkhvtRiTzekSWkiHvPSQs3F1ff1felRAs48b4BDnnUh'
    consumer_key = 'HL3LtKtQsIbXTNXEzG0DdqOcg'
    consumer_secret = 'bLriKHY08atzvRDf0KFa3pYGEwABbEVEgpHSsWqGbYNIrI2tqy'
    return access_token, access_token_secret, consumer_key, consumer_secret

# api key: HL3LtKtQsIbXTNXEzG0DdqOcg
# api secret key: bLriKHY08atzvRDf0KFa3pYGEwABbEVEgpHSsWqGbYNIrI2tqy
# bearer token: AAAAAAAAAAAAAAAAAAAAAFIDUwEAAAAAlYCursbswfBbzMY2QXFQ6Vz6oY4%3D59wgu9t5NxCx77Mo5rHIl22gMfsopW7onPB58wtXBgRXf1W49M

def chrome_path():
    import os
    from os.path import join
    path = os.path.join('C:\Program Files\Google\Chrome\Application\chrome.exe')
    return path
