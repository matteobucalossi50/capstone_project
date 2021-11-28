from logging import PlaceHolder
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from nltk.corpus.reader import twitter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from pyzipcode import ZipCodeDatabase
from embeddings_w2v import w2v_model, tsne_plot
from documenu import restaurants_by_zip

        
twitter_df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/nlp/20211026_195518_clean_scraping_custom_hashtags_data_zipcodes.csv', index_col=0)

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

FOOTER_TEXT_STYLE = {
    'textAlign': 'center',
    'color': 'grey',
    'padding': '0 0 10px 0'
}

controls = dbc.FormGroup(
    [
        html.P('Zip code', style={
            'textAlign': 'center'
        }),
        dbc.Input(
            id='zipcode',
            type='number',
            min='00000',
            max='99999',
            step=1,
            placeholder='Input a five-digit zip code...'),
        html.Br(),
        html.P('Food', style={
            'textAlign': 'center'
        }),
        dbc.Input(
            id='food',
            type='text',
            placeholder='Input a food or drink...'
        ),
        html.Br(),
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            block=True
        )
    ]
)

sidebar = html.Div(
    [
        html.H2('Find trending food in your area', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                    html.H4(id='card_title_1', children=["Find out what's trending!"], className='card-title', style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=[], style=CARD_TEXT_STYLE),
                    ]
                ),
                dcc.Store(id='intermediate-value')
            ]
        ),
    )
])

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='tsne', figure={}), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_2'), md=4
        ),
        dbc.Col(
            dcc.Graph(id='graph_3'), md=4
        )
    ],
)

content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='map', figure={}), md=12,
        )
    ]
)

footer = dbc.Row(
    [
        dbc.Col(
            html.Footer(id='footer', children=['Michael Pagan | Matteo Buccalosi'])
        )
    ],
    style=FOOTER_TEXT_STYLE
)

content = html.Div(
    [
        html.H2("DATS 6501 Capstone: Mining & Modeling Food Trends", style=TEXT_STYLE),
        html.Hr(),
        content_first_row,
        content_second_row,
        content_third_row,
        footer
        # content_fourth_row
    ],
    style=CONTENT_STYLE
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])

# documenu_df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/documenu/data/documenu/documenu.csv', index_col=0)
@app.callback(
    Output('intermediate-value', 'data'),
    [Input('submit_button', 'n_clicks')],
    [State('zipcode', 'value'), State('food', 'value')], 
    prevent_initial_call=True
)
def process_data(n_clicks, zipcode_value, food_value):
    
    print(n_clicks)
    print(zipcode_value)
    print(type(zipcode_value))

    if type(zipcode_value) == int:
        print('success')
        # Narrow twitter data to the provided zipcode
        
        # Create and run model
        twitter_dfc = twitter_df[twitter_df['zipcode'] == zipcode_value]
        print('twitter subset', len(twitter_dfc))
        tokens = pickle.load(open('tokens_pkl.p', 'rb'))
        print('success2')
        embedding_clusters, word_clusters = w2v_model(tokens, [food_value])
        print(word_clusters)
        print(embedding_clusters)
        print('success3')
        
        # Get menu information for the zipcode
        documenu_df = restaurants_by_zip(zipcode_value, 25, True)
        documenu_df = documenu_df.to_json(date_format='iso', orient='split')
        # bool_series = documenu_df['menu_items.description'].isin(word_clusters)
        # documenu_df = documenu_df[bool_series]
        # print(documenu_df.head())
    
    return [word_clusters, documenu_df, zipcode_value, food_value, embedding_clusters]

@app.callback(
    Output('card_title_1', 'children'),
    Output('card_text_1', 'children'),
    Output('map', 'figure'),
    # Output('tsne', 'figure'),
    [Input('intermediate-value', 'data')],
    prevent_initial_call=True
)
def update_page(data):
    print(type(data))
    
    word_clusters, documenu_df, zipcode_value, food_value, embedding_clusters = data[0], data[1], data[2], data[3], data[4]
    
    # Update card
    zcdb = ZipCodeDatabase()
    city, state = zcdb[zipcode_value].city, zcdb[zipcode_value].state, 
    title = f"Here's what people on Twitter are saying about {food_value} in {city}, {state}: "
    text = ', '.join(word_clusters)
    print(title)
    print(text)
    
    # Update map
    mapbox_access_token = open(".mapbox_token").read()
    documenu_copy = pd.read_json(documenu_df, orient='split')
    print(documenu_copy.shape)
    documenu_copy = documenu_copy.loc[documenu_copy['menu_items.description'].str.contains(food_value, case=False, na=False)]
    print(documenu_copy.shape)
    restaurants = documenu_copy.restaurant_name.unique().tolist()
    lat = documenu_copy['geo.lat'].unique().tolist()
    lon = documenu_copy['geo.lon'].unique().tolist()

    fig = [go.Scattermapbox(
        lat = lat,
        lon = lon,
        mode='markers',
        marker={
            'color': 'blue'
        },
        unselected={'marker': {'opacity': 0.5}},
        selected={'marker': {'opacity': 0.2, 'size': 25}},
        text=restaurants
    )]
    
    print(lat)
    print(lon)
    
    map = {
        'data': fig, 
        'layout': go.Layout(
            title='Restaurants',
            uirevision='foo',
            clickmode='event+select',
            hovermode='closest',
            autosize=True,
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                style='light',
                bearing=0,
                center=dict(
                    lat = lat[round(len(lat)/2)],
                    lon = lon[round(len(lon)/2)]
                ),
                pitch=0,
                zoom=11,
            ),
        )
    }
    
    # Update tsne graph
    # tsne_graph = tsne_plot(embedding_clusters, word_clusters, [food_value])

    return title, text, map #, tsne_graph
#%%
app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter
