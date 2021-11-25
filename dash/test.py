from logging import PlaceHolder
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
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
        dbc.FormText(
        "Enter a zip code to view results in that area.",
        color="secondary"),
        html.Br(),
        dbc.Button(
            id='submit_button',
            n_clicks=0,
            children='Submit',
            color='primary',
            block=True
        ),
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
                        html.H4(id='card_title_1', children=['Card Title 1'], className='card-title',
                                style=CARD_TEXT_STYLE),
                        html.P(id='card_text_1', children=['Sample text.'], style=CARD_TEXT_STYLE),
                    ]
                )
            ]
        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4('Card Title 2', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Card Title 3', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Card Title 4', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Sample text.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]
        ),
        md=3
    )
])

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_1'), md=4
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

content = html.Div(
    [
        html.H2("What are we going to eat?", style=TEXT_STYLE),
        html.Hr(),
        content_first_row,
        content_second_row,
        content_third_row
        # content_fourth_row
    ],
    style=CONTENT_STYLE
)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])

# df = pd.read_csv('/Users/test/Desktop/github_mp/capstone_project/documenu/data/documenu/documenu.csv', index_col=0)

@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input(component_id='submit_button', component_property='n_clicks')],
    [State(component_id='zipcode', component_property='value')]
)
def update_map(n_clicks, zipcode_value):
    from documenu import restaurants_by_zip
    
    if zipcode_value is None:
        print('button not clicked')
        
    print(n_clicks)
    print(zipcode_value)
    print(type(zipcode_value))
    
    if type(zipcode_value) == int:
        df = restaurants_by_zip(zipcode_value, 25, True)
        mapbox_access_token = open(".mapbox_token").read()
        restaurants = df.restaurant_name.unique().tolist()
        lat = df['geo.lat'].unique().tolist()
        lon = df['geo.lon'].unique().tolist()

        fig = [go.Scattermapbox(
            lat = lat,
            lon = lon,
            mode='markers',
            marker={
                'color': 'red'
            },
            unselected={'marker': {'opacity': 1}},
            selected={'marker': {'opacity': 0.5, 'size': 25}},
            text=restaurants
        )]
        
    return {
        'data': fig, 
        'layout': go.Layout(
            uirevision='foo',
            clickmode='event+select',
            hovermode='closest',
            autosize=True,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                style='light',
                bearing=0,
                center=dict(
                    lat = lat[round(len(lat)/2)],
                    lon = lon[round(len(lon)/2)]
                ),
                pitch=0,
                zoom=10,
            ),
        )
    }
    
#%%
app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter
