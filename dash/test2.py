#%%
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

#%%
mapbox_access_token = open(".mapbox_token").read()
df = pd.read_csv(r'C:\Users\raide\OneDrive\Documents\GitHub\capstone_project\documenu\data\documenu\documenu.csv', index_col=0)

app = dash.Dash(__name__)
blackbold = {'color': 'black', 'font-weight': 'bold'}

app.layout = html.Div([
    html.Div([
        html.Div([
        html.Label(children=['Zipcode: '], style=blackbold),
        dcc.Checklist(id='zipcode',
            options=[{'label': str(b), 'value': b} for b in df['address.postal_code'].unique()],
            value=[b for b in df['address.postal_code'].unique()]
        )
        ]),
        html.Div([
            # Map
            dcc.Graph(id='map', config={'displayModeBar': False, 'scrollZoom': True})
        ])
    ])
])

# Map callback
@app.callback(
    Output('map', 'figure'),
    [Input('zipcode', 'value')]
)
def update_map(zipcode_value):
    print(zipcode_value)
    print(type(zipcode_value))
    # df = restaurants_by_zip(zipcode_value, 25, True)
    
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

app.run_server(debug=True, use_reloader=True)