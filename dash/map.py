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
mapbox_access_token = open(".mapbox_token").read()
restaurants = df.restaurant_name.tolist()
lat = df['geo.lat'].tolist()
lon = df['geo.lon'].tolist()
app = dash.Dash(__name__)
blackbold = {'color': 'black', 'font-weight': 'bold'}

#%%
fig = go.Figure(go.Scattermapbox(
    lat = lat,
    lon = lon,
    mode='markers',
    marker={
        'color': 'red'
    },
    unselected={'marker': {'opacity': 1}},
    selected={'marker': {'opacity': 0.5, 'size': 25}},
    text=restaurants
))

fig.update_layout(uirevision='foo',
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
                zoom=12,
            ),
        )

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig, config={'displayModeBar': False, 'scrollZoom': True}, style={'padding-bottom':'2px','padding-left':'2px','height':'100vh'})
])
app.run_server(debug=True, use_reloader=True)

##%%
restaurants = df.restaurant_name.unique().tolist()
lat = df['geo.lat'].unique().tolist()
lon = df['geo.lon'].unique().tolist()

