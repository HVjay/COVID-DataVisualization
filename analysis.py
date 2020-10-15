import pandas as pd
import numpy as np
import plotly.express as px
import pycountry
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from pandas_datareader import data as web
from datetime import datetime as dt
import utils
import pycountry_convert as pc
from country_utils import country_utils
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, suppress_callback_exceptions=True)
df = pd.read_csv('Data/train.csv')
df = utils.clean_df(df)

# df_daily = df.copy()
# df_daily = df_daily.groupby('Date', as_index=False)[
#     'ConfirmedCases', 'Fatalities', 'Daily Cases', 'Daily Deaths'].sum()
# df_daily = utils.daily_metrics_world(df_daily)
# fig = utils.world_graph(df_daily, 'Date', 'Daily Cases', 'Daily Deaths',
#                         '<b>Worldwide: Daily Cases & Deaths</b><br>   With 7-Day Rolling averages')

options = utils.compile_options(df)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div(children=[
    html.Div([
        dcc.Link('Look at world views', href='/world'),
        html.Br(),
        dcc.Link('Look at US views', href='/usa'),
        html.Br(),
        dcc.Link('Look at Canada views', href='/canada')
    ]),
    html.Br(),
    html.Div([
        dcc.Dropdown(
            id='my-dropdown',
            options=options,
            value='US'
        ),
        dcc.Graph(id='my-graph'),
    ]),
    html.Br(),
    html.Div([
        dcc.Graph(
            id='world_stats',
            figure=utils.share_world_cases(df)
        )
    ]),
])


us_layout = html.Div(children=[
    html.Div([
        dcc.Graph(
            id='usa_cases',
            figure=utils.usa_map(df)[0],
            style={'display': 'inline-block'}
        ),
        dcc.Graph(
            id='usa_deaths',
            figure=utils.usa_map(df)[1],
            style={'display': 'inline-block'}
        )
    ]),
    html.Div([
        dcc.Graph(
            id='usa_state_cases',
            figure=utils.usa_map(df)[2]
            # style={'display':'inline-b'}
        )
    ]),
    html.Div([
        dcc.Graph(
            id='usa_daily_counts',
            figure=utils.usa_daily_counts(df)
        )
    ])
])

world_layout = html.Div(children=[

    html.Div([
        dcc.Graph(
            id='world-trend',
            figure=utils.world_rolling_avg(df)
        ),
    ]),
    html.Div([
        dcc.Graph(
            id='world_map_cases',
            figure=utils.world_map(df)[0],
            style={'display': 'inline-block'}
        ),
        dcc.Graph(
            id='world_map_cases',
            figure=utils.world_map(df)[1],
            style={'display': 'inline-block'}
        )
    ]),
])

canada_layout = html.Div(children=[
    html.Div([
        dcc.Graph(
            id='Cad_Cases',
            figure=utils.canada_map(df)[0]

        )
    ])
])


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    temp = df[df['Country_Region'] == selected_dropdown_value]
    temp = temp.groupby(['Date']).sum()
    x = temp.index
    y = temp.ConfirmedCases
    # temp_df=pd.DataFrame(Date,ConfirmedCases)

    # fig=px.line(temp_df,x=temp_df.Date,y=temp_df.ConfirmedCases)
    return {
        # fig
        'data': [{
            'x': x,
            'y': y,
        }],
        'layout': {
            'title': 'Confirmed Cases ',
            'xaxis': {
                'title': 'Date'
            },
            'yaxis': {
                'title': 'Confirmed Cases'
            },
            'margin': {'l': 60, 'r': 0, 't': 40, 'b': 30}
        }
        # 'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
    }


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/world':
        return world_layout
    elif pathname == '/usa':
        return us_layout
    elif pathname == '/canada':
        return canada_layout
    else:
        return index_page


app.css.append_css(
    {'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
