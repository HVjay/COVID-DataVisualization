from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from country_utils import country_utils
import us_states
import math
import requests


def clean_df(df):
    df.rename(columns={'ObservationDate': 'Date', 'Province/State': 'Province_State',
                       'Country/Region': 'Country_Region', 'Confirmed': 'ConfirmedCases',
                       'Deaths': 'Fatalities'}, inplace=True)
    df.loc[df['Country_Region'] == 'Mainland China', 'Country_Region'] = 'China'
    # df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y')
    # df['Day'] = df.Date.dt.dayofyear
    df['cases_lag_1'] = df.groupby(['Country_Region', 'Province_State'])[
        'ConfirmedCases'].shift(1)
    df['deaths_lag_1'] = df.groupby(['Country_Region', 'Province_State'])[
        'Fatalities'].shift(1)
    df['Daily Cases'] = df['ConfirmedCases'] - df['cases_lag_1']
    df['Daily Deaths'] = df['Fatalities'] - df['deaths_lag_1']
    return df


def share_world_cases(df):
    df.ConfirmedCases = np.abs(df.ConfirmedCases)
    df_tm = df.copy()
    date = df_tm.Date.max()  # get current date
    df_tm = df_tm[df_tm['Date'] == date]
    obj = country_utils()
    df_tm.Province_State.fillna('', inplace=True)
    df_tm['continent'] = df_tm.apply(
        lambda x: obj.fetch_continent(x['Country_Region']), axis=1)
    df_tm["world"] = "World"  # in order to have a single root node
    fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region'], values='ConfirmedCases',
                     color='ConfirmedCases', hover_data=['Country_Region'],
                     color_continuous_scale='dense', title='Current share of Worldwide COVID19 Cases')
    fig.update_layout(width=700, template='seaborn')
    return fig


def compile_options(df):
    options = []
    countries = df['Country_Region'].unique()
    for country in countries:
        options.append({'label': country, 'value': country})
    return options


def world_rolling_avg(df):
    df_daily = df.copy()
    df_daily = df_daily.groupby('Date', as_index=False)[
        'ConfirmedCases', 'Fatalities', 'Daily Cases', 'Daily Deaths'].sum()
    df_daily = daily_metrics_world(df_daily)
    fig = world_graph(df_daily, 'Date', 'Daily Cases', 'Daily Deaths',
                      '<b>Worldwide: Daily Cases & Deaths</b><br>   With 7-Day Rolling averages')
    return fig


def daily_metrics_world(df):
    df.loc[0, 'Daily Cases'] = df.loc[0, 'ConfirmedCases']
    df.loc[0, 'Daily Deaths'] = df.loc[0, 'Fatalities']
    for i in range(1, len(df)):
        daily_cases = (df.loc[i, 'ConfirmedCases'] -
                       df.loc[i-1, 'ConfirmedCases'])
        daily_deaths = (df.loc[i, 'Fatalities']-df.loc[i-1, 'Fatalities'])
        df.loc[i, 'Daily Cases'] = daily_cases
        df.loc[i, 'Daily Deaths'] = daily_deaths
    df.loc[0, 'Daily Cases'] = 0
    df.loc[0, 'Daily Deaths'] = 0

    return df


def world_graph(df, x, y1, y2, title, days=7):
    colors = dict(case='#4285F4', death='#EA4335')
    df['cases_roll_avg'] = df[y1].rolling(days).mean()
    df['deaths_roll_avg'] = df[y2].rolling(days).mean()
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(name='Daily Cases',
                             x=df[x], y=df[y1], mode='lines',
                             line=dict(width=0.5, color=colors['case'])),
                  secondary_y=False)
    fig.add_trace(go.Scatter(name='Daily Deaths',
                             x=df[x], y=df[y2], mode='lines',
                             line=dict(width=0.5, color=colors['death'])),
                  secondary_y=True)
    fig.add_trace(go.Scatter(name='Cases: '+str(days)+'-Day Rolling average',
                             x=df[x], y=df['cases_roll_avg'], mode='lines',
                             line=dict(width=3, color=colors['case'])),
                  secondary_y=False)
    fig.add_trace(go.Scatter(name='Deaths: '+str(days)+'-Day rolling average',
                             x=df[x], y=df['deaths_roll_avg'], mode='lines',
                             line=dict(width=3, color=colors['death'])),
                  secondary_y=True)
    fig.update_yaxes(title_text='Cases', title_font=dict(color=colors['case']), secondary_y=False, nticks=5,
                     tickfont=dict(color=colors['case']), linewidth=2, linecolor='black', gridcolor='darkgrey',
                     zeroline=False)
    fig.update_yaxes(title_text='Deaths', title_font=dict(color=colors['death']), secondary_y=True, nticks=5,
                     tickfont=dict(color=colors['death']), linewidth=2, linecolor='black', gridcolor='darkgray',
                     zeroline=False)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=100, b=30), autosize=True, hovermode='x',
                      legend=dict(x=0.01, y=0.99, bordercolor='black', borderwidth=1, bgcolor='#EED8E4',
                                  font=dict(family='arial', size=10)),
                      xaxis=dict(mirror=True, linewidth=2,
                                 linecolor='black', gridcolor='darkgray'),
                      plot_bgcolor='rgb(255,255,255)')
    return fig


def world_map(df):
    df_map = df.copy()
    obj = country_utils()
    df_map['Date'] = df_map['Date'].astype(str)
    df_map = df_map.groupby(['Date', 'Country_Region'], as_index=False)[
        'ConfirmedCases', 'Fatalities'].sum()
    obj = country_utils()
    df_map['iso_alpha'] = df_map.apply(
        lambda x: obj.fetch_iso3(x['Country_Region']), axis=1)
    df_map['Confirmed_Cases'] = np.log(df_map.ConfirmedCases+1)
    df_map['World_Fatalities'] = np.log(df_map.Fatalities+1)

    fig_cases = px.choropleth(df_map,
                              locations="iso_alpha",
                              color="Confirmed_Cases",
                              hover_name="Country_Region",
                              hover_data=["ConfirmedCases"],
                              animation_frame="Date",
                              color_continuous_scale=px.colors.sequential.dense,
                              title='Total Confirmed Cases growth(Logarithmic Scale)')
    fig_deaths = px.choropleth(df_map,
                               locations='iso_alpha',
                               color='World_Fatalities',
                               hover_name='Country_Region',
                               hover_data=["Fatalities"],
                               animation_frame='Date',
                               color_continuous_scale=px.colors.sequential.dense,
                               title='Total Deaths growth (Logarithmic Scale)')
    return fig_cases, fig_deaths


def usa_map(df):
    df_us = df.copy()
    df_us = df_us[df_us['Country_Region'] == 'US']
    df_us['Date'] = df_us['Date'].astype(str)
    df_us['state_code'] = df_us.apply(
        lambda x: us_states.us_state_abbrev.get(x.Province_State, float('nan')), axis=1)
    df_us['log(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)
    df_us['log(Fatalities)'] = np.log(df_us.Fatalities + 1)
    fig_cases = px.choropleth(df_us,
                              locationmode="USA-states",
                              scope="usa",
                              locations="state_code",
                              color="log(ConfirmedCases)",
                              hover_name="Province_State",
                              hover_data=["ConfirmedCases"],
                              animation_frame="Date",
                              color_continuous_scale=px.colors.sequential.Darkmint,
                              title='Total Cases growth for USA(Logarithmic Scale)')
    fig_deaths = px.choropleth(df_us,
                               locationmode="USA-states",
                               scope="usa",
                               locations="state_code",
                               color="log(Fatalities)",
                               hover_name="Province_State",
                               hover_data=["ConfirmedCases"],
                               animation_frame="Date",
                               color_continuous_scale=px.colors.sequential.Darkmint,
                               title='Total Fatalities growth for USA(Logarithmic Scale)')
    fig_state_cases = px.line(df_us, x='Date', y='ConfirmedCases',
                              color='Province_State', title='COVID19 Confirmed Cases US')
    fig_state_cases.update_layout(hovermode='closest', template='seaborn', width=700, xaxis=dict(mirror=True, linewidth=2, linecolor='black', showgrid=False),
                                  yaxis=dict(mirror=True, linewidth=2, linecolor='black'))
    return fig_cases, fig_deaths, fig_state_cases


def find_minimum(np_log):
    if np_log < 1:
        return 0
    else:
        return math.log(np_log)


def canada_map(df):
    df_ca = df.copy()
    df_ca = df_ca[df_ca['Country_Region'] == 'Canada']
    df_ca['Date'] = df_ca['Date'].astype(str)
    log_Cases = np.log(df_ca.ConfirmedCases + 1)
    df_ca['log(ConfirmedCases)'] = df_ca.apply(
        lambda x: find_minimum(x.ConfirmedCases), axis=1)
    df_ca['log(Fatalities)'] = np.log(df_ca.Fatalities + 1)
    r = requests.get(
        url='https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson')
    canada_geojson = r.json()
    cad_cases = px.choropleth(df_ca,
                              geojson=canada_geojson,
                              color="log(ConfirmedCases)",
                              locations='Province_State',

                              #               hover_data=['log(ConfirmedCases)'],
                              animation_frame='Date',
                              featureidkey='properties.name',
                              color_continuous_scale=px.colors.sequential.dense,
                              range_color=(0, 10),
                              title='Confirmed cases of Canada (Log Scale)'
                              )
    cad_cases.update_geos(fitbounds="locations", visible=True)
    cad_cases.update_geos(projection_type="orthographic")
    cad_cases.update_layout(
        height=600, margin={"r": 0, "t": 30, "l": 0, "b": 30})

    cad_deaths = px.choropleth(df_ca,
                               geojson=canada_geojson,
                               color="log(Fatalities)",
                               locations='Province_State',

                               #               hover_data=['log(ConfirmedCases)'],
                               animation_frame='Date',
                               featureidkey='properties.name',
                               color_continuous_scale=px.colors.sequential.dense,
                               range_color=(0, 10),
                               title='Canada'
                               )
    cad_deaths.update_geos(fitbounds="locations", visible=True)
    cad_deaths.update_geos(projection_type="orthographic")
    cad_deaths.update_layout(
        height=600, margin={"r": 0, "t": 30, "l": 0, "b": 30})
    return cad_cases, cad_deaths


def add_daily_measures_country(df, country):
    df = df[df.Country_Region == country]
    df = df.groupby('Date', as_index=False)[
        'ConfirmedCases', 'Fatalities'].sum()
    df['Daily Cases'] = df['ConfirmedCases'] - df['ConfirmedCases'].shift(1)
    df['Daily Deaths'] = df['Fatalities'] - df['Fatalities'].shift(1)
    return df


def usa_daily_counts(df):
    df_usa = df.copy()
    df_usa = add_daily_measures_country(df_usa, 'US')
    fig = go.Figure(data=[
        go.Bar(name='Cases', x=df_usa['Date'], y=df_usa['Daily Cases']),
        go.Bar(name='Deaths', x=df_usa['Date'], y=df_usa['Daily Deaths'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='overlay',
                      title='Daily Case and Death count(USA)')
    fig.update_layout(hovermode='closest', template='seaborn', width=700, xaxis=dict(mirror=True, linewidth=2, linecolor='black', showgrid=False),
                      yaxis=dict(mirror=True, linewidth=2, linecolor='black'))
    return fig
