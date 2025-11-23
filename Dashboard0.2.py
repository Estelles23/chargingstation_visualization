import dash
from dash import dcc, html, Input, Output, State, ctx, callback
import plotly.express as px
import pandas as pd

#Options config
pd.set_option("display.max_columns", None)

#Read the dataset
df = pd.read_csv("Session_data.csv")

#Function for correcting data types
def correct_data_types(df):

    #Parsing the date as a timestamp
    df["Arrival"] = pd.to_datetime(df["Arrival"])
    df["Departure"] = pd.to_datetime(df["Departure"])

    return df

def create_derived_columns(df):

    def derived_columns_graph_1(df):

        df['Date'] = df['Arrival'].dt.date
        df['Hour'] = df['Arrival'].dt.hour

        #Day of week (0=Monday, 6=Sunday)
        df['DayOfWeek'] = df['Arrival'].dt.dayofweek

        #Week number and year
        df['Week'] = df['Arrival'].dt.isocalendar().week
        df['Year'] = df['Arrival'].dt.year

        #Extract month and year
        df['Month'] = df['Arrival'].dt.month

        #For calculating aggregates
        df['YearMonth_Period'] = df['Arrival'].dt.to_period('M')
        #For Plotly
        df['YearMonth'] = df['Arrival'].dt.to_period('M').astype(str)


        # Convert Wh to kWh and calculate revenue (prices for Switzerland)
        df['Energy_kWh'] = df['Energy (Wh)'] / 1000
        df['Revenue_EUR'] = df['Energy_kWh'] * 0.50  # 0.50 EUR per kWh
        return df

    df = derived_columns_graph_1(df)

    return df

def create_daily_aggregate_df(df):

    #Group by Date
    daily_agg = df.groupby('Date').agg({
        'Energy_kWh': 'sum',  #Total kWh sold
        'Session': 'count',  #Number of sessions
        'Revenue_EUR': 'mean'  #Average revenue per session
    }).reset_index()

    daily_agg.columns = ['Date', 'Total_kWh', 'Num_Sessions', 'Avg_Revenue_per_Session']

    return daily_agg

def create_weekly_aggregate_df(df):

    weekly_agg = df.groupby(['Year', 'Week']).agg({
        'Energy_kWh': 'sum',
        'Session': 'count',
        'Revenue_EUR': 'mean'
    }).reset_index()

    weekly_agg.columns = ['Year', 'Week', 'Total_kWh', 'Num_Sessions', 'Avg_Revenue_per_Session']

    #Finding the monday of the week and creating a datestamp (we need to do this for plotting later)
    weekly_agg['Date'] = pd.to_datetime(
        weekly_agg['Year'].astype(str) + '-W' + weekly_agg['Week'].astype(str) + '-1',
        format='%Y-W%W-%w'
    )

    return weekly_agg

def create_monthly_aggregate_df(df):
    monthly_agg = df.groupby('YearMonth_Period').agg({  # Use YearMonth_Period here
        'Energy_kWh': 'sum',
        'Session': 'count',
        'Revenue_EUR': 'mean'
    }).reset_index()

    monthly_agg.columns = ['YearMonth_Period', 'Total_kWh', 'Num_Sessions', 'Avg_Revenue_per_Session']

    # Selecting the first day of the month and creating the column
    monthly_agg['Date'] = monthly_agg['YearMonth_Period'].dt.to_timestamp()

    return monthly_agg

def create_hourly_sessions(df):
    hourly_sessions = df.groupby('Hour').size().reset_index(name='Session_Count')

    return hourly_sessions

def create_weekday_sessions(df):
    weekday_sessions = df.groupby('DayOfWeek').size().reset_index(name='Session_Count')

    return weekday_sessions

def create_monthly_sessions(df):
    monthly_sessions = df.groupby('YearMonth').size().reset_index(name='Session_Count')

    return monthly_sessions

#Cleaning the data
df = correct_data_types(df)
df = create_derived_columns(df)

#Auxiliar dfs
daily_agg = create_daily_aggregate_df(df)
weekly_agg = create_weekly_aggregate_df(df)
monthly_agg = create_monthly_aggregate_df(df)
hourly_sessions = create_hourly_sessions(df)
weekday_sessions = create_weekday_sessions(df)
monthly_sessions = create_monthly_sessions(df)

print("*" * 80)
print("*" * 80)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Revenue & Utilization Metrics"),

    html.Label("Time Period:"),
    dcc.Dropdown(
        id='time-period-dropdown',
        options=[
            {'label': 'Daily', 'value': 'daily'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'}
        ],
        value='daily'
    ),

    html.Label("Metric:"),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Total kWh Sold', 'value': 'Total_kWh'},
            {'label': 'Number of Sessions', 'value': 'Num_Sessions'},
            {'label': 'Average Revenue per Session', 'value': 'Avg_Revenue_per_Session'}
        ],
        value='Total_kWh'
    ),

    dcc.Graph(id='revenue-graph'),

    # NEW SECTION for Utilization
    html.H2("Utilization Rate by Time of Day"),

    html.Label("View By:"),
    dcc.Dropdown(
        id='utilization-view-dropdown',
        options=[
            {'label': 'Hour of Day', 'value': 'hour'},
            {'label': 'Day of Week', 'value': 'weekday'},
            {'label': 'Month', 'value': 'month'}
        ],
        value='hour'
    ),

    dcc.Graph(id='hourly-utilization-graph')
])


@app.callback(
    Output('hourly-utilization-graph', 'figure'),
    Input('utilization-view-dropdown', 'value')
)
def update_utilization_graph(view_type):
    # Select which data to use based on dropdown
    if view_type == 'hour':
        data = hourly_sessions
        x_col = 'Hour'
        title = 'Number of Sessions by Hour of Day'
        x_label = 'Hour of Day (0-23)'
    elif view_type == 'weekday':
        data = weekday_sessions
        x_col = 'DayOfWeek'
        title = 'Number of Sessions by Day of Week'
        x_label = 'Day of Week (0=Monday, 6=Sunday)'
    else:  # month
        data = monthly_sessions
        x_col = 'YearMonth'
        title = 'Number of Sessions by Month'
        x_label = 'Month'

    # Create the bar chart
    fig = px.bar(data, x=x_col, y='Session_Count',
                 title=title,
                 labels={x_col: x_label, 'Session_Count': 'Total Sessions'})
    if view_type == "month":
        fig.update_xaxes(
            tickmode='array',
            tickvals=data['YearMonth'],  # Use all values from your data
            tickangle=-45
        )
    return fig

@app.callback(
    Output('revenue-graph', 'figure'),
    [Input('time-period-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)

def update_revenue_graph(time_period, metric):
    #dataframe selection
    if time_period == 'daily':
        data = daily_agg
    elif time_period == 'weekly':
        data = weekly_agg
    else:  #monthly
        data = monthly_agg

    #Linechart
    fig = px.line(data, x='Date', y=metric,
                  title=f'{metric.replace("_", " ")} - {time_period.capitalize()}')

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8004)
