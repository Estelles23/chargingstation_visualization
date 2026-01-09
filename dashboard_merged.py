import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression


################################################################################################
# DATA PREPROCESSING FUNCTIONS
################################################################################################

def correct_data_types(df):
    """Parse dates and add derived columns for analysis."""
    # Parsing the date as a timestamp
    df["Arrival"] = pd.to_datetime(df["Arrival"], errors="coerce")
    df["Departure"] = pd.to_datetime(df["Departure"], errors="coerce")

    # RQ1: Revenue and temporal columns
    df["Energy_capacity_kWh"] = df["Energy capacity (Wh)"] / 1000
    df["Revenue"] = df["Energy_capacity_kWh"] * 0.15  # Assuming $0.15 per kWh
    df["Hour"] = df["Arrival"].dt.hour
    df["dayoftheweek"] = df["Arrival"].dt.day_name()
    df["only_date"] = df["Arrival"].dt.date

    # RQ2: Derived columns for customer behavior analysis
    df["Energy_kWh"] = df["Energy (Wh)"] / 1000
    df["AvgPower_kW"] = df["Energy_kWh"] / (df["Stay (min)"] / 60)
    df["SOC_Gained"] = df["SOC departure"] - df["SOC arrival"]
    df["Controlled_Cat"] = df["Controlled session (0=False, 1=True)"].map({0: "Not Controlled", 1: "Controlled"})
    df["Arrival_Hour"] = df["Arrival"].dt.hour
    df["Arrival_TimePeriod"] = df["Arrival_Hour"].apply(time_period)

    return df


def time_period(hour):
    """Categorize hour into time period of day."""
    if pd.isna(hour):
        return "Unknown"
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 23:
        return "Evening"
    else:
        return "Night"


################################################################################################
# RQ1: REVENUE & UTILIZATION FUNCTIONS
################################################################################################

def create_revenue_figures(df):
    """Create all revenue and utilization figures for RQ1."""

    # Define consistent color scheme (matching RQ4)
    COLOR_PRIMARY = '#074051'  # Dark teal (controlled)
    COLOR_SECONDARY = '#D3F2A4'  # Light green (not controlled)
    COLOR_ACCENT = '#0C4A59'  # Medium teal
    COLOR_NEUTRAL = '#808080'  # Gray

    # Hourly summary
    hourly_summary = df.groupby('Hour').agg({
        'Session': 'count',
        'Energy_capacity_kWh': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    # Weekday usage
    weekday_usage = df.groupby('dayoftheweek').agg({'Session': 'count'}).reset_index()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_usage['dayoftheweek'] = pd.Categorical(
        weekday_usage['dayoftheweek'],
        categories=weekdays,
        ordered=True
    )
    weekday_usage = weekday_usage.sort_values('dayoftheweek')

    # Date summary with trend line
    date_summary = df.groupby('only_date').agg({
        'Session': 'count',
        'Revenue': 'sum'
    }).reset_index()
    date_summary['time_index'] = date_summary.index

    # Linear regression for trend
    X = date_summary[['time_index']].values.reshape(-1, 1)
    y = date_summary['Revenue']
    model = LinearRegression()
    model.fit(X, y)
    date_summary['Predicted_Revenue'] = model.predict(X)

    # Peak vs Off-Peak summary
    hourly_summary['period'] = hourly_summary['Hour'].apply(
        lambda h: 'Peak' if 8 <= h <= 18 else 'Off-Peak'
    )
    peak_offpeak_summary = hourly_summary.groupby('period').agg({
        'Session': 'sum',
        'Revenue': 'sum',
        'Energy_capacity_kWh': 'sum'
    }).reset_index()

    # Idle time analysis
    df_idle = df.copy()
    df_idle = df_idle.sort_values('Arrival').reset_index(drop=True)
    df_idle['End'] = df_idle['Arrival'] + pd.Timedelta(minutes=45)  # Assumption: 45 min avg session
    df_idle['Next_Arrival'] = df_idle['Arrival'].shift(-1)
    df_idle['Idle_Time'] = df_idle['Next_Arrival'] - df_idle['End']
    df_idle['Idle_Minutes'] = df_idle['Idle_Time'].dt.total_seconds() / 60
    df_idle = df_idle[df_idle['Idle_Minutes'] > 0]
    df_idle['Peak_Type'] = df_idle['Hour'].apply(
        lambda h: 'Peak' if 8 <= h <= 18 else 'Off-Peak'
    )

    idle_by_hour = df_idle.groupby('Hour')['Idle_Minutes'].sum().reset_index()
    idle_peak_summary = df_idle.groupby('Peak_Type')['Idle_Minutes'].sum().reset_index()

    # Create figures with consistent styling
    fig_hour_sessions = go.Figure(
        go.Scatter(
            x=hourly_summary['Hour'],
            y=hourly_summary['Session'],
            mode='lines+markers',
            line=dict(color="#5daf87", width=3),
            marker=dict(size=6, color="#5daf87")
        )
    )
    fig_hour_sessions.update_layout(
        title={'text': "Sessions per Hour", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Hour",
        yaxis_title="Number of Sessions",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    fig_hour_energy = go.Figure(
        go.Scatter(
            x=hourly_summary['Hour'],
            y=hourly_summary['Energy_capacity_kWh'],
            mode='lines+markers',
            line=dict(color="#5daf87", width=3),
            marker=dict(size=6, color="#5daf87")
        )
    )
    fig_hour_energy.update_layout(
        title={'text': "Energy Consumption per Hour", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Hour",
        yaxis_title="Energy (kWh)",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    fig_hour_revenue = go.Figure(
        go.Scatter(
            x=hourly_summary['Hour'],
            y=hourly_summary['Revenue'],
            mode='lines+markers',
            line=dict(color="#5daf87", width=3),
            marker=dict(size=6, color="#5daf87")
        )
    )
    fig_hour_revenue.update_layout(
        title={'text': "Revenue per Hour", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Hour",
        yaxis_title="Revenue ($)",
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    fig_weekday_sessions = go.Figure(
        go.Bar(
            x=weekday_usage['dayoftheweek'],
            y=weekday_usage['Session'],
            marker_color="#084051"
        )
    )
    fig_weekday_sessions.update_layout(
        title={'text': "Sessions per Weekday", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Day of the Week",
        yaxis_title="Number of Sessions"
    )

    fig_date_revenue = go.Figure()
    fig_date_revenue.add_trace(
        go.Scatter(
            x=date_summary['only_date'],
            y=date_summary['Revenue'],
            name='Daily Revenue',
            line=dict(color="#084051", width=2),
            marker=dict(color="#e53b51")
        )
    )
    fig_date_revenue.add_trace(
        go.Scatter(
            x=date_summary['only_date'],
            y=date_summary['Predicted_Revenue'],
            mode='lines',
            name='Long-term Trend',
            line=dict(dash='dash', color="#e53b51", width=3)
        )
    )
    fig_date_revenue.update_layout(
        title={'text': "Revenue per Date with Long-term Trend", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    fig_peak_offpeak = go.Figure()
    fig_peak_offpeak.add_trace(
        go.Bar(x=peak_offpeak_summary['period'], y=peak_offpeak_summary['Revenue'],
               name='Revenue ($)', marker_color="#5daf87")
    )
    fig_peak_offpeak.add_trace(
        go.Bar(x=peak_offpeak_summary['period'], y=peak_offpeak_summary['Session'],
               name='Sessions', marker_color=COLOR_SECONDARY)
    )
    fig_peak_offpeak.add_trace(
        go.Bar(x=peak_offpeak_summary['period'], y=peak_offpeak_summary['Energy_capacity_kWh'],
               name='Energy (kWh)', marker_color=COLOR_ACCENT)
    )
    fig_peak_offpeak.update_layout(
        title={'text': "Peak vs Off-Peak Comparison", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Period",
        yaxis_title="Value",
        barmode="group",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    fig_idle_hour = go.Figure(
        go.Bar(x=idle_by_hour['Hour'], y=idle_by_hour['Idle_Minutes'], marker_color=COLOR_PRIMARY)
    )
    fig_idle_hour.update_layout(
        title={'text': 'Idle Time by Hour', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Hour of Day',
        yaxis_title='Idle Minutes',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    fig_idle_peak = go.Figure(
        go.Bar(x=idle_peak_summary['Peak_Type'], y=idle_peak_summary['Idle_Minutes'],
               marker_color=["#5daf87", COLOR_PRIMARY])
    )
    fig_idle_peak.update_layout(
        title={'text': 'Idle Time: Peak vs Off-Peak', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Period',
        yaxis_title='Idle Minutes'
    )

    return {
        'hour_sessions': fig_hour_sessions,
        'hour_energy': fig_hour_energy,
        'hour_revenue': fig_hour_revenue,
        'weekday_sessions': fig_weekday_sessions,
        'date_revenue': fig_date_revenue,
        'peak_offpeak': fig_peak_offpeak,
        'idle_hour': fig_idle_hour,
        'idle_peak': fig_idle_peak
    }


################################################################################################
# RQ3: OPERATIONAL EFFICIENCY FUNCTIONS
################################################################################################

def create_rq3_figures(session_df, measurement_ccs1_path, measurement_ccs2_path):
    """Create all operational efficiency figures for RQ3."""

    # Define consistent color scheme (matching RQ4)
    COLOR_PRIMARY = '#074051'  # Dark teal (controlled)
    COLOR_SECONDARY = '#5daf87'  # Light green (not controlled)
    COLOR_ACCENT = '#0C4A59'  # Medium teal
    COLOR_CCS1 = '#074051'  # Dark teal for CCS1
    COLOR_CCS2 = '#D3F2A4'  # Light green for CCS2

    # Load and merge measurement data
    ccs1 = pd.read_csv(measurement_ccs1_path)
    ccs2 = pd.read_csv(measurement_ccs2_path)
    measurement_data = pd.merge(ccs1, ccs2, on='Date and time')
    measurement_data.columns = [col.replace('_x', '_1').replace('_y', '_2') for col in measurement_data.columns]

    # --- Line Chart: Max Power vs Charging Time ---
    data_line = session_df.loc[:, ['Stay (min)', 'Pmax (W)']].drop_duplicates()
    fig_line = px.line(
        data_line.groupby("Stay (min)", as_index=False)["Pmax (W)"].sum(),
        x="Stay (min)",
        y="Pmax (W)",
        labels={
            "Stay (min)": "Charging Time (min)",
            "Pmax (W)": "Maximum Power (W)"
        }
    )
    fig_line.update_traces(line=dict(color='#5daf87', width=3))
    fig_line.update_layout(
        title={'text': "Maximum Power vs Charging Time", 'x': 0.5, 'xanchor': 'center'}
    )

    # --- Bar Chart: Controlled Sessions Percentage ---
    data_bar = session_df[["Controlled session (0=False, 1=True)"]]
    data_bar = (
        data_bar
        .value_counts(normalize=True)
        .reset_index(name="Percentage")
    )
    data_bar["Percentage"] = data_bar["Percentage"] * 100

    fig_bar = px.bar(
        data_bar,
        x="Controlled session (0=False, 1=True)",
        y="Percentage",
        text="Percentage"
    )
    fig_bar.update_traces(marker_color=[COLOR_SECONDARY, COLOR_PRIMARY])
    fig_bar.update_yaxes(title_text=None)
    fig_bar.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        textfont=dict(color='white', size=18)
    )
    fig_bar.update_layout(
        title={'text': "Controlled Sessions Percentage", 'x': 0.5, 'xanchor': 'center'}
    )

    # --- Scatter Plot: Preq vs Pset ---
    data_scatter = measurement_data.loc[:, [
        'Date and time', "Preq (W)_1", "Preq (W)_2",
        "Pset (W)_1", "Pset (W)_2", "Session_1", "Session_2"
    ]]

    # Compute CCS Count (how many sessions exist)
    data_scatter["CCS Count"] = data_scatter[["Session_1", "Session_2"]].notna().sum(axis=1)
    data_scatter = data_scatter[data_scatter["CCS Count"] != 0]

    # Helper function for conditional averaging
    def percentage(df, col1, col2):
        column1 = df[col1].fillna(0)
        column2 = df[col2].fillna(0)
        return np.where(
            df["CCS Count"] == 2,
            (column1 + column2) / 2,
            column1 + column2
        )

    data_scatter["Preq tot"] = percentage(data_scatter, "Preq (W)_1", "Preq (W)_2")
    data_scatter["Pset tot"] = percentage(data_scatter, "Pset (W)_1", "Pset (W)_2")
    data_scatter["CCS Count"] = data_scatter["CCS Count"].astype("category")

    fig_scatter = px.scatter(
        data_scatter,
        x="Preq tot",
        y="Pset tot",
        color="CCS Count",
        color_discrete_map={1: COLOR_SECONDARY, 2: COLOR_PRIMARY},
        labels={
            "Preq tot": "Total Requested Power (W)",
            "Pset tot": "Total Set Power (W)",
            "CCS Count": "Number of Charging Stations"
        }
    )
    fig_scatter.update_layout(legend_title_text='')
    fig_scatter.for_each_trace(lambda t: t.update(name="1 Car" if t.name == "1" else "2 Cars"))
    fig_scatter.update_layout(
        title={'text': "Requested vs Set Power by Number of Cars", 'x': 0.5, 'xanchor': 'center'},
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    # --- Timeline: Session Timeline by CCS ---
    data_timeline = session_df.loc[:, ['CCS', 'Arrival', 'Departure']].copy()
    data_timeline['Arrival'] = pd.to_datetime(data_timeline['Arrival'], errors='coerce')
    data_timeline['Departure'] = pd.to_datetime(data_timeline['Departure'], errors='coerce')

    fig_timeline = px.timeline(
        data_timeline,
        x_start="Arrival",
        x_end="Departure",
        y="CCS",
        color="CCS",
        color_discrete_map={"CCS1": COLOR_PRIMARY, "CCS2": COLOR_SECONDARY}
    )
    fig_timeline.update_yaxes(autorange="reversed", title_text=None)
    fig_timeline.update_xaxes(
        rangeslider_visible=True,
        range=[pd.Timestamp('2023-02-26'), pd.Timestamp('2023-02-28')],
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="daily", step="day", stepmode="backward"),
                dict(count=7, label="weekly", step="day", stepmode="backward"),
                dict(count=1, label="monthly", step="month", stepmode="backward"),
                dict(count=1, label="yearly", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig_timeline.update_layout(
        title={'text': "Timeline by Charging Station", 'x': 0.5, 'xanchor': 'center'},
        yaxis=dict(autorange="reversed"),
        bargap=0.05,
        height=300,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )

    return {
        'timeline': fig_timeline,
        'bar_controlled': fig_bar,
        'scatter_preq_pset': fig_scatter,
        'line_power_time': fig_line
    }


################################################################################################
# RQ1: HEATMAP FUNCTIONS
################################################################################################

def get_session_matrix(df):
    def create_heatmap_dataframe(df):
        # Creating a first dataframe grouping by time and day.
        df['hour'] = df['Arrival'].dt.hour
        df['day_of_week'] = df['Arrival'].dt.day_name()
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='session_count')

        # Empty grid with all the hours
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        all_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        all_combinations = []

        for day in all_days:
            for hour in all_hours:
                all_combinations.append({'day_of_week': day, 'hour': hour})

        complete_grid = pd.DataFrame(all_combinations)

        heatmap_data_complete = complete_grid.merge(
            heatmap_data,
            on=['day_of_week', 'hour'],
            how='left'
        )

        # Replacing NaN with 0 (there were no sessions on that time slot) and converting to integers
        heatmap_data_complete['session_count'] = heatmap_data_complete['session_count'].fillna(0)
        heatmap_data_complete['session_count'] = heatmap_data_complete['session_count'].astype(int)

        return heatmap_data_complete

    def create_heatmap_matrix(df):
        heatmap_matrix = df.pivot(
            index='day_of_week',
            columns='hour',
            values='session_count'
        )
        day_order = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
        heatmap_matrix = heatmap_matrix.reindex(day_order)
        return heatmap_matrix

    # Copy dataframe for not modifying the global
    df_fig_1 = df.copy()
    # Create dataframe with usage per day
    heatmap_data_complete = create_heatmap_dataframe(df_fig_1)
    # Pivot dataframe into a matrix
    heatmap_matrix = create_heatmap_matrix(heatmap_data_complete)

    return heatmap_matrix


def get_overlap_matrix(df):
    def create_overlap_dataframe(df):
        sessions = df
        sessions_sorted = sessions.sort_values('Arrival').reset_index(drop=True)
        overlaps = []

        for i in range(len(sessions_sorted)):
            current_session = sessions_sorted.iloc[i]
            for j in range(i + 1, len(sessions_sorted)):
                next_session = sessions_sorted.iloc[j]
                if next_session['Arrival'] >= current_session['Departure']:
                    break
                if current_session['CCS'] != next_session['CCS']:
                    overlap_start = max(current_session['Arrival'], next_session['Arrival'])
                    overlap_end = min(current_session['Departure'], next_session['Departure'])
                    overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60

                    overlaps.append({
                        'session_1': current_session['Session'],
                        'session_2': next_session['Session'],
                        'ccs_1': current_session['CCS'],
                        'ccs_2': next_session['CCS'],
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_minutes': overlap_minutes
                    })
        overlaps_df = pd.DataFrame(overlaps)
        return overlaps_df

    def create_overlap_heatmap_dataframe(overlaps_df):
        # Extract hour and day from overlap_start
        overlaps_df['hour'] = overlaps_df['overlap_start'].dt.hour
        overlaps_df['day_of_week'] = overlaps_df['overlap_start'].dt.day_name()

        # Group by day and hour, count overlaps
        heatmap_data = overlaps_df.groupby(['day_of_week', 'hour']).size().reset_index(name='overlap_count')

        # Create empty grid with all hours
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        all_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        all_combinations = []

        for day in all_days:
            for hour in all_hours:
                all_combinations.append({'day_of_week': day, 'hour': hour})

        complete_grid = pd.DataFrame(all_combinations)

        # Merge with actual overlap data
        heatmap_data_complete = complete_grid.merge(
            heatmap_data,
            on=['day_of_week', 'hour'],
            how='left'
        )

        # Replace NaN with 0 and convert to integers
        heatmap_data_complete['overlap_count'] = heatmap_data_complete['overlap_count'].fillna(0)
        heatmap_data_complete['overlap_count'] = heatmap_data_complete['overlap_count'].astype(int)

        return heatmap_data_complete

    def create_overlap_heatmap_matrix(df):
        heatmap_matrix = df.pivot(
            index='day_of_week',
            columns='hour',
            values='overlap_count'
        )
        # Day order for Y axis
        day_order = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']
        heatmap_matrix = heatmap_matrix.reindex(day_order)
        return heatmap_matrix

    # Create the overlap heatmap
    overlaps_df = create_overlap_dataframe(df)
    overlap_heatmap_data = create_overlap_heatmap_dataframe(overlaps_df)
    overlap_heatmap_matrix = create_overlap_heatmap_matrix(overlap_heatmap_data)

    return overlap_heatmap_matrix


################################################################################################
# RQ4: INFRASTRUCTURE CAPACITY FUNCTIONS
################################################################################################

def controlled_breakdown(df):
    def get_concurrent_session_ids(df):
        sessions_sorted = df.sort_values('Arrival').reset_index(drop=True)
        concurrent_sessions = set()

        for i in range(len(sessions_sorted)):
            current_session = sessions_sorted.iloc[i]
            for j in range(i + 1, len(sessions_sorted)):
                next_session = sessions_sorted.iloc[j]
                if next_session['Arrival'] >= current_session['Departure']:
                    break
                if current_session['CCS'] != next_session['CCS']:
                    concurrent_sessions.add(int(current_session['Session']))
                    concurrent_sessions.add(int(next_session['Session']))

        return list(concurrent_sessions)

    def get_controlled_breakdown(df):
        controlled_sessions = df[df['Controlled session (0=False, 1=True)'] == True]
        total_controlled = len(controlled_sessions)

        total_not_controlled = 1878 - total_controlled
        concurrent_session_ids = get_concurrent_session_ids(df)
        controlled_alone = controlled_sessions[~controlled_sessions['Session'].isin(concurrent_session_ids)]
        total_controlled_alone = len(controlled_alone)

        total_controlled_concurrent = total_controlled - total_controlled_alone

        concurrent_sessions_df = controlled_sessions[controlled_sessions['Session'].isin(concurrent_session_ids)]
        not_controlled_sessions = df[df['Controlled session (0=False, 1=True)'] == False]
        total_not_controlled_concurrent = len(
            not_controlled_sessions[not_controlled_sessions['Session'].isin(concurrent_session_ids)])
        total_not_controlled_alone = total_not_controlled - total_not_controlled_concurrent

        return total_controlled, total_not_controlled, total_controlled_alone, total_controlled_concurrent, total_not_controlled_concurrent, total_not_controlled_alone

    def create_figure_3(df):
        (total_controlled, total_not_controlled, total_controlled_alone, total_controlled_concurrent,
         total_not_controlled_concurrent, total_not_controlled_alone) = get_controlled_breakdown(df)

        fig3 = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[
                    "All Sessions<br>(1878)",
                    "Controlled<br>(1416)",
                    "Not Controlled<br>(462)",
                    "Alone<br>(868)",
                    "Concurrent<br>(548)",
                    "Alone<br>(328)",
                    "Concurrent<br>(134)"
                ],
                color=[
                    "#808080",
                    "rgba(7, 64, 81, 1)",
                    "rgba(211, 242, 164, 1)",
                    "rgba(7, 64, 81, 1)",
                    "rgba(7, 64, 81, 1)",
                    "rgba(211, 242, 164, 1)",
                    "rgba(211, 242, 164, 1)"
                ],
                x=[0.01, 0.4, 0.4, 0.99, 0.99, 0.99, 0.99],
                y=[0.5, 0.2, 0.8, 0.05, 0.35, 0.65, 0.95]
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2],
                target=[1, 2, 3, 4, 5, 6],
                value=[
                    total_controlled,
                    total_not_controlled,
                    total_controlled_alone,
                    total_controlled_concurrent,
                    total_not_controlled_alone,
                    total_not_controlled_concurrent
                ],
                color=[
                    "rgba(7, 64, 81, 0.8)",
                    "rgba(211, 242, 164, 0.8)",
                    "rgba(7, 64, 81, 0.7)",
                    "rgba(7, 64, 81, 0.7)",
                    "rgba(211, 242, 164, 0.7)",
                    "rgba(211, 242, 164, 0.7)"
                ]
            )
        )])

        fig3.update_layout(
            title={
                'text': "Session Control Flow Analysis<br><sub>Infrastructure Constraint Assessment</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(size=12),
            height=600,
            width=1000
        )
        return fig3

    fig3 = create_figure_3(df)
    return fig3


def power_ampliation(df, measurements):
    def get_concurrent_session_ids(df):
        sessions_sorted = df.sort_values('Arrival').reset_index(drop=True)
        concurrent_sessions = set()

        for i in range(len(sessions_sorted)):
            current_session = sessions_sorted.iloc[i]
            for j in range(i + 1, len(sessions_sorted)):
                next_session = sessions_sorted.iloc[j]
                if next_session['Arrival'] >= current_session['Departure']:
                    break
                if current_session['CCS'] != next_session['CCS']:
                    concurrent_sessions.add(int(current_session['Session']))
                    concurrent_sessions.add(int(next_session['Session']))

        return list(concurrent_sessions)

    def categorize_sessions(df_sessions):
        """Categorize sessions into four types."""
        concurrent_session_ids = set(get_concurrent_session_ids(df_sessions))
        categories = {}

        for _, row in df_sessions.iterrows():
            session_id = int(row['Session'])
            is_controlled = row['Controlled session (0=False, 1=True)'] == True
            is_concurrent = session_id in concurrent_session_ids

            if is_controlled and not is_concurrent:
                categories[session_id] = 'Controlled Alone'
            elif is_controlled and is_concurrent:
                categories[session_id] = 'Controlled Concurrent'
            elif not is_controlled and not is_concurrent:
                categories[session_id] = 'Not Controlled Alone'
            else:
                categories[session_id] = 'Not Controlled Concurrent'

        return categories

    def calculate_mean_preq_by_minute(df_measurements, session_categories, max_minutes=60):
        """Calculate mean Preq for each minute of the session, grouped by session category."""
        df_measurements['Category'] = df_measurements['Session'].map(session_categories)
        df_with_category = df_measurements[df_measurements['Category'].notna()].copy()

        # Calculate minute of session for each measurement
        first_times = df_with_category.groupby('Session')['Date and time'].min().to_dict()
        df_with_category['Minute'] = df_with_category.apply(
            lambda row: int((row['Date and time'] - first_times[row['Session']]).total_seconds() / 60),
            axis=1
        )

        # Filter to max_minutes
        df_with_category = df_with_category[df_with_category['Minute'] <= max_minutes]

        # Calculate mean Preq by minute and category
        grouped = df_with_category.groupby(['Category', 'Minute'])['Preq (W)'].agg(
            ['mean', 'std', 'count']).reset_index()
        grouped['mean_kW'] = grouped['mean'] / 1000
        grouped['std_kW'] = grouped['std'] / 1000

        return grouped

    def create_graph_not_controlled_by_minute(grouped_data):
        """Graph: Not Controlled Alone + Not Controlled Concurrent (by minute)"""
        fig4 = go.Figure()

        colors = {
            'Not Controlled Alone': '#D3F2A4',
            'Not Controlled Concurrent': '#A3E499'
        }

        for category in ['Not Controlled Alone', 'Not Controlled Concurrent']:
            data = grouped_data[grouped_data['Category'] == category].sort_values('Minute')

            if len(data) == 0:
                continue

            fig4.add_trace(go.Scatter(
                x=data['Minute'],
                y=data['mean_kW'],
                mode='lines+markers',
                name=category,
                line=dict(color=colors[category], width=3),
                marker=dict(size=6),
                hovertemplate=(
                        '<b>%{fullData.name}</b><br>' +
                        'Minute: %{x}<br>' +
                        'Mean Preq: %{y:.1f} kW<br>' +
                        '<extra></extra>'
                )
            ))

            fig4.add_trace(go.Scatter(
                x=data['Minute'].tolist() + data['Minute'].tolist()[::-1],
                y=(data['mean_kW'] + data['std_kW']).tolist() +
                  (data['mean_kW'] - data['std_kW']).tolist()[::-1],
                fill='toself',
                fillcolor=colors[category],
                opacity=0.1,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig4.add_hline(y=172.5, line_dash="dash", line_color="black", line_width=2,
                       annotation_text="Station Max: 172.5 kW", annotation_position="right")

        fig4.update_layout(
            title={
                'text': 'Not Controlled Sessions: Mean Power Requested by Minute<br><sub>Comparing alone vs concurrent charging over time</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Minutes from Session Start",
            yaxis_title="Mean Power Requested (kW)",
            xaxis=dict(range=[0, 60]),
            yaxis=dict(range=[0, 370]),
            height=600,
            autosize=True,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )

        return fig4

    def create_graph_controlled_alone_by_minute(grouped_data):
        """Graph: Controlled Alone only (by minute)"""
        fig5 = go.Figure()
        data = grouped_data[grouped_data['Category'] == 'Controlled Alone'].sort_values('Minute')

        if len(data) > 0:
            fig5.add_trace(go.Scatter(
                x=data['Minute'],
                y=data['mean_kW'],
                mode='lines+markers',
                name='Controlled Alone',
                line=dict(color='#0C4A59', width=3),
                marker=dict(size=6),
                hovertemplate=(
                        '<b>Controlled Alone</b><br>' +
                        'Minute: %{x}<br>' +
                        'Mean Preq: %{y:.1f} kW<br>' +
                        '<extra></extra>'
                )
            ))

            fig5.add_trace(go.Scatter(
                x=data['Minute'].tolist() + data['Minute'].tolist()[::-1],
                y=(data['mean_kW'] + data['std_kW']).tolist() +
                  (data['mean_kW'] - data['std_kW']).tolist()[::-1],
                fill='toself',
                fillcolor='#0C4A59',
                opacity=0.1,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig5.add_hline(y=172.5, line_dash="dash", line_color="black", line_width=2,
                       annotation_text="Station Max: 172.5 kW", annotation_position="right")

        fig5.update_layout(
            title={
                'text': 'Controlled Alone Sessions: Mean Power Requested by Minute<br><sub>Power-limited despite charging alone over time</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Minutes from Session Start",
            yaxis_title="Mean Power Requested (kW)",
            xaxis=dict(range=[0, 60]),
            yaxis=dict(range=[0, 370]),
            height=600,
            autosize=True,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )

        return fig5

    def create_graph_controlled_concurrent_by_minute(grouped_data):
        """Graph: Controlled Concurrent only (by minute)"""
        fig6 = go.Figure()
        data = grouped_data[grouped_data['Category'] == 'Controlled Concurrent'].sort_values('Minute')

        if len(data) > 0:
            fig6.add_trace(go.Scatter(
                x=data['Minute'],
                y=data['mean_kW'],
                mode='lines+markers',
                name='Controlled Concurrent',
                line=dict(color='#0C4A59', width=3),
                marker=dict(size=6),
                hovertemplate=(
                        '<b>Controlled Concurrent</b><br>' +
                        'Minute: %{x}<br>' +
                        'Mean Preq: %{y:.1f} kW<br>' +
                        '<extra></extra>'
                )
            ))

            fig6.add_trace(go.Scatter(
                x=data['Minute'].tolist() + data['Minute'].tolist()[::-1],
                y=(data['mean_kW'] + data['std_kW']).tolist() +
                  (data['mean_kW'] - data['std_kW']).tolist()[::-1],
                fill='toself',
                fillcolor='#0C4A59',
                opacity=0.1,
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig6.add_hline(y=172.5, line_dash="dash", line_color="black", line_width=2,
                       annotation_text="Station Max: 172.5 kW", annotation_position="right")
        fig6.add_hline(y=86.25, line_dash="dot", line_color="gray", line_width=2,
                       annotation_text="Half Capacity: 86.25 kW", annotation_position="right")

        fig6.update_layout(
            title={
                'text': 'Controlled Concurrent Sessions: Mean Power Requested by Minute<br><sub>Power-limited due to sharing with another vehicle over time</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Minutes from Session Start",
            yaxis_title="Mean Power Requested (kW)",
            xaxis=dict(range=[0, 60]),
            yaxis=dict(range=[0, 370]),
            height=600,
            autosize=True,
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )

        return fig6

    session_categories = categorize_sessions(df)

    measurements['Date and time'] = pd.to_datetime(measurements['Date and time'], dayfirst=True)

    grouped_data_minute = calculate_mean_preq_by_minute(measurements, session_categories, max_minutes=60)

    fig4 = create_graph_not_controlled_by_minute(grouped_data_minute)
    fig5 = create_graph_controlled_alone_by_minute(grouped_data_minute)
    fig6 = create_graph_controlled_concurrent_by_minute(grouped_data_minute)

    return fig4, fig5, fig6


################################################################################################
# MAIN APPLICATION
################################################################################################

def main():
    pd.set_option("display.max_columns", None)

    # Load data
    df = pd.read_csv("Session_data.csv")
    df = correct_data_types(df)
    measurements = pd.read_csv("Measurement_data.csv")

    # RQ1: Generate revenue analysis figures
    revenue_figs = create_revenue_figures(df)

    # RQ3: Generate operational efficiency figures
    rq3_figs = create_rq3_figures(df, "Measurement_data CCS1.csv", "Measurement_data CCS2.csv")

    # RQ1: Generate matrices for heatmaps
    session_matrix = get_session_matrix(df)
    overlap_matrix = get_overlap_matrix(df)

    # RQ4: Generate infrastructure analysis figures
    fig3 = controlled_breakdown(df)
    fig4, fig5, fig6 = power_ampliation(df, measurements)

    # Initialize Dash app
    app = dash.Dash(__name__)

    ################################################################################################
    # LAYOUT
    ################################################################################################

    app.layout = html.Div([

        # ==================== RQ1: REVENUE ANALYSIS (NEW) ====================
        html.H1("Revenue & Temporal Analysis", style={'textAlign': 'center'}),
        html.Hr(),

        # Row 1: Sessions per Hour and Energy per Hour
        html.Div([
            html.Div([
                dcc.Graph(figure=revenue_figs['hour_sessions'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'}),
            html.Div([
                dcc.Graph(figure=revenue_figs['hour_energy'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px'}),

        # Row 2: Revenue per Hour and Sessions per Weekday
        html.Div([
            html.Div([
                dcc.Graph(figure=revenue_figs['hour_revenue'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'}),
            html.Div([
                dcc.Graph(figure=revenue_figs['weekday_sessions'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'marginTop': '10px'}),

        # Row 3: Revenue Trend (full width)
        dcc.Graph(figure=revenue_figs['date_revenue'], style={'height': '400px', 'marginTop': '10px'}),

        # Row 4: Peak vs Off-Peak and Idle Time Analysis
        html.Div([
            html.Div([
                dcc.Graph(figure=revenue_figs['peak_offpeak'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'}),
            html.Div([
                dcc.Graph(figure=revenue_figs['idle_hour'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'}),
            html.Div([
                dcc.Graph(figure=revenue_figs['idle_peak'], style={'height': '400px'})
            ], style={'flex': '1', 'minWidth': '0'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'marginTop': '10px'}),

        # ==================== RQ2: CUSTOMER CHARGING BEHAVIOR ====================
        html.Div([
            html.Hr(),
            html.H1("Customer Charging Behavior", style={'textAlign': 'center'}),

            # RQ2 Filter Dropdown
            html.Div([
                html.Label('Filter by Control Status:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id="controlled-filter",
                    options=[
                        {"label": "All sessions", "value": "all"},
                        {"label": "Controlled only", "value": 1},
                        {"label": "Uncontrolled only", "value": 0},
                    ],
                    value="all",
                    clearable=False,
                    style={"width": "200px"}
                )
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'padding': '20px 10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'
            }),

            # RQ2 Charts - Row 1: Histogram and Scatter side by side
            html.Div([
                html.Div([
                    dcc.Graph(id="hist_soc_arrival", style={'height': '500px'})
                ], style={'flex': '1', 'minWidth': '0'}),

                html.Div([
                    dcc.Graph(id="scatter_stay_energy", style={'height': '500px'})
                ], style={'flex': '1', 'minWidth': '0'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px'}),

            # RQ2 Charts - Row 2: Box plot (full width)
            dcc.Graph(id="box_time_period")
        ]),  # End of RQ2 section

        # ==================== RQ3: OPERATIONAL EFFICIENCY ====================
        html.Div([
            html.Hr(),
            html.H1("Operational Efficiency & Power Management", style={'textAlign': 'center'}),

            # Timeline (full width)
            dcc.Graph(figure=rq3_figs['timeline'], style={'height': '300px'}),

            # Row: Bar chart, Scatter plot, Line chart side by side
            html.Div([
                html.Div([
                    dcc.Graph(figure=rq3_figs['bar_controlled'], style={'height': '400px'})
                ], style={'flex': '1', 'minWidth': '0'}),

                html.Div([
                    dcc.Graph(figure=rq3_figs['scatter_preq_pset'], style={'height': '400px'})
                ], style={'flex': '2', 'minWidth': '0'}),

                html.Div([
                    dcc.Graph(figure=rq3_figs['line_power_time'], style={'height': '400px'})
                ], style={'flex': '2', 'minWidth': '0'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'marginTop': '10px'})
        ]),

        # ==================== RQ1: HEATMAP UTILIZATION ====================
        html.Div([
            html.Hr(),
            html.H1("Utilization Heatmap Analysis", style={'textAlign': 'center'}),

            # --- RQ1 CONTROL PANEL ---
            html.Div([
                # Left: Dynamic Title
                html.H2(id='chart-title', children="Sessions by Hour and Day", style={
                    'margin': '0', 'fontSize': '1.5em', 'minWidth': '350px'
                }),

                # Center: Legend
                html.Div(id='highlight-legend', style={'display': 'none', 'alignItems': 'center'}, children=[
                    html.Div(
                        style={'width': '10px', 'height': '10px', 'backgroundColor': 'green', 'marginRight': '5px'}),
                    html.Span('Off-Peak Hours n = 57 (3.1%)', style={'marginRight': '15px', 'fontSize': '0.9em'}),
                    html.Div(style={'width': '10px', 'height': '10px', 'backgroundColor': 'red', 'marginRight': '5px'}),
                    html.Span('Business Hours n = 1503 (80%)', style={'fontSize': '0.9em'}),
                ]),

                # Right: Controls
                html.Div([
                    html.Div([
                        html.Label('Metric', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[
                                {'label': 'Session Counts', 'value': 'SESSIONS'},
                                {'label': 'Concurrent Overlaps', 'value': 'OVERLAPS'}
                            ],
                            value='SESSIONS',
                            clearable=False,
                            style={'width': '180px'}
                        )
                    ], style={'marginRight': '30px', 'display': 'flex', 'alignItems': 'center'}),

                    html.Div([
                        html.Div([
                            html.Label('Show Labels'),
                            dcc.Checklist(id='show-count-labels', options=[{'label': ' Yes', 'value': 'SHOW_TEXT'}],
                                          value=[], inline=True)
                        ], style={'marginRight': '15px'}),

                        html.Div([
                            html.Label('Show Highlights'),
                            dcc.Checklist(id='show-highlight-boxes',
                                          options=[{'label': ' Yes', 'value': 'SHOW_SHAPES'}],
                                          value=[], inline=True)
                        ])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                'padding': '20px 10px 10px 10px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'
            }),

            # --- RQ1 HEATMAP GRAPH ---
            dcc.Graph(id='heatmap-graph', style={'marginTop': '-5px'})
        ]),

        # ==================== RQ4: INFRASTRUCTURE CAPACITY ====================
        html.Div([
            html.Hr(),
            html.H1("Infrastructure Capacity Constraints Analysis", style={'textAlign': 'center'}),

            # Row 1: Sankey (fig3) and Not Controlled (fig4) side by side
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='sankey-graph',
                        figure=fig3,
                        style={'height': '500px'},
                        responsive=True
                    )
                ], style={'flex': '1', 'minWidth': '0'}),

                html.Div([
                    dcc.Graph(
                        id='not-controlled-graph',
                        figure=fig4,
                        style={'height': '500px'},
                        responsive=True
                    )
                ], style={'flex': '1', 'minWidth': '0'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px'}),

            # Row 2: Controlled Alone (fig5) and Controlled Concurrent (fig6) side by side
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='controlled-alone-graph',
                        figure=fig5,
                        style={'height': '500px'},
                        responsive=True
                    )
                ], style={'flex': '1', 'minWidth': '0'}),

                html.Div([
                    dcc.Graph(
                        id='controlled-concurrent-graph',
                        figure=fig6,
                        style={'height': '500px'},
                        responsive=True
                    )
                ], style={'flex': '1', 'minWidth': '0'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '10px', 'marginTop': '10px'})
        ])
    ])

    ################################################################################################
    # CALLBACKS
    ################################################################################################

    # RQ1: Heatmap callback
    @app.callback(
        [Output('heatmap-graph', 'figure'),
         Output('chart-title', 'children'),
         Output('highlight-legend', 'style'),
         Output('highlight-legend', 'children')],
        [Input('metric-dropdown', 'value'),
         Input('show-count-labels', 'value'),
         Input('show-highlight-boxes', 'value')]
    )
    def update_heatmap(metric, label_values, shape_values):
        # 1. Switch Data Source
        if metric == 'OVERLAPS':
            matrix = overlap_matrix
            colorscale = 'emrld'
            title_text = "Concurrent Charging Events"
        else:
            matrix = session_matrix
            colorscale = 'emrld'
            title_text = "Sessions by Hour and Day"

        # 2. Create Figure
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='<b>%{y} @ %{x}:00</b><br>Count: %{z}<extra></extra>'
        ))

        fig.update_layout(xaxis_title='Hour of Day', yaxis_title='Day of Week', margin=dict(l=80, r=20, t=10, b=50))
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, title_font={"weight": "bold"})
        fig.update_yaxes(title_font={"weight": "bold"})

        # 3. Apply Checkboxes
        if 'SHOW_TEXT' in label_values:
            fig.update_traces(text=matrix.values, texttemplate='%{text}', textfont={"size": 10})

        # 4. Dynamic Shapes based on Metric
        shapes = []
        legend_style = {'display': 'none'}
        legend_children = []
        if 'SHOW_SHAPES' in shape_values:
            legend_style = {'display': 'flex', 'alignItems': 'center', 'marginLeft': '20px'}

            if metric == 'OVERLAPS':
                shapes.append(dict(type="rect", x0=-0.5, x1=7.5, y0=-0.5, y1=6.5, line=dict(color="green", width=3)))
                shapes.append(dict(type="rect", x0=14.5, x1=18.5, y0=-0.5, y1=6.5, line=dict(color="red", width=3)))

                legend_children = [
                    html.Div(
                        style={'width': '10px', 'height': '10px', 'backgroundColor': 'green', 'marginRight': '5px'}),
                    html.Span('Off-Peak Hours n=6 (1.5%)', style={'marginRight': '15px', 'fontSize': '0.9em'}),
                    html.Div(style={'width': '10px', 'height': '10px', 'backgroundColor': 'red', 'marginRight': '5px'}),
                    html.Span('Peak Congestion n=160 (41.3%)', style={'fontSize': '0.9em'}),
                ]

            else:  # SESSIONS
                shapes.append(dict(type="rect", x0=-0.5, x1=5.5, y0=-0.5, y1=6.5, line=dict(color="green", width=3)))
                shapes.append(dict(type="rect", x0=8.5, x1=18.5, y0=-0.5, y1=6.5, line=dict(color="red", width=3)))

                legend_children = [
                    html.Div(
                        style={'width': '10px', 'height': '10px', 'backgroundColor': 'green', 'marginRight': '5px'}),
                    html.Span('Off-Peak Hours n=57 (3.1%)', style={'marginRight': '15px', 'fontSize': '0.9em'}),
                    html.Div(style={'width': '10px', 'height': '10px', 'backgroundColor': 'red', 'marginRight': '5px'}),
                    html.Span('Business Hours n=1503 (80%)', style={'fontSize': '0.9em'}),
                ]

        fig.update_layout(shapes=shapes)

        return fig, title_text, legend_style, legend_children

    # RQ2: Histogram callback
    @app.callback(
        Output("hist_soc_arrival", "figure"),
        Input("controlled-filter", "value")
    )
    def update_histogram(controlled_value):
        # Define consistent color scheme (matching RQ4)
        COLOR_PRIMARY = '#074051'  # Dark teal (controlled)
        COLOR_SECONDARY = '#5daf87'  # Light green (not controlled)

        dfh = df.copy()

        if controlled_value in [0, 1]:
            dfh = dfh[dfh["Controlled session (0=False, 1=True)"] == controlled_value]

        fig = px.histogram(
            dfh,
            x="SOC arrival",
            nbins=30,
            color="Controlled_Cat",
            color_discrete_map={"Not Controlled": COLOR_SECONDARY, "Controlled": COLOR_PRIMARY},
            title="Distribution of SOC at Arrival",
            labels={
                "SOC arrival": "SOC at Arrival (%)",
                "Controlled_Cat": "Session Type"
            },
            category_orders={"Controlled_Cat": ["Not Controlled", "Controlled"]}
        )

        fig.update_layout(
            bargap=0.05,
            title={'text': "Distribution of SOC at Arrival", 'x': 0.5, 'xanchor': 'center'},
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )
        return fig

    # RQ2: Scatter callback
    @app.callback(
        Output("scatter_stay_energy", "figure"),
        Input("controlled-filter", "value")
    )
    def update_scatter(controlled_value):
        # Define consistent color scheme (matching RQ4)
        COLOR_PRIMARY = '#074051'  # Dark teal (controlled)
        COLOR_SECONDARY = '#5daf87'  # Light green (not controlled)

        dfsc = df.copy()

        if controlled_value in [0, 1]:
            dfsc = dfsc[dfsc["Controlled session (0=False, 1=True)"] == controlled_value]

        fig = px.scatter(
            dfsc,
            x="Stay (min)",
            y="Energy_kWh",
            color="Controlled_Cat",
            color_discrete_map={"Not Controlled": COLOR_SECONDARY, "Controlled": COLOR_PRIMARY},
            labels={
                "Stay (min)": "Stay duration (min)",
                "Energy_kWh": "Energy delivered (kWh)",
                "Controlled_Cat": "Session Type"
            },
            hover_data=["Session", "AvgPower_kW", "SOC arrival", "SOC departure", "SOC_Gained"],
            title="Stay Duration vs Energy Delivered",
            category_orders={"Controlled_Cat": ["Not Controlled", "Controlled"]}
        )
        fig.update_layout(
            title={'text': "Stay Duration vs Energy Delivered", 'x': 0.5, 'xanchor': 'center'},
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )
        return fig

    # RQ2: Box plot callback
    @app.callback(
        Output("box_time_period", "figure"),
        Input("controlled-filter", "value")
    )
    def update_boxplot(controlled_value):

        dfb = df.copy()

        if controlled_value in [0, 1]:
            dfb = dfb[dfb["Controlled session (0=False, 1=True)"] == controlled_value]

        fig = px.box(
            dfb,
            x="Arrival_TimePeriod",
            y="SOC_Gained",
            color="Arrival_TimePeriod",
            color_discrete_map={
                "Morning": "#5daf87",
                "Afternoon": "#2E8B57",
                "Evening": "#0C4A59",
                "Night": "#074051"
            },
            title="SOC Gain by Time Period of the Day",
            labels={
                "Arrival_TimePeriod": "Time of Day (Arrival)",
                "SOC_Gained": "SOC Gain (%)"
            },
            category_orders={
                "Arrival_TimePeriod": ["Morning", "Afternoon", "Evening", "Night"]
            }
        )
        fig.update_layout(
            title={'text': "SOC Gain by Time Period of the Day", 'x': 0.5, 'xanchor': 'center'},
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )
        return fig

################################################################################################

app = dash.Dash(__name__)
server = app.server
