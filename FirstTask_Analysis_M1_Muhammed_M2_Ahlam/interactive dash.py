import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Load and preprocess the data
df = pd.read_csv('/home/omran-xy/Workspace/Cellula/Task one/first inten project.csv')

df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')
df.dropna(subset=['date of reservation'], inplace=True)
df['room type'] = df['room type'].str.extract('(\d+)').astype(int)
df['type of meal'] = df['type of meal'].map({'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Not Selected': 0})
df['booking status'] = df['booking status'].map({'Canceled': 1, 'Not_Canceled': 0})
df['market segment type'] = df['market segment type'].map(
    {'Offline': 'Offline', 'Online': 'Online', 'Corporate': 'Corporate', 'Aviation': 'Aviation', 'Complementary': 'Complementary'}
)
df['number of weekend nights'] = df['number of weekend nights'].astype(int)
df['number of week nights'] = df['number of week nights'].astype(int)

df.dropna(inplace=True)

app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Hotel Booking Dashboard'),
    
    html.Div([
        html.Div([
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=df['date of reservation'].min(),
                max_date_allowed=df['date of reservation'].max(),
                initial_visible_month=df['date of reservation'].min(),
                start_date=df['date of reservation'].min(),
                end_date=df['date of reservation'].max()
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.RangeSlider(
                id='weekend-nights-slider',
                min=df['number of weekend nights'].min(),
                max=df['number of weekend nights'].max(),
                value=[df['number of weekend nights'].min(), df['number of weekend nights'].max()],
                marks={i: str(i) for i in range(df['number of weekend nights'].min(), df['number of weekend nights'].max() + 1)},
                step=1
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='adults-children-chart')
    ], style={'width': '48%', 'display': 'inline-block'}),
    
    html.Div([
        dcc.Graph(id='room-types-chart')
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='market-segment-chart')
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='booking-status-chart')
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])

@app.callback(
    [Output('adults-children-chart', 'figure'),
     Output('room-types-chart', 'figure'),
     Output('market-segment-chart', 'figure'),
     Output('booking-status-chart', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('weekend-nights-slider', 'value')]
)
def update_charts(start_date, end_date, weekend_nights):
    filtered_df = df[
        (df['date of reservation'] >= start_date) & 
        (df['date of reservation'] <= end_date) &
        (df['number of weekend nights'] >= weekend_nights[0]) & 
        (df['number of weekend nights'] <= weekend_nights[1])
    ]
    
    # Adults and Children Chart
    adults_children = filtered_df[['number of adults', 'number of children']].sum()
    adults_children_fig = go.Figure(data=[
        go.Bar(name='Adults', x=['Adults'], y=[adults_children['number of adults']]),
        go.Bar(name='Children', x=['Children'], y=[adults_children['number of children']])
    ])
    adults_children_fig.update_layout(title='Number of Adults and Children')
    
    # Room Types Chart
    room_types = filtered_df['room type'].value_counts().sort_index()
    room_types_fig = px.bar(x=room_types.index, y=room_types.values, labels={'x': 'Room Type', 'y': 'Count'})
    room_types_fig.update_layout(title='Distribution of Room Types')
    
    # Market Segment Type Pie Chart
    market_segment = filtered_df['market segment type'].value_counts()
    market_segment_fig = px.pie(values=market_segment.values, names=market_segment.index, title='Market Segment Distribution')
    
    # Booking Status Pie Chart
    booking_status = filtered_df['booking status'].map({1: 'Canceled', 0: 'Not Canceled'}).value_counts()
    booking_status_fig = px.pie(values=booking_status.values, names=booking_status.index, title='Booking Status Distribution')
    
    return adults_children_fig, room_types_fig, market_segment_fig, booking_status_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)