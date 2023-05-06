import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash import dash_table
import plotly.express as px
from dash import dcc
from plotly.subplots import make_subplots
import numpy as np
from dash import html
from dash.dependencies import Output
from dash.dependencies import Input, Output


df = pd.read_csv('dataset/Train_Test.csv')
# Filter the data to include only Train and Test
train_test_df = df[df['label'].isin(['Train', 'Test'])]
raw_df = pd.read_csv('dataset/fullnews2.csv')
clean_df = pd.read_csv('dataset/clean2.csv')
dfAccuracy = pd.read_csv("dataset/AccuracyAll.csv")
dfLossAccuracy = pd.read_csv("dataset/Loss_Accuracy.csv")
dfLoss3 = pd.read_csv("dataset/3-loss.csv")
# Calculate the grand total
grand_total = train_test_df['Value'].sum()


# Lấy danh sách tên model
model_names = dfAccuracy['Model_name'].unique()

# Thêm option "All Models" vào danh sách tên model
model_names = np.append(model_names, "All Models")

# Tạo dropdown options cho model selection
dropdown_options = [{'label': name, 'value': name} for name in model_names]
# Create figure
fig = px.bar(dfLossAccuracy, x="Model_name", y=["Loss", "Accuracy"],
             barmode="group", title="Loss and Accuracy by Model (Test dataset)")

# Define the pie chart traces
traces = [
    go.Pie(labels=train_test_df['label'], values=train_test_df['Value']),
    go.Pie(labels=['Total'], values=[grand_total])
]


external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "VN30INDEX Prediction"


app.layout = html.Div(
    children=[
    html.Div(
        children=[
            html.P(children="📈", className="header-emoji"),
            html.H1(
                children="VN30INDEX Prediction", className="header-title"
            ),
            html.P(
                children=(
                    "Đỗ Chí Bảo - "
                    "Nguyễn Thị Mỹ Hải - "
                    "Lê Hoàng Khiêm"
                ),
                className="header-description",
            ),
        ],
        className="header",
    ),
        dcc.Graph(
            id='pie-chart',
            figure={
                'data': traces,
                'layout': go.Layout(title='Train vs Test Pie Chart')
            }
        ),
        html.H2("Raw data"),
        dash_table.DataTable(
            id='raw-and-clean-table',
            columns=[{"name": i, "id": i} for i in raw_df.columns],
            data=raw_df.head(10).to_dict('records'),
            style_cell={'textAlign': 'center', 'width': 'auto', 'minWidth': '50px', 'maxWidth': '200px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=10,
            fixed_rows={'headers': True, 'data': 0},
            style_table={
                'maxHeight': '500px',
                'overflowY': 'scroll',
                'width': '100%'
            }
        ),
        html.H2("Clean data"),
        dash_table.DataTable(
            id='clean-table',
            columns=[{"name": i, "id": i} for i in clean_df.columns],
            data=clean_df.head(10).to_dict('records'),
            style_cell={'textAlign': 'center', 'width': 'auto', 'minWidth': '50px', 'maxWidth': '200px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            page_size=10,
            fixed_rows={'headers': True, 'data': 0},
            style_table={
                'maxHeight': '500px',
                'overflowY': 'scroll',
                'width': '100%'
            }
        ),
        html.Label("Model Selection"),
        dcc.Dropdown(
            id='model-selection',
            options=dropdown_options,
            value="BiGRU",
            clearable=False
        ),
        html.H1(children='Accuracy Comparison'),
        dcc.Graph(
            id='accuracy-graph',
            figure={}
        ),
        html.H1("Loss and Accuracy by Model (Test dataset)"),
        dcc.Graph(figure=fig)


        
    ]
)

# Callback function to update the graph based on the selected model

@app.callback(
    Output('accuracy-graph', 'figure'),
    Input('model-selection', 'value')
)
def update_graph(model):
    if model == "All Models":
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=model_names[:5])
        for i, name in enumerate(model_names[:5]):
            df = dfAccuracy[dfAccuracy['Model_name'] == name]
            fig.add_trace(go.Scatter(x=df['Epoch_No'], y=df['accuracy'], mode='lines', name='accuracy'), row=i+1, col=1)
        fig.update_layout(height=1200, showlegend=False)

        fig.update_yaxes(range=[0, 1])
    else:
        # Hiển thị biểu đồ cho mô hình được lựa chọn
        df = dfAccuracy[dfAccuracy['Model_name'] == model]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Epoch_No'], y=df['accuracy'],
                                 mode='lines', name='accuracy'))
        fig.add_trace(go.Scatter(x=df['Epoch_No'], y=df['val_accuracy'],
                                 mode='lines', name='val_accuracy'))
        fig.add_trace(go.Scatter(x=df['Epoch_No'], y=df['test_accuracy'],
                                 mode='lines', name='test_accuracy'))
        fig.update_layout(title=model + " Accuracy",
                          xaxis_title="Epoch No",
                          yaxis_title="Accuracy")
        fig.update_layout(height=500, showlegend=True)
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)
