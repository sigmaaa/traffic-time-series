import base64
import io
from dash import Dash, State, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit

# Завантаження даних
df = pd.read_csv("train_ML_IOT.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])  # Конвертація дати

# Ініціалізація Dash
app = Dash(__name__)

# Макет сторінки
app.layout = html.Div(children=[
    dcc.Store(id='stored-data'),
    html.H1('Traffic Monitoring Dashboard', style={'textAlign': 'center'}),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Label('Select Junction:'),
    dcc.Dropdown(
        id='junction-selection',
        options=[{'label': f'Junction {j}', 'value': j}
                 for j in sorted(df["Junction"].unique())],
        value=df["Junction"].unique()[0],
        placeholder="Select a junction...",
        searchable=True
    ),
    dcc.Store(id="filtered-df-store"),
    dcc.Loading(
        type="circle",
        children=[dcc.Graph(id='indicator-graphic')]
    ),

    html.H5("Coeficients for Weighted moving average l=7, p=3"),
    dcc.Markdown('$$U_t = a_0 + a_1t + a_2t^2 + a_3t^3$$', mathjax=True),
    dash_table.DataTable(
        columns=[
            {"name": f"a{i}", "id": f"a{i}"} for i in range(7)
        ],
        data=[{f"a{i}": round(w, 4)
               for i, w in enumerate([-2, 3, 6, 7, 6, 3, -2])}],
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'},
        style_table={'overflowX': 'auto'},
    ),


    html.H3("Autocorrelation Function (ACF)", style={'textAlign': 'center'}),
    dcc.Graph(id='correlation-graph'),

    html.Label("Select number of lags:"),
    dcc.Slider(
        id='lag-slider',
        min=1,
        max=50,
        step=1,
        value=10,
        marks={i: {'label': str(i), 'style': {'fontSize': '14px'}}
               for i in range(0, 51, 5)},
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag',
        included=False
    ),

    # Таблиця з filtered_df
    html.H3("Filtered Data Table", style={
            'textAlign': 'center', 'marginTop': '40px'}),
    html.Div(id='filtered-table-container'),
    html.Div([
        html.Button(
            "Download Table Data",
            id="download-button",
            n_clicks=0,
            style={
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'margin': '10px 0'
            }
        ),
        dcc.Download(id="download-component")
    ]),
    # Таблиця з прогнозованими даними
    html.H3("Forecasted Data Table", style={
            'textAlign': 'center', 'marginTop': '40px'}),
    html.Div(id='forecast-table-container'),

    # Вивід формул та коефіцієнтів трендів
    html.H3("Trend Formulas and Coefficients", style={
            'textAlign': 'center', 'marginTop': '40px'}),
    html.Div(id='trend-formulas-container')
])


def weighted_moving_average(data, l=7, p=3):
    weights = np.array([-2, 3, 6, 7, 6, 3, -2])
    weights = weights / 21

    n = len(data)
    smoothed = np.full(n, np.nan)

    for i in range(p, n - l):
        window_data = np.array(data[i:i + l])
        smoothed_value = np.sum(window_data * weights)
        smoothed[i] = smoothed_value

    # Replace np.nan with original data where smoothing wasn't applied
    smoothed = np.where(np.isnan(smoothed), data, smoothed)

    return smoothed


def get_linear_trend(data):
    t = np.arange(1, len(data) + 1)
    b, a = np.polyfit(t, data, deg=1)
    trend = a + b * t
    return trend, a, b


def get_nonlinear_trend(data):
    t = np.arange(1, len(data) + 1)
    log_y = np.log(data)
    slope, intercept = np.polyfit(t, log_y, deg=1)
    a1 = np.exp(slope)
    a0 = np.exp(intercept)
    nonlinear_trend = a0 * (a1 ** t)
    return nonlinear_trend, a0, a1


def trend_sin(x, a, b, A, phi):
    omega = 2 * np.pi / 24
    days = (x - 1) // 24 + 1
    day_of_week = ((days - 1) % 7) + 1
    modifier = np.where((day_of_week == 7) | (day_of_week == 1), 0.5, 1.0)
    return a * x + b + A * modifier * np.sin(omega * x + phi)


def get_custom_trend(data):
    initial_guess = [0, np.mean(data), 5, 0]
    t = np.arange(1, len(data) + 1)
    params, covariance = curve_fit(trend_sin, t, data, p0=initial_guess)
    trend_values = trend_sin(t, *params)
    return trend_values, params


def seasonal_fourier_component(data, k=2, period=24):
    t = np.arange(1, len(data) + 1)
    result = np.zeros(len(data))

    for i in range(1, k + 1):
        omega = 2 * np.pi * i / period
        a = np.sum(data * np.cos(omega * t)) * 2 / len(data)
        b = np.sum(data * np.sin(omega * t)) * 2 / len(data)
        result += a * np.cos(omega * t) + b * np.sin(omega * t)
    return result


def forecast_next_n_steps(data, n_steps=4):
    t_existing = np.arange(1, len(data) + 1)
    t_future = np.arange(len(data) + 1, len(data) + n_steps + 1)

    # Лінійний тренд
    b, a = np.polyfit(t_existing, data, deg=1)
    future_trend = a + b * t_future

    # Фур'є-сезонність
    future_seasonal = np.zeros(n_steps)
    for i in range(1, 3):
        omega = 2 * np.pi * i / 24
        a_coef = np.sum(data * np.cos(omega * t_existing)) * 2 / len(data)
        b_coef = np.sum(data * np.sin(omega * t_existing)) * 2 / len(data)
        future_seasonal += a_coef * \
            np.cos(omega * t_future) + b_coef * np.sin(omega * t_future)

    for i in range(1, 3):
        omega = 2 * np.pi * i / (24 * 7)
        a_coef = np.sum(data * np.cos(omega * t_existing)) * 2 / len(data)
        b_coef = np.sum(data * np.sin(omega * t_existing)) * 2 / len(data)
        future_seasonal += a_coef * \
            np.cos(omega * t_future) + b_coef * np.sin(omega * t_future)

    forecast = future_trend + future_seasonal
    return forecast, a, b  # повертаємо також коефіцієнти лінійного тренду

# === Колбеки ===


@callback(
    [Output('indicator-graphic', 'figure'),
     Output('filtered-table-container', 'children'),
     Output('forecast-table-container', 'children'),
     Output('trend-formulas-container', 'children'),
     Output('filtered-df-store', 'data')],
    [Input('junction-selection', 'value'),
     Input('stored-data', 'data')]
)
def update_all(selected_junction, stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
    else:
        df = pd.read_csv("train_ML_IOT.csv")
        df["DateTime"] = pd.to_datetime(df["DateTime"])

    filtered_df = df[df["Junction"] == selected_junction][[
        "DateTime", "Vehicles"]].dropna()

    # Обчислення трендів і коефіцієнтів
    filtered_df["Smoothed"] = weighted_moving_average(filtered_df["Vehicles"])
    filtered_df["LinearTrend"], a_lin, b_lin = get_linear_trend(
        filtered_df["Smoothed"])
    filtered_df["NonlinearTrend"], a_nonlin, b_nonlin = get_nonlinear_trend(
        filtered_df["Smoothed"])
    filtered_df["CustomTrend"], params_custom = get_custom_trend(
        filtered_df["Smoothed"])
    daily_fourier = seasonal_fourier_component(
        filtered_df["Smoothed"], k=3, period=24)
    weekly_fourier = seasonal_fourier_component(
        filtered_df["Smoothed"], k=2, period=24*7)
    seasonal = daily_fourier + weekly_fourier
    filtered_df["FourierSeasonal"] = seasonal
    filtered_df["FourierSeasonal + LinearTrend"] = seasonal + \
        filtered_df["LinearTrend"]

    # Прогноз
    n_steps = 100
    forecast_values, a_forecast, b_forecast = forecast_next_n_steps(
        filtered_df["Smoothed"], n_steps=n_steps)
    last_date = filtered_df["DateTime"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(hours=1), periods=n_steps, freq='H')

    # Фігура графіку
    fig = px.line(title=f'Vehicle Count at Junction {selected_junction}')
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["Vehicles"],
                    mode='lines', name='Original', line=dict(color='lightgray'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["Smoothed"],
                    mode='lines', name='Smoothed (WMA)', line=dict(color='blue'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["LinearTrend"],
                    mode='lines', name='Linear Trend (LMS)', line=dict(color='red'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["NonlinearTrend"],
                    mode='lines', name='Nonlinear Trend', line=dict(color='green'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["CustomTrend"],
                    mode='lines', name='CustomTrend', line=dict(color='cyan'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["FourierSeasonal"],
                    mode='lines', name='Fourier Seasonal (k=2)', line=dict(color='orange'))
    fig.add_scatter(x=filtered_df["DateTime"], y=filtered_df["FourierSeasonal + LinearTrend"],
                    mode='lines', name='Fourier Seasonal + Linear Trend', line=dict(color='purple'))
    fig.add_scatter(x=future_dates, y=forecast_values,
                    mode='lines', name='Forecast', line=dict(color='magenta', dash='dot'))

    # Таблиця з filtered_df
    filtered_table = dash_table.DataTable(
        data=filtered_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )

    # Таблиця з прогнозом
    forecast_df = pd.DataFrame({
        "DateTime": future_dates,
        "Forecast_Vehicles": forecast_values
    })
    forecast_table = dash_table.DataTable(
        data=forecast_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in forecast_df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
    )

    # Вивід формул трендів у html
    trend_text = html.Div([
        html.Div([
            dcc.Markdown("**Linear Trend:**"),
            dcc.Markdown(
                f"""$$
            y = a + b \\cdot t =  {a_lin:.2f} + {b_lin:.5f} \\cdot t
            $$""",
                mathjax=True
            ),
        ]),
        html.Div([
            dcc.Markdown("**Nonlinear Trend:**"),
            dcc.Markdown(
                f"""$$
            y = a_0 \\cdot a_1^t =  {a_nonlin:.2f} \\cdot ({b_nonlin:.5f})^t
            $$""",
                mathjax=True
            ),
        ]),
        html.Div([
            dcc.Markdown(
                "**Forecast Model: Linear Trend with Multiple Seasonal Components (Fourier Series):**"),
            dcc.Markdown(
                f"""$$
        y(t) = {a_forecast:.2f} + {b_forecast:.5f} \\cdot t + 
        \\sum_{{k=1}}^{{2}} \\left( a_k^{{(24)}} \\cdot \\cos\\left(\\frac{{2\\pi k t}}{{24}}\\right) + 
        b_k^{{(24)}} \\cdot \\sin\\left(\\frac{{2\\pi k t}}{{24}}\\right) \\right) + 
        \\sum_{{k=1}}^{{2}} \\left( a_k^{{(168)}} \\cdot \\cos\\left(\\frac{{2\\pi k t}}{{168}}\\right) + 
        b_k^{{(168)}} \\cdot \\sin\\left(\\frac{{2\\pi k t}}{{168}}\\right) \\right)
        $$""",
                mathjax=True
            )
        ])
    ])

    return fig, filtered_table, forecast_table, trend_text, filtered_df.to_dict('records')


@callback(
    Output("download-component", "data"),
    Input("download-button", "n_clicks"),
    State("filtered-df-store", "data"),
    prevent_initial_call=True
)
def download_filtered_df(n_clicks, stored_filtered_df):

    df = pd.DataFrame(stored_filtered_df)
    return dcc.send_data_frame(df.to_csv, "filtered_data.csv", index=False)


@callback(
    Output('correlation-graph', 'figure'),
    [Input('junction-selection', 'value'),
     Input('lag-slider', 'value'),
     Input('stored-data', 'data')]
)
def update_correlation(selected_junction, lag, stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
    else:
        df = pd.read_csv("train_ML_IOT.csv")
        df["DateTime"] = pd.to_datetime(df["DateTime"])

    filtered_df = df[df["Junction"] == selected_junction]
    filtered_df = filtered_df.dropna(subset=["Vehicles"])

    acf_vals = acf(filtered_df["Vehicles"], nlags=lag, fft=False)

    fig = px.line(x=list(range(len(acf_vals))), y=acf_vals,
                  labels={"x": "Lag", "y": "ACF"},
                  title=f"Autocorrelation Function for Junction {selected_junction}")

    return fig


@callback(
    Output('stored-data', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_data(contents, filenames):
    if contents is not None:
        dfs = []
        for content, filename in zip(contents, filenames):
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if filename.endswith('.csv'):
                    df_new = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                else:
                    continue
            except Exception as e:
                print(e)
                continue
            dfs.append(df_new)
        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            df_all["DateTime"] = pd.to_datetime(df_all["DateTime"])
            return df_all.to_dict('records')
    return None


if __name__ == '__main__':
    app.run(debug=False)
