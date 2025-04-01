from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import acf

# Завантаження даних
df = pd.read_csv("train_ML_IOT.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])  # Конвертація дати

# Ініціалізація Dash
app = Dash(__name__)

# Макет сторінки
app.layout = html.Div(children=[
    html.H1('Traffic Monitoring Dashboard', style={'textAlign': 'center'}),

    html.Label('Select Junction:'),
    dcc.Dropdown(
        id='junction-selection',
        options=[{'label': f'Junction {j}', 'value': j}
                 for j in sorted(df["Junction"].unique())],
        value=df["Junction"].unique()[0],
        placeholder="Select a junction...",
        searchable=True
    ),

    dcc.Loading(
        type="circle",
        children=[dcc.Graph(id='indicator-graphic')]
    ),

    html.H3("Sign Method Analysis", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='sign-method-table',
        columns=[
            {"name": "Positive Signs Count", "id": "Positive Signs Count"},
            {"name": "Expected Positive Signs", "id": "Expected Positive Signs"},
            {"name": "Positive Signs Variance", "id": "Positive Signs Variance"},
            {"name": "Standardized Test Stat", "id": "Standardized Test Stat"},
            {"name": "Critical Threshold", "id": "Critical Threshold"},
            {"name": "Hypothesis Test", "id": "Hypothesis Test"},
        ],
        style_table={'margin': 'auto'},
        style_cell={'textAlign': 'center'}
    ),

    html.H3("Mann Rank Statistic", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='mann-rank-table',
        columns=[
            {"name": "Mann Rank Statistic", "id": "Mann Rank Statistic"},
            {"name": "p-value", "id": "p-value"},
            {"name": "Hypothesis Test", "id": "Hypothesis Test"},
        ],
        style_table={'margin': 'auto'},
        style_cell={'textAlign': 'center'}
    ),

    html.H3("Autocorrelation Function (ACF)", style={'textAlign': 'center'}),



    dcc.Graph(id='correlation-graph'),
    # Повзунок для вибору лагів
    html.Label("Select number of lags:"),
    html.Div([
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
        )
    ], style={'margin-bottom': '10%'})
])

# Колбек для оновлення графіка, таблиць та кореляційного аналізу


@callback(
    [Output('indicator-graphic', 'figure'),
     Output('sign-method-table', 'data'),
     Output('mann-rank-table', 'data')],
    Input('junction-selection', 'value')
)
def update_graph(selected_junction):
    filtered_df = df[df["Junction"] ==
                     selected_junction][["DateTime", "Vehicles"]]

    # Візуалізація
    fig = px.line(filtered_df, x='DateTime', y='Vehicles',
                  title=f'Vehicle Count at Junction {selected_junction}')
    fig.update_layout(xaxis_title="Date & Time",
                      yaxis_title="Number of Vehicles", hovermode='x unified')

    # --- Аналіз методом знаків ---
    df_grouped = filtered_df.copy()
    df_grouped["diff"] = df_grouped["Vehicles"].diff()
    df_grouped["sign"] = df_grouped["diff"].apply(lambda x: 1 if x > 0 else 0)

    positive_signs_count = df_grouped["sign"].sum()
    expected_positive_signs = (len(df_grouped) - 1) / 2
    positive_signs_variance = (len(df_grouped) + 1) / 12
    standardized_test_stat = (
        positive_signs_count - expected_positive_signs) / (positive_signs_variance ** 0.5)
    critical_threshold = stats.norm.ppf(1 - 0.05 / 2)

    hypothesis_result = "Trend Detected" if abs(
        standardized_test_stat) > critical_threshold else "No Trend"

    table_data = [{
        "Positive Signs Count": positive_signs_count,
        "Expected Positive Signs": expected_positive_signs,
        "Positive Signs Variance": round(positive_signs_variance, 4),
        "Standardized Test Stat": round(standardized_test_stat, 4),
        "Critical Threshold": round(critical_threshold, 4),
        "Hypothesis Test": hypothesis_result
    }]

    # --- Ранговий тест Манна ---
    data = df_grouped["Vehicles"].dropna().values  # Видаляємо NaN, якщо є
    n = len(data)
    half = n // 2

    data_part1 = data[:half]
    data_part2 = data[half:]

    statistic, p_value = stats.mannwhitneyu(
        data_part1, data_part2, alternative='two-sided')
    mannwhitney_conclusion = "Trend Detected" if p_value < 0.05 else "No Trend"

    mannwhitney_table_data = [{
        "Mann Rank Statistic": statistic,
        "p-value": round(p_value, 4),
        "Hypothesis Test": mannwhitney_conclusion
    }]

    return fig, table_data, mannwhitney_table_data


@callback(
    Output('correlation-graph', 'figure'),
    [Input('junction-selection', 'value'),
     Input('lag-slider', 'value')]
)
def update_correlation_graph(selected_junction, lags):
    # Фільтрація даних
    vehicles = df[df["Junction"] == selected_junction]["Vehicles"].dropna()

    # Розрахунок ACF (автокореляційної функції)
    acf_values = acf(vehicles, nlags=lags, fft=True)

    # Створення DataFrame для візуалізації
    acf_df = pd.DataFrame(
        {"Lag": np.arange(len(acf_values)), "ACF": acf_values})

    # Побудова графіка з plotly.express
    fig = px.line(acf_df, x="Lag", y="ACF",
                  title=f"Autocorrelation Function (Junction {selected_junction})", markers=True)

    # Налаштування графіка
    fig.update_layout(
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        hovermode="x"
    )

    return fig


# Запуск сервера
if __name__ == '__main__':
    app.run(debug=True)
