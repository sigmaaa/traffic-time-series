from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats
import pymannkendall as mk
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
            {"name": "Trend Direction", "id": "trend_direction"},
            {"name": "p-value", "id": "p_value"},
            {"name": "Hypothesis Test Result (Reject Null)",
             "id": "test_result"},
            {"name": "Z-statistic", "id": "z_statistic"},
            {"name": "Kendall's Tau", "id": "kendall_tau"},
            {"name": "S-statistic", "id": "s_statistic"},
            {"name": "Variance of S", "id": "var_s"},
            {"name": "Slope of Trend Line", "id": "slope"},
            {"name": "Intercept of Trend Line", "id": "intercept"}
        ],
        style_table={'margin': 'auto'},
        style_cell={'textAlign': 'center'}
    ),

    html.H3("Autocorrelation Function (ACF)", style={'textAlign': 'center'}),



    dcc.Graph(id='correlation-graph'),
    # Повзунок для вибору лагів
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

    html.H3("Trend-Removed Traffic Graph", style={'textAlign': 'center'}),
    dcc.Graph(id='residuals-graphic'),

    html.H3("Sign Method Residuals Analysis", style={'textAlign': 'center'}),
    dash_table.DataTable(
        id='sign-method-table-residuals',
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
])

# Колбек для оновлення графіка, таблиць та кореляційного аналізу

# === Функції для обчислень ===


def calculate_sign_test(data):
    """Виконує знаковий тест для визначення тренду."""
    diff = data.diff().dropna()
    positive_signs_count = (diff > 0).sum()

    n = len(diff)
    expected_positive_signs = n / 2
    positive_signs_variance = (n + 1) / 12
    standardized_test_stat = (
        positive_signs_count - expected_positive_signs) / (positive_signs_variance ** 0.5)
    critical_threshold = stats.norm.ppf(1 - 0.05 / 2)

    hypothesis_result = "Trend Detected" if abs(
        standardized_test_stat) > critical_threshold else "No Trend"

    return [{
        "Positive Signs Count": positive_signs_count,
        "Expected Positive Signs": expected_positive_signs,
        "Positive Signs Variance": round(positive_signs_variance, 4),
        "Standardized Test Stat": round(standardized_test_stat, 4),
        "Critical Threshold": round(critical_threshold, 4),
        "Hypothesis Test": hypothesis_result
    }]


def calculate_mann_rank_test(data):
    mann_ranking_test_result = mk.original_test(data)
    return [{
        "trend_direction": mann_ranking_test_result.trend,
        "p_value": float(mann_ranking_test_result.p),
        "test_result": "Reject" if mann_ranking_test_result.h else "Do not reject",
        "z_statistic": float(mann_ranking_test_result.z),
        "kendall_tau": float(mann_ranking_test_result.Tau),
        "s_statistic": float(mann_ranking_test_result.s),
        "var_s": float(mann_ranking_test_result.var_s),
        "slope": float(mann_ranking_test_result.slope),
        "intercept": float(mann_ranking_test_result.intercept)
    }]
# === Колбеки ===


@callback(
    [Output('indicator-graphic', 'figure'),
     Output('sign-method-table', 'data'),
     Output('mann-rank-table', 'data'),
     Output('residuals-graphic', 'figure'),
     Output('sign-method-table-residuals', 'data')],
    Input('junction-selection', 'value')
)
def update_graph(selected_junction):
    filtered_df = df[df["Junction"] == selected_junction][[
        "DateTime", "Vehicles"]].dropna()

    # Візуалізація
    fig = px.line(filtered_df, x='DateTime', y='Vehicles',
                  title=f'Vehicle Count at Junction {selected_junction}')
    fig.update_layout(xaxis_title="Date & Time",
                      yaxis_title="Number of Vehicles", hovermode='x unified')

    # Виконання тестів
    sign_test_result = calculate_sign_test(filtered_df["Vehicles"])
    mann_ranking_test_result = calculate_mann_rank_test(
        filtered_df["Vehicles"])

    # Видалення тренду
    residuals = filtered_df["Vehicles"].diff()
    fig_residuals = px.line(filtered_df, x='DateTime', y=residuals,
                            title=f'Vehicle Count at Junction {selected_junction}')
    fig_residuals.update_layout(xaxis_title="Date & Time",
                                yaxis_title="Number of Vehicles", hovermode='x unified')
    sign_test_residuals_result = calculate_sign_test(residuals)
    return fig, sign_test_result, mann_ranking_test_result, fig_residuals, sign_test_residuals_result


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
    app.run(debug=False)
