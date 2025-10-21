import os

import random
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px


app = Dash(__name__)

app.layout = html.Div([
    html.H1("Profit After Tax Simulation Dashboard", style={'textAlign': 'center'}),

    # --- INPUTS ---
    html.Div([
        html.H3("Sales Price (Uniform)"),
        html.Label("Min:", style={'marginRight': '5px'}),
        dcc.Input(id='sp_min', type='number', value=7.2, step=0.1, style={'marginRight': '20px'}),
        html.Label("Max:", style={'marginRight': '5px'}),
        dcc.Input(id='sp_max', type='number', value=8.1, step=0.1),

        html.H3("Cost of Sales (Normal)"),
        html.Label("Mean:", style={'marginRight': '5px'}),
        dcc.Input(id='cost_mean', type='number', value=4.8, step=0.1, style={'marginRight': '20px'}),
        html.Label("Std Dev:", style={'marginRight': '5px'}),
        dcc.Input(id='cost_std', type='number', value=0.75, step=0.1),

        html.H3("Sales Volume (Triangular)"),
        html.Label("Min:", style={'marginRight': '5px'}),
        dcc.Input(id='vol_min', type='number', value=20000, step=1000, style={'marginRight': '20px'}),
        html.Label("Moderate:", style={'marginRight': '5px'}),
        dcc.Input(id='vol_mode', type='number', value=40000, step=1000, style={'marginRight': '20px'}),
        html.Label("Max:", style={'marginRight': '5px'}),
        dcc.Input(id='vol_max', type='number', value=60000, step=1000),

        html.H3("Simulations"),
        html.Label("Number of simulations:", style={'marginRight': '5px'}),
        dcc.Input(id='n_sim', type='number', value=10000, step=100),

        html.Br(), html.Br(),
        html.Button('Run Simulation', id='run_btn', n_clicks=0, style={'fontSize': '18px', 'padding': '10px'})
    ], style={
        'padding': 20, 'border': '2px solid #ddd', 'borderRadius': '10px',
        'margin': '20px', 'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto'
    }),

    html.Hr(),

    # --- OUTPUTS ---
    html.Div(id='results-stats', style={
        'fontSize': '18px',
        'textAlign': 'center',
        'margin': '20px',
        'padding': '15px',
        'border': '1px solid #ccc',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '10px'
    }),

    # --- GRAPHS ---
    html.Div([
        dcc.Graph(id='histogram', style={'width': '50%'}),
        dcc.Graph(id='scurve-chart', style={'width': '50%'})  # NEW: S-Curve Graph
    ], style={'display': 'flex'})

])


# --- CALLBACK TO RUN SIMULATION ---
@app.callback(
    [Output('results-stats', 'children'),
     Output('histogram', 'figure'),
     Output('scurve-chart', 'figure')],  # NEW: Added S-Curve output
    [Input('run_btn', 'n_clicks')],
    [State('sp_min', 'value'),
     State('sp_max', 'value'),
     State('cost_mean', 'value'),
     State('cost_std', 'value'),
     State('vol_min', 'value'),
     State('vol_mode', 'value'),
     State('vol_max', 'value'),
     State('n_sim', 'value')]
)
def simulate_profit(n_clicks, sp_min, sp_max, cost_mean, cost_std, vol_min, vol_mode, vol_max, n_sim):
    # Don't run until the button is clicked
    if n_clicks == 0:
        # Return empty figures for both graphs
        return "Click 'Run Simulation' to see the results.", px.Figure(), px.Figure()

    # --- 1. Run Simulation ---
    profits = []
    n_sim_int = int(n_sim)
    for _ in range(n_sim_int):
        sales_price = random.uniform(sp_min, sp_max)
        cost = random.gauss(cost_mean, cost_std)
        sales_volume = random.triangular(vol_min, vol_max, vol_mode)
        # Discrete Tax: 80% chance 0%, 20% chance 30%
        tax_rate = np.random.choice([0, 0.3], p=[0.8, 0.2])

        profit = (sales_price - cost) * sales_volume * (1 - tax_rate)
        profits.append(profit)

    # --- 2. Calculate Statistics (with Percentiles) ---
    mean_val = np.mean(profits)
    median_val = np.median(profits)
    std_val = np.std(profits)
    min_val = np.min(profits)
    max_val = np.max(profits)

    # NEW: Calculate Percentiles
    p10 = np.percentile(profits, 10)
    p25 = np.percentile(profits, 25)
    p75 = np.percentile(profits, 75)
    p90 = np.percentile(profits, 90)

    loss_count = sum(1 for p in profits if p < 0)
    prob_loss = (loss_count / n_sim_int) * 100

    # --- 3. Format Statistics Output (Updated) ---
    stats_children = [
        html.H3("Profit After Tax Statistics"),
        # Using flexbox to create columns
        html.Div([
            html.Div([
                html.P(f"Mean: ${mean_val:,.2f}"),
                html.P(f"Median: ${median_val:,.2f}"),
                html.P(f"Std. Dev: ${std_val:,.2f}"),
                html.P(f"Prob. of Loss: {prob_loss:.2f}%"),
            ], style={'width': '33%', 'textAlign': 'left', 'paddingLeft': '5%'}),

            html.Div([
                html.P(f"Min: ${min_val:,.2f}"),
                html.P(f"10th Percentile: ${p10:,.2f}"),
                html.P(f"25th Percentile: ${p25:,.2f}"),
            ], style={'width': '33%', 'textAlign': 'left'}),

            html.Div([
                html.P(f"Max: ${max_val:,.2f}"),
                html.P(f"90th Percentile: ${p90:,.2f}"),
                html.P(f"75th Percentile: ${p75:,.2f}"),
            ], style={'width': '33%', 'textAlign': 'left'}),

        ], style={'display': 'flex', 'justifyContent': 'center'})
    ]

    # --- 4. Create Histogram Figure ---
    df = pd.DataFrame(profits, columns=['Profit'])
    fig_hist = px.histogram(df, x='Profit', nbins=50, title='Profit After Tax Distribution')
    fig_hist.update_layout(bargap=0.1)

    # --- 5. Create S-Curve Figure (NEW) ---
    sorted_profits = np.sort(profits)
    y_vals = np.arange(1, n_sim_int + 1) / n_sim_int  # Cumulative probability

    fig_scurve = px.line(x=sorted_profits, y=y_vals, title='Cumulative Probability (S-Curve)')
    fig_scurve.update_layout(
        xaxis_title='Profit After Tax',
        yaxis_title='Cumulative Probability',
        yaxis_tickformat='.0%'  # Format Y-axis as percentage
    )

    return stats_children, fig_hist, fig_scurve


# --- RUN THE APP ---

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
