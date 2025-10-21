import random
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# --- NEON TECH THEME DEFINITIONS ---
NEON_PURPLE = "#BE00FE"
NEON_BLUE = "#00BFFF"
TEXT_SHADOW_BLUE = f"0 0 5px {NEON_BLUE}, 0 0 10px {NEON_BLUE}, 0 0 15px {NEON_BLUE}"
BOX_SHADOW_PURPLE = f"0 0 5px {NEON_PURPLE}, 0 0 10px {NEON_PURPLE}, 0 0 15px {NEON_PURPLE}"
PLOTLY_TEMPLATE = "plotly_dark"
FONT_FAMILY = "'Orbitron', sans-serif"

# --- APP INITIALIZATION with CYBORG Theme and Google Font ---
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap"
    ]
)
server = app.server

# --- Reusable Input Group Component with Neon Styling ---
def make_input_group(label, input_id, input_type='number', **kwargs):
    """Creates a neatly formatted input group with a label."""
    return dbc.Row([
        dbc.Label(label, width=5, style={'textAlign': 'right', 'color': NEON_BLUE}),
        dbc.Col(dcc.Input(id=input_id, type=input_type, className="neon-input", **kwargs), width=7)
    ], className="mb-3")

# --- APP LAYOUT ---
app.layout = html.Div(style={'fontFamily': FONT_FAMILY}, children=[
    # Inline CSS for custom neon styles
    html.Style("""
        .neon-input {
            background-color: #1a1a1a;
            border: 1px solid #00BFFF;
            color: #00BFFF;
            border-radius: 5px;
            padding: 5px;
            width: 100%;
        }
        .neon-input:focus {
            outline: none;
            box-shadow: 0 0 5px #00BFFF, 0 0 10px #00BFFF;
        }
        .card-header-neon {
            color: #BE00FE;
            text-shadow: 0 0 5px #BE00FE, 0 0 10px #BE00FE;
            border-bottom: 1px solid #BE00FE;
        }
        .nav-tabs .nav-link.active {
            background-color: #2a2a2a !important;
            border-bottom: 1px solid #BE00FE !important;
            color: #BE00FE !important;
            text-shadow: 0 0 5px #BE00FE;
        }
        .card-neon {
            border: 1px solid #BE00FE;
            box-shadow: """ + BOX_SHADOW_PURPLE + """;
        }
    """),

    dcc.Loading(
        id="loading-fullscreen",
        type="cube",
        fullscreen=True,
        color=NEON_PURPLE,
        children=html.Div(id="loading-output-dummy")
    ),
    dbc.Container([
        dbc.NavbarSimple(
            brand="Profit Simulation & Analysis [Neon Tech]",
            brand_href="#",
            color="dark",
            className="mb-4",
            style={'borderBottom': f'1px solid {NEON_PURPLE}', 'boxShadow': BOX_SHADOW_PURPLE}
        ),
        dbc.Row([
            # --- INPUTS COLUMN ---
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H4("Simulation Parameters", className="card-title"), className="card-header-neon"),
                    dbc.CardBody([
                        html.H5("Sales Price (Uniform)", className="mt-3", style={'color': NEON_BLUE}),
                        make_input_group("Min Price:", 'sp_min', value=7.2, step=0.1),
                        make_input_group("Max Price:", 'sp_max', value=8.1, step=0.1),

                        html.H5("Cost of Sales (Normal)", className="mt-4", style={'color': NEON_BLUE}),
                        make_input_group("Mean Cost:", 'cost_mean', value=4.8, step=0.1),
                        make_input_group("Std Dev Cost:", 'cost_std', value=0.75, step=0.1),

                        html.H5("Sales Volume (Triangular)", className="mt-4", style={'color': NEON_BLUE}),
                        make_input_group("Min Volume:", 'vol_min', value=20000, step=1000),
                        make_input_group("Mode Volume:", 'vol_mode', value=40000, step=1000),
                        make_input_group("Max Volume:", 'vol_max', value=60000, step=1000),

                        html.H5("Simulations", className="mt-4", style={'color': NEON_BLUE}),
                        make_input_group("Number:", 'n_sim', value=10000, step=100),

                        html.Div(
                            dbc.Button('Run Simulation', id='run_btn', n_clicks=0, color="primary", size="lg",
                                       style={'boxShadow': TEXT_SHADOW_BLUE, 'border': f'1px solid {NEON_BLUE}'}),
                            className="d-grid gap-2 mt-4"
                        )
                    ])
                ], className="card-neon"),
                width=12, lg=4
            ),

            # --- OUTPUTS COLUMN (Tabs for different analyses) ---
            dbc.Col(
                dbc.Tabs([
                    dbc.Tab(label="Summary & Distributions", children=[
                        dbc.Card(dbc.CardBody(id='results-output'), className="card-neon mt-3")
                    ]),
                    dbc.Tab(label="Sensitivity Analysis", children=[
                        dbc.Card(dbc.CardBody(id='sensitivity-output', children=[
                                    html.P("Run a simulation first to enable sensitivity analysis.", className="text-muted")
                                ]), className="card-neon mt-3")
                    ]),
                    dbc.Tab(label="Data Table", children=[
                        dbc.Card(dbc.CardBody(id='data-table-output', children=[
                                    html.P("Run a simulation to view a sample of the generated data.", className="text-muted")
                                ]), className="card-neon mt-3")
                    ])
                ]),
                width=12, lg=8, className="mt-4 mt-lg-0"
            )
        ]),
        dcc.Store(id='simulation-data')
    ], fluid=True, className="py-4")
])


# --- Main Callback to Run Simulation and store data ---
@callback(
    [Output('simulation-data', 'data'),
     Output('loading-output-dummy', 'children')],
    Input('run_btn', 'n_clicks'),
    [State('sp_min', 'value'), State('sp_max', 'value'),
     State('cost_mean', 'value'), State('cost_std', 'value'),
     State('vol_min', 'value'), State('vol_mode', 'value'),
     State('vol_max', 'value'), State('n_sim', 'value')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, sp_min, sp_max, cost_mean, cost_std, vol_min, vol_mode, vol_max, n_sim):
    n_sim_int = int(n_sim)
    sim_data = []
    for i in range(n_sim_int):
        sales_price = random.uniform(sp_min, sp_max)
        cost = random.gauss(cost_mean, cost_std)
        sales_volume = random.triangular(vol_min, vol_max, vol_mode)
        tax_rate = np.random.choice([0, 0.3], p=[0.8, 0.2])
        profit = (sales_price - cost) * sales_volume * (1 - tax_rate)
        sim_data.append({
            'Run': i + 1,
            'Sales Price': sales_price,
            'Cost': cost,
            'Sales Volume': sales_volume,
            'Profit': profit
        })
    df = pd.DataFrame(sim_data)
    return df.to_json(date_format='iso', orient='split'), ""


# --- Callback to display main results ---
@callback(
    Output('results-output', 'children'),
    Input('simulation-data', 'data')
)
def update_results_display(jsonified_data):
    if jsonified_data is None:
        return html.Div("Click 'Run Simulation' to see the results.", className="text-center text-muted mt-4")

    df = pd.read_json(jsonified_data, orient='split')
    profits = df['Profit']

    mean_val, median_val, std_val = profits.mean(), profits.median(), profits.std()
    min_val, max_val = profits.min(), profits.max()
    p10, p25, p75, p90 = np.percentile(profits, [10, 25, 75, 90])
    prob_loss = (profits < 0).mean() * 100

    stats_output = dbc.Row([
        dbc.Col([
            html.H5("Key Metrics", className="card-header-neon mb-3"),
            html.P(f"Mean Profit: ${mean_val:,.2f}"),
            html.P(f"Median Profit: ${median_val:,.2f}"),
            html.P(f"Std. Deviation: ${std_val:,.2f}"),
            html.P(f"Probability of Loss: {prob_loss:.2f}%", style={'color': NEON_PURPLE, 'fontWeight': 'bold'}),
        ], width=12, md=4),
        dbc.Col([
            html.H5("Percentiles", className="card-header-neon mb-3"),
            html.P(f"10th: ${p10:,.2f}"),
            html.P(f"25th: ${p25:,.2f}"),
            html.P(f"75th: ${p75:,.2f}"),
            html.P(f"90th: ${p90:,.2f}"),
        ], width=12, md=4),
        dbc.Col([
            html.H5("Range", className="card-header-neon mb-3"),
            html.P(f"Minimum: ${min_val:,.2f}"),
            html.P(f"Maximum: ${max_val:,.2f}"),
        ], width=12, md=4)
    ], className="mb-4")

    fig_hist = px.histogram(df, x='Profit', nbins=50, title='Profit Distribution', template=PLOTLY_TEMPLATE, color_discrete_sequence=[NEON_BLUE])
    fig_hist.update_layout(bargap=0.1, title_x=0.5, font_family=FONT_FAMILY)
    fig_hist.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color=NEON_PURPLE, annotation_text="Mean")

    sorted_profits = np.sort(profits)
    y_vals = np.arange(1, len(profits) + 1) / len(profits)
    fig_scurve = px.line(x=sorted_profits, y=y_vals, title='Cumulative Probability (S-Curve)', template=PLOTLY_TEMPLATE, color_discrete_sequence=[NEON_BLUE])
    fig_scurve.update_layout(
        xaxis_title='Profit After Tax', yaxis_title='Cumulative Probability',
        yaxis_tickformat='.0%', title_x=0.5, font_family=FONT_FAMILY
    )

    graphs_output = dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_hist), width=12, lg=6),
        dbc.Col(dcc.Graph(figure=fig_scurve), width=12, lg=6)
    ])
    return [stats_output, html.Hr(style={'borderColor': NEON_PURPLE, 'borderWidth': '2px'}), graphs_output]

# --- Callback for Sensitivity Analysis ---
@callback(
    Output('sensitivity-output', 'children'),
    Input('simulation-data', 'data'),
    [State('sp_min', 'value'), State('sp_max', 'value'),
     State('cost_mean', 'value'), State('vol_mode', 'value')]
)
def update_sensitivity_analysis(jsonified_data, sp_min, sp_max, base_cost, base_vol):
    if jsonified_data is None:
        return html.P("Run a simulation first to enable sensitivity analysis.", className="text-muted")

    base_sp = (sp_min + sp_max) / 2
    def calculate_single_profit(price, cost, volume):
        return (price - cost) * volume * (1 - np.random.choice([0, 0.3], p=[0.8, 0.2]))

    # Vary each variable +/- 20%
    price_range = np.linspace(base_sp * 0.8, base_sp * 1.2, 20)
    price_profits = [calculate_single_profit(p, base_cost, base_vol) for p in price_range]

    cost_range = np.linspace(base_cost * 0.8, base_cost * 1.2, 20)
    cost_profits = [calculate_single_profit(base_sp, c, base_vol) for c in cost_range]

    volume_range = np.linspace(base_vol * 0.8, base_vol * 1.2, 20)
    volume_profits = [calculate_single_profit(base_sp, base_cost, v) for v in volume_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=(price_range / base_sp - 1), y=price_profits, mode='lines', name='Sales Price', line=dict(color=NEON_BLUE)))
    fig.add_trace(go.Scatter(x=(cost_range / base_cost - 1), y=cost_profits, mode='lines', name='Cost of Sales', line=dict(color=NEON_PURPLE)))
    fig.add_trace(go.Scatter(x=(volume_range / base_vol - 1), y=volume_profits, mode='lines', name='Sales Volume', line=dict(color='#39FF14')))

    fig.update_layout(
        title="Profit Sensitivity to Key Inputs", xaxis_title="Variable Change from Base", yaxis_title="Estimated Profit",
        template=PLOTLY_TEMPLATE, title_x=0.5, font_family=FONT_FAMILY, xaxis_tickformat='.0%'
    )
    return dcc.Graph(figure=fig)

# --- Callback for Data Table ---
@callback(
    Output('data-table-output', 'children'),
    Input('simulation-data', 'data')
)
def update_data_table(jsonified_data):
    if jsonified_data is None:
        return html.P("Run a simulation to view a sample of the generated data.", className="text-muted")

    df = pd.read_json(jsonified_data, orient='split')
    # Formatting columns for better readability
    for col in ['Sales Price', 'Cost', 'Profit']:
        df[col] = df[col].map('{:,.2f}'.format)
    df['Sales Volume'] = df['Sales Volume'].map('{:,.0f}'.format)
    return dbc.Table.from_dataframe(df.head(100), striped=True, bordered=True, hover=True, responsive=True, dark=True)

# --- RUN THE APP ---
if __name__ == "__main__":
    app.run(debug=True)

