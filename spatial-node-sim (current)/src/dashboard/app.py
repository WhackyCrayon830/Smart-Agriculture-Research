"""
Enhanced Agricultural Market (Mandi) Network Simulation Dashboard

This dashboard simulates realistic price dynamics in a network of Indian agricultural markets,
incorporating:
- Weather-driven supply shocks
- Seasonal harvest patterns
- Market arbitrage and price convergence
- Transportation costs and friction
- Storage dynamics
- Festival demand cycles

The simulation can generate synthetic datasets for training predictive models.
"""

import time
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from datetime import datetime

from src.region.region3d import Region3D
from src.simulation.node import Node
from src.graph.neighbor_graph import build_knn_graph, analyze_network
from src.field.weather_field_3d import MarketWeatherField
from src.blockchain.mandi_blockchain import MandiBlockchain, PriceSmartContract

app = Dash(__name__)

# Global singletons
region = Region3D(100, 100, 100, seed=42)
field = MarketWeatherField(seed=42, monsoon_strength=0.8)
blockchain = MandiBlockchain(block_time=5.0, difficulty=2)  # New block every 5 seconds
smart_contracts = PriceSmartContract(blockchain)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_control_card(title, children, icon="", color_start="#667eea", color_end="#764ba2"):
    """Modern card component for controls"""
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px", "marginRight": "8px"}),
            html.H3(title, style={"margin": "0", "fontSize": "16px", "fontWeight": "600"})
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),
        html.Div(children)
    ], style={
        "background": f"linear-gradient(135deg, {color_start} 0%, {color_end} 100%)",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "16px",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.1)",
        "color": "white"
    })


def param_slider(label, slider_id, input_id, default, min_val=0, max_val=1, step=0.05):
    """Unified parameter control with modern styling"""
    return html.Div([
        html.Div([
            html.Label(label, style={"fontWeight": "500", "fontSize": "13px"}),
            dcc.Input(
                id=input_id,
                type="number",
                value=default,
                step=step,
                min=min_val,
                max=max_val,
                style={
                    "width": "70px",
                    "padding": "4px 8px",
                    "borderRadius": "6px",
                    "border": "1px solid rgba(255,255,255,0.3)",
                    "background": "rgba(255,255,255,0.1)",
                    "color": "white",
                    "fontSize": "13px"
                }
            )
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px"}),
        dcc.Slider(
            min=min_val,
            max=max_val,
            step=step,
            value=default,
            id=slider_id,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], style={"marginBottom": "16px"})


def action_button(text, btn_id, color="#10b981", icon=""):
    """Modern action button"""
    return html.Button([
        html.Span(icon, style={"marginRight": "6px"} if icon else {}),
        text
    ], id=btn_id, style={
        "background": color,
        "color": "white",
        "border": "none",
        "borderRadius": "8px",
        "padding": "10px 20px",
        "fontSize": "14px",
        "fontWeight": "600",
        "cursor": "pointer",
        "transition": "all 0.2s",
        "width": "100%",
        "marginBottom": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
    })

# ============================================================================
# LAYOUT
# ============================================================================

app.layout = html.Div([
    # Data stores
    dcc.Store(id="sim-store", data={}),
    dcc.Store(id="dataset-store", data={"running": False}),
    
    # Timers
    dcc.Interval(id="sim-timer", interval=500, disabled=True, n_intervals=0, max_intervals=-1),
    dcc.Interval(id="dataset-timer", interval=50, disabled=True, n_intervals=0, max_intervals=-1),
    
    # Sidebar
    html.Div([
        html.Div([
            html.H1("🌾 Mandi Network", style={
                "margin": "0 0 4px 0",
                "fontSize": "28px",
                "fontWeight": "700",
                "background": "linear-gradient(135deg, #059669 0%, #10b981 100%)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent"
            }),
            html.P("Agricultural Market Simulation", style={
                "margin": "0",
                "color": "#6b7280",
                "fontSize": "13px"
            })
        ], style={"marginBottom": "24px"}),
        
        # Simulation Controls
        create_control_card("⚙️ Simulation", [
            html.Label("Markets (Mandis)", style={"fontSize": "13px", "fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Slider(5, 100, 5, value=30, id="num-nodes", marks=None, 
                      tooltip={"placement": "bottom", "always_visible": False}),
            
            html.Div(style={"height": "16px"}),
            
            html.P("Price Dynamics Parameters:", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "8px"}),
            param_slider("α (Memory/Stickiness)", "alpha-slider", "alpha-input", 0.5, 0, 1),
            param_slider("β (Weather Effect)", "beta-slider", "beta-input", 0.3, 0, 1),
            param_slider("γ (Arbitrage)", "gamma-slider", "gamma-input", 0.15, 0, 0.5),
            param_slider("δ (Supply/Demand)", "delta-slider", "delta-input", 0.05, 0, 0.3),
            
            html.Label("Time (Days)", style={"fontSize": "13px", "fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Slider(0, 365, 1, value=0, id="time-slider", marks=None,
                      tooltip={"placement": "bottom", "always_visible": True}),
            
            html.Div(style={"height": "16px"}),
            
            html.Div([
                action_button("▶ Play", "play-btn", "#10b981", "▶"),
                action_button("⏸ Pause", "pause-btn", "#f59e0b", "⏸"),
            ]),
            
            dcc.Checklist(
                options=[
                    {"label": " Show Weather Field", "value": "show"},
                    {"label": " Show Price Gradients", "value": "gradient"}
                ],
                value=[],
                id="show-field",
                style={"fontSize": "13px"}
            )
        ], color_start="#059669", color_end="#10b981"),
        
        # Weather Status
        create_control_card("☁️ Weather Status", [
            html.Pre(
                id="weather-info",
                style={
                    "fontSize": "12px",
                    "margin": "0",
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.5"
                }
            )
        ], color_start="#0ea5e9", color_end="#06b6d4"),
        
        # Blockchain Status
        create_control_card("⛓️ Blockchain", [
            html.Pre(
                id="blockchain-info",
                style={
                    "fontSize": "11px",
                    "margin": "0",
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.6"
                }
            ),
            html.Div(style={"height": "8px"}),
            html.Div([
                html.Label("Price Oracle", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "4px", "display": "block"}),
                html.Pre(
                    id="price-oracle",
                    style={
                        "fontSize": "11px",
                        "background": "rgba(255,255,255,0.1)",
                        "padding": "8px",
                        "borderRadius": "6px",
                        "margin": "0"
                    }
                )
            ])
        ], color_start="#7c3aed", color_end="#a855f7"),
        
        # Dataset Generator
        create_control_card("📊 Dataset Generator", [
            html.Label("Dataset Size", style={"fontSize": "13px", "fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Dropdown(
                id="dataset-size",
                options=[
                    {"label": "🔬 Tiny (10k rows, 100 days)", "value": 100},
                    {"label": "📈 Small (100k rows, 1k days)", "value": 1000},
                    {"label": "🚀 Medium (1M rows, 10k days)", "value": 10000},
                    {"label": "🌌 Large (10M rows, 100k days)", "value": 100000},
                ],
                value=1000,
                clearable=False,
                style={"marginBottom": "12px", "color": "#000"}
            ),
            
            html.Label("Batch Size (rows/tick)", style={"fontSize": "13px", "fontWeight": "500", "marginBottom": "8px", "display": "block"}),
            dcc.Input(
                id="batch-size",
                type="number",
                value=100,
                min=1,
                max=1000,
                style={
                    "width": "100%",
                    "padding": "8px",
                    "borderRadius": "6px",
                    "border": "1px solid rgba(255,255,255,0.3)",
                    "background": "rgba(255,255,255,0.1)",
                    "color": "white",
                    "marginBottom": "12px"
                }
            ),
            
            dcc.Checklist(
                id="multi-world",
                options=[{"label": " Multi-World (randomize topology)", "value": "multi"}],
                value=[],
                style={"fontSize": "13px", "marginBottom": "12px"}
            ),
            
            action_button("🚀 Generate Dataset", "gen-dataset", "#8b5cf6"),
            
            html.Div([
                html.Div([
                    html.Div(
                        id="dataset-progress-bar",
                        style={
                            "width": "0%",
                            "height": "8px",
                            "background": "linear-gradient(90deg, #10b981, #3b82f6)",
                            "borderRadius": "4px",
                            "transition": "width 0.3s ease"
                        }
                    )
                ], style={
                    "width": "100%",
                    "height": "8px",
                    "background": "rgba(0,0,0,0.2)",
                    "borderRadius": "4px",
                    "marginBottom": "8px",
                    "overflow": "hidden"
                }),
                html.Div(id="dataset-status", style={"fontSize": "12px", "marginBottom": "4px"}),
                html.Div(id="dataset-eta", style={"fontSize": "11px", "opacity": "0.8"}),
                html.Pre(
                    id="dataset-current",
                    style={
                        "fontSize": "11px",
                        "background": "rgba(0,0,0,0.2)",
                        "padding": "8px",
                        "borderRadius": "4px",
                        "marginTop": "8px",
                        "maxHeight": "60px",
                        "overflowY": "auto"
                    }
                )
            ])
        ], color_start="#8b5cf6", color_end="#a855f7"),
        
        # Market Info
        html.Div([
            html.H3("📍 Selected Market", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "12px"}),
            html.Pre(
                id="node-info",
                style={
                    "fontSize": "11px",
                    "background": "#f3f4f6",
                    "padding": "12px",
                    "borderRadius": "8px",
                    "border": "1px solid #e5e7eb",
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.6"
                }
            )
        ], style={"marginBottom": "16px"}),
        
        # Trade Connections
        html.Div([
            html.H3("🔗 Trade Connections", style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "12px"}),
            html.Pre(
                id="conn-info",
                style={
                    "fontSize": "11px",
                    "background": "#f3f4f6",
                    "padding": "12px",
                    "borderRadius": "8px",
                    "border": "1px solid #e5e7eb",
                    "height": "180px",
                    "overflowY": "scroll",
                    "whiteSpace": "pre-wrap"
                }
            )
        ])
        
    ], style={
        "width": "360px",
        "height": "100vh",
        "overflowY": "auto",
        "padding": "20px",
        "background": "#fafafa",
        "borderRight": "1px solid #e5e7eb"
    }),
    
    # Main plot area
    html.Div([
        dcc.Graph(
            id="region-plot",
            style={"height": "100vh", "width": "100%"},
            config={"scrollZoom": True, "displayModeBar": True}
        )
    ], style={"flex": "1", "background": "#ffffff"})
    
], style={
    "display": "flex",
    "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "margin": "0",
    "padding": "0",
    "height": "100vh",
    "overflow": "hidden"
})

# ============================================================================
# CALLBACKS: SYNC SLIDERS
# ============================================================================

def make_sync_callback(slider_id, input_id):
    @app.callback(
        Output(slider_id, "value"),
        Output(input_id, "value"),
        Input(slider_id, "value"),
        Input(input_id, "value"),
    )
    def _sync(slider_val, input_val):
        trig = ctx.triggered_id
        if trig == input_id:
            return input_val, input_val
        return slider_val, slider_val

make_sync_callback("alpha-slider", "alpha-input")
make_sync_callback("beta-slider", "beta-input")
make_sync_callback("gamma-slider", "gamma-input")
make_sync_callback("delta-slider", "delta-input")

# ============================================================================
# CALLBACKS: PLAY/PAUSE
# ============================================================================

@app.callback(
    Output("sim-timer", "disabled"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    State("dataset-store", "data"),
    prevent_initial_call=True
)
def control_sim_timer(p, s, ds_store):
    """Pause simulation during dataset generation"""
    if ds_store and ds_store.get("running"):
        return True
    return ctx.triggered_id != "play-btn"

# ============================================================================
# CALLBACKS: DATASET GENERATION START
# ============================================================================

@app.callback(
    Output("dataset-store", "data", allow_duplicate=True),
    Output("dataset-progress-bar", "style", allow_duplicate=True),
    Output("dataset-status", "children", allow_duplicate=True),
    Output("dataset-timer", "disabled", allow_duplicate=True),
    Output("sim-timer", "disabled", allow_duplicate=True),
    Input("gen-dataset", "n_clicks"),
    State("dataset-size", "value"),
    State("batch-size", "value"),
    State("multi-world", "value"),
    State("num-nodes", "value"),
    State("sim-store", "data"),
    prevent_initial_call=True
)
def start_dataset(n_clicks, steps, batch_size, multi_world, num_nodes, sim_store):
    """Initialize dataset generation"""    
    if not steps or steps < 1:
        return no_update, no_update, "Invalid parameters", True, no_update
    
    total_steps = int(steps)
    batch_size = max(1, int(batch_size or 100))
    
    # Initialize topology
    if not sim_store or "coords" not in sim_store:
        coords = region.sample_nodes(num_nodes)
        nodes = [Node(i, tuple(p), base_price=100.0 + np.random.uniform(-20, 20)) 
                 for i, p in enumerate(coords)]
        graph = build_knn_graph(nodes, k=3)
        sim_store = {
            "coords": coords.tolist(),
            "graph": {str(k): list(v) for k, v in graph.items()}
        }
    else:
        coords = np.array(sim_store["coords"])
        nodes = [Node(i, tuple(p), base_price=100.0 + np.random.uniform(-20, 20)) 
                 for i, p in enumerate(coords)]
        if "graph" not in sim_store:
            graph = build_knn_graph(nodes, k=3)
            sim_store["graph"] = {str(k): list(v) for k, v in graph.items()}
    
    # Create CSV with enhanced header
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"data/mandi_dataset_{timestamp}.csv"
    
    header = pd.DataFrame(columns=[
        "world_id", "time", "node_id", "x", "y", "z",
        "price_state", "actual_price", "weather_effect",
        "supply", "demand", "storage",
        "neighbor_mean_price", "degree", 
        "volatility", "price_trend",
        "blockchain_price", "price_consensus", "on_chain"
    ])
    header.to_csv(csv_path, index=False)
    
    # Initialize state
    ds = {
        "running": True,
        "t": 0,
        "T": total_steps,
        "batch_size": batch_size,
        "multi_world": "multi" in multi_world,
        "world_id": 0,
        "start_ts": time.time(),
        "csv_path": csv_path,
        "total_rows": 0,
        "sim_store_snapshot": sim_store
    }
    
    total_rows = total_steps * len(nodes)
    progress_style = {
        "width": "0%",
        "height": "8px",
        "background": "linear-gradient(90deg, #10b981, #3b82f6)",
        "borderRadius": "4px",
        "transition": "width 0.3s ease"
    }    
    return ds, progress_style, f"Starting: {total_rows:,} rows", False, True

# ============================================================================
# CALLBACKS: DATASET GENERATION STEP
# ============================================================================

@app.callback(
    Output("dataset-store", "data", allow_duplicate=True),
    Output("dataset-progress-bar", "style", allow_duplicate=True),
    Output("dataset-status", "children", allow_duplicate=True),
    Output("dataset-eta", "children"),
    Output("dataset-current", "children"),
    Output("dataset-timer", "disabled", allow_duplicate=True),
    Output("sim-timer", "disabled", allow_duplicate=True),
    Input("dataset-timer", "n_intervals"),
    Input("dataset-store", "data"),
    prevent_initial_call=True
)
def dataset_step(n_intervals, ds):
    """Process batch of timesteps for dataset generation"""
    
    if ctx.triggered_id != "dataset-timer":
        raise PreventUpdate
    
    if not ds or not ds.get("running"):
        return ds, no_update, no_update, no_update, no_update, True, no_update
    
    # Restore topology
    sim_store = ds["sim_store_snapshot"]
    coords = np.array(sim_store["coords"])
    nodes = [Node(i, tuple(p), base_price=100.0 + np.random.uniform(-20, 20)) 
             for i, p in enumerate(coords)]
    graph = sim_store["graph"]
    
    # Set up neighbors with distances
    positions = coords
    for i, neigh_ids in graph.items():
        i_int = int(i)
        nodes[i_int].neighbors = neigh_ids
        # Calculate distances to neighbors
        dists = [np.linalg.norm(positions[i_int] - positions[j]) for j in neigh_ids]
        nodes[i_int].neighbor_distances = dists
    
    t = ds["t"]
    T = ds["T"]
    batch_size = ds["batch_size"]
    world_id = ds["world_id"]
    
    # Generate batch
    rows = []
    for _ in range(batch_size):
        if t >= T:
            break
        
        # Multi-world: regenerate topology periodically
        if ds["multi_world"] and t > 0 and t % 365 == 0:  # Yearly reset
            coords = region.sample_nodes(len(nodes))
            nodes = [Node(i, tuple(p), base_price=100.0 + np.random.uniform(-20, 20)) 
                     for i, p in enumerate(coords)]
            graph = build_knn_graph(nodes, k=3)
            positions = coords
            for i, neigh_ids in graph.items():
                i_int = int(i)
                nodes[i_int].neighbors = neigh_ids
                dists = [np.linalg.norm(positions[i_int] - positions[j]) for j in neigh_ids]
                nodes[i_int].neighbor_distances = dists
            world_id += 1
        
        # Update all nodes
        new_states = []
        for n in nodes:
            # Get weather effect
            weather_val = field.value(np.array(n.position), t)
            
            # Update supply/demand based on weather
            n.update_supply_demand(weather_val, t)
            
            # Get neighbor states and distances
            if n.neighbors:
                neighbor_states = [nodes[j].state for j in n.neighbors]
                neighbor_dists = n.neighbor_distances if hasattr(n, 'neighbor_distances') else [50] * len(n.neighbors)
            else:
                neighbor_states = []
                neighbor_dists = []
            
            # Calculate new state
            new_state = n.update_state(
                weather_val,
                neighbor_states,
                neighbor_dists,
                alpha=0.5,  # Memory
                beta=0.3,   # Weather
                gamma=0.15, # Arbitrage
                delta=0.05  # Supply/demand
            )
            new_states.append(new_state)
        
        # Apply all state updates
        for i, n in enumerate(nodes):
            n.state = new_states[i]
        
        # Blockchain: Submit transactions every 10 timesteps
        if t % 10 == 0:
            # Update validator stakes based on trading volume
            volumes = {i: nodes[i].demand * (1.0 - abs(nodes[i].storage - 0.5)) 
                      for i in range(len(nodes))}
            blockchain.update_validator_stakes(volumes)
            
            # Each mandi submits price
            for n in nodes:
                n.create_blockchain_transaction(blockchain)
            
            # Try to create block
            blockchain.create_block(max_transactions=100)
        
        # Get blockchain oracle price
        oracle = blockchain.get_price_oracle(commodity="wheat", lookback_blocks=5)
        blockchain_price = oracle['average_price'] if oracle else 0.0
        price_consensus = oracle['unique_mandis'] >= 3 if oracle else False
        
        # Record data
        for n in nodes:
            neighbor_mean = np.mean([nodes[j].state for j in n.neighbors]) if n.neighbors else 0.0
            
            rows.append({
                "world_id": world_id,
                "time": t,
                "node_id": n.id,
                "x": n.position[0],
                "y": n.position[1],
                "z": n.position[2],
                "price_state": n.state,
                "actual_price": n.actual_price,
                "weather_effect": field.value(np.array(n.position), t),
                "supply": n.supply,
                "demand": n.demand,
                "storage": n.storage,
                "neighbor_mean_price": neighbor_mean,
                "degree": len(n.neighbors),
                "volatility": n.volatility,
                "price_trend": n.state - n.price_history[0],
                "blockchain_price": blockchain_price,
                "price_consensus": 1 if price_consensus else 0,
                "on_chain": 1 if t % 10 == 0 else 0  # Transaction submitted
            })
        
        t += 1
    
    # Write batch
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(ds["csv_path"], mode="a", header=False, index=False)
        ds["total_rows"] += len(rows)
    
    # Update progress
    ds["t"] = t
    ds["world_id"] = world_id
    
    done = t
    total = T
    progress_pct = int(100 * done / total) if total > 0 else 0
    
    progress_style = {
        "width": f"{progress_pct}%",
        "height": "8px",
        "background": "linear-gradient(90deg, #10b981, #3b82f6)",
        "borderRadius": "4px",
        "transition": "width 0.3s ease"
    }
    
    # ETA
    elapsed = time.time() - ds["start_ts"]
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    
    eta_text = f"⏱ {remaining:.1f}s remaining | {rate:.1f} days/s | {ds['total_rows']:,} rows"
    status_text = f"Day {done:,} / {total:,} ({progress_pct}%)"
    current_text = f"World {world_id} | Day {t} | {len(rows)} rows/batch"
    
    # Completion check
    if t >= T:
        ds["running"] = False
        final_status = f"✅ Complete: {ds['csv_path']}"
        final_eta = f"Generated {ds['total_rows']:,} rows in {elapsed:.1f}s"
        complete_style = {
            "width": "100%",
            "height": "8px",
            "background": "linear-gradient(90deg, #10b981, #3b82f6)",
            "borderRadius": "4px"
        }
        return ds, complete_style, final_status, final_eta, "Done", True, False
    
    return ds, progress_style, status_text, eta_text, current_text, False, True

# ============================================================================
# CALLBACKS: MAIN VISUALIZATION
# ============================================================================

@app.callback(
    Output("region-plot", "figure"),
    Output("conn-info", "children"),
    Output("node-info", "children"),
    Output("weather-info", "children"),
    Output("blockchain-info", "children"),
    Output("price-oracle", "children"),
    Output("sim-store", "data"),
    Input("num-nodes", "value"),
    Input("sim-timer", "n_intervals"),
    Input("time-slider", "value"),
    Input("region-plot", "clickData"),
    Input("show-field", "value"),
    Input("alpha-slider", "value"),
    Input("beta-slider", "value"),
    Input("gamma-slider", "value"),
    Input("delta-slider", "value"),
    State("sim-store", "data"),
    State("dataset-store", "data"),
)
def update_viz(num_nodes, n_intervals, t_slider, clickData, show_field, 
               alpha, beta, gamma, delta, store, ds_store):
    """Main visualization update"""
    
    # Freeze during dataset generation
    if ds_store and ds_store.get("running"):
        return no_update, no_update, no_update, no_update, no_update, no_update, store
    
    if store is None:
        store = {}
    
    # Time source
    t = n_intervals if ctx.triggered_id == "sim-timer" else t_slider
    
    # Initialize nodes
    if "coords" not in store or len(store["coords"]) != num_nodes:
        coords = region.sample_nodes(num_nodes)
        store["coords"] = coords.tolist()
        store.pop("graph", None)
    else:
        coords = np.array(store["coords"])
    
    nodes = [Node(i, tuple(pos), base_price=100.0 + np.random.uniform(-20, 20)) 
             for i, pos in enumerate(coords)]
    
    # Build/restore graph
    if "graph" not in store:
        graph = build_knn_graph(nodes, k=3)
        store["graph"] = {str(k): list(v) for k, v in graph.items()}
    else:
        graph = store["graph"]
    
    # Set neighbors with distances
    for i, neigh_ids in graph.items():
        i_int = int(i)
        nodes[i_int].neighbors = neigh_ids
        dists = [np.linalg.norm(coords[i_int] - coords[j]) for j in neigh_ids]
        nodes[i_int].neighbor_distances = dists
    
    # State update
    new_states = []
    for n in nodes:
        weather_val = field.value(np.array(n.position), t)
        n.update_supply_demand(weather_val, t)
        
        if n.neighbors:
            neighbor_states = [nodes[j].state for j in n.neighbors]
            neighbor_dists = n.neighbor_distances
        else:
            neighbor_states = []
            neighbor_dists = []
        
        new_state = n.update_state(
            weather_val, neighbor_states, neighbor_dists,
            alpha, beta, gamma, delta
        )
        new_states.append(new_state)
    
    for i, n in enumerate(nodes):
        n.state = new_states[i]
    
    # Blockchain integration: Submit transactions periodically
    if t % 10 == 0:  # Every 10 time steps
        # Update validator stakes
        volumes = {i: nodes[i].demand * (1.0 - abs(nodes[i].storage - 0.5)) 
                  for i in range(len(nodes))}
        blockchain.update_validator_stakes(volumes)
        
        # Submit price transactions
        for n in nodes:
            n.create_blockchain_transaction(blockchain)
        
        # Try to mine a block
        new_block = blockchain.create_block(max_transactions=100)
        if new_block:
            store["last_block"] = new_block.index
    
    # Visualization
    xs = [n.position[0] for n in nodes]
    ys = [n.position[1] for n in nodes]
    zs = [n.position[2] for n in nodes]
    prices = [n.actual_price for n in nodes]
    
    node_scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=8,
            color=prices,
            colorscale="RdYlGn_r",  # Red=high, Green=low
            showscale=True,
            colorbar=dict(title="Price<br>(₹/quintal)", thickness=15, len=0.7),
            cmin=min(prices) - 5,
            cmax=max(prices) + 5
        ),
        customdata=[n.id for n in nodes],
        hovertemplate=(
            "<b>Mandi %{customdata}</b><br>"
            "Price: ₹%{marker.color:.1f}<br>"
            "<extra></extra>"
        )
    )
    
    # Edges
    edge_traces = []
    conn_text = []
    for i, n in enumerate(nodes):
        for j_idx, j in enumerate(n.neighbors):
            p1 = coords[i]
            p2 = coords[j]
            d = n.neighbor_distances[j_idx] if hasattr(n, 'neighbor_distances') else np.linalg.norm(p1 - p2)
            
            # Price difference determines edge color
            price_diff = abs(nodes[j].state - n.state)
            strength = np.exp(-price_diff * 5)  # Similar prices = strong connection
            
            edge_traces.append(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode='lines',
                line=dict(
                    width=1 + 3 * strength,
                    color=f"rgba(100, 100, 100, {0.2 + 0.5 * strength})"
                ),
                showlegend=False,
                hoverinfo='none'
            ))
            
            conn_text.append(
                f"M{i:2d} ↔ M{j:2d} | d={d:5.1f} | "
                f"ΔP={nodes[j].actual_price - n.actual_price:+6.2f}"
            )
    
    data = [node_scatter] + edge_traces
    
    # Optional weather field
    if "show" in show_field:
        gx = np.linspace(0, region.Lx, 8)
        gy = np.linspace(0, region.Ly, 8)
        gz = np.linspace(0, region.Lz, 8)
        X, Y, Z = np.meshgrid(gx, gy, gz)
        pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        vals = np.array([field.value(p, t) for p in pts])
        
        data.append(go.Volume(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            value=vals,
            opacity=0.15,
            surface_count=10,
            colorscale="Blues",
            showscale=False
        ))
    
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, region.Lx], showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            yaxis=dict(range=[0, region.Ly], showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            zaxis=dict(range=[0, region.Lz], showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
            aspectmode="cube",
            bgcolor="rgba(250,250,245,1)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        uirevision="keep",
        title=dict(
            text=f"<b>🌾 Mandi Network Simulation</b> | Day {t} of Year {t % 365}",
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        )
    )
    
    # Weather status
    weather_status = field.get_status(t)
    weather_text = (
        f"Season: {weather_status['season']}\n"
        f"Day {weather_status['day_of_year']}/365\n"
        f"Monsoon: {weather_status['monsoon_intensity']:.1%}\n"
        f"Temp Effect: {weather_status['temperature_effect']:+.2f}\n"
    )
    if weather_status['extreme_event']:
        weather_text += f"\n⚠️ {weather_status['extreme_event'].upper()}\n"
        weather_text += f"   ({weather_status['extreme_days_remaining']} days left)"
    
    # Node click info
    node_text = "Click a market to inspect"
    if clickData and "customdata" in clickData["points"][0]:
        nid = clickData["points"][0]["customdata"]
        n = nodes[nid]
        info = n.get_market_info()
        node_text = (
            f"🏪 Mandi ID: {info['id']}\n"
            f"📍 Location: ({info['position'][0]:.1f}, {info['position'][1]:.1f})\n\n"
            f"💰 Price: ₹{info['price']:.2f}/quintal\n"
            f"   State: {info['state']:+.3f}\n"
            f"   Trend (5d): {info['price_trend']:+.3f}\n"
            f"   Volatility: {info['volatility']:.3f}\n\n"
            f"📊 Market:\n"
            f"   Supply: {info['supply']:.2f}x\n"
            f"   Demand: {info['demand']:.2f}x\n"
            f"   Balance: {info['market_balance']:+.2f}\n"
            f"   Storage: {info['storage']:.1%}\n\n"
            f"🔗 Connections: {info['num_connections']}"
        )
    
    store["t"] = t
    
    # Blockchain status
    stats = blockchain.get_network_statistics()
    blockchain_text = (
        f"⛓️ Chain Length: {stats['total_blocks']} blocks\n"
        f"📝 Pending TX: {stats['pending_transactions']}\n"
        f"✅ Valid Chain: {'Yes' if stats['chain_valid'] else 'No'}\n"
        f"📊 Total TX: {stats['total_transactions']}\n"
        f"⏱️ Block Time: {stats['block_time']}s\n"
        f"📈 Avg TX/Block: {stats['average_tx_per_block']:.1f}"
    )
    
    # Price oracle
    oracle = blockchain.get_price_oracle(commodity="wheat", lookback_blocks=5)
    if oracle:
        oracle_text = (
            f"💰 Avg Price: ₹{oracle['average_price']:.2f}\n"
            f"📊 Range: ₹{oracle['min_price']:.2f} - ₹{oracle['max_price']:.2f}\n"
            f"📈 Std Dev: ₹{oracle['std_price']:.2f}\n"
            f"🏪 Mandis: {oracle['unique_mandis']}/{oracle['num_reports']}\n"
            f"📦 Volume: {oracle['total_volume']:.2f}"
        )
    else:
        oracle_text = "Waiting for data...\n(Need 3+ mandis reporting)"
    
    return fig, "\n".join(conn_text[:30]), node_text, weather_text, blockchain_text, oracle_text, store


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🌾 Mandi Network Simulation Dashboard")
    print("="*60)
    print("Starting server at http://localhost:8050")
    print("\nThis simulation models:")
    print("  • Realistic weather patterns (monsoons, temperature)")
    print("  • Market price dynamics with arbitrage")
    print("  • Supply/demand from harvest cycles & festivals")
    print("  • Storage and transportation effects")
    print("="*60)
    app.run(debug=True, host="0.0.0.0", port=8050)