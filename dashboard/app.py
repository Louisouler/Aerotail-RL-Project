#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# app.py - AeroTail live monitoring dashboard (Plotly Dash).
#
# This process never calls the AirSim API, never loads a model and never
# touches VecNormalize/LSTM state. Its inputs are:
#   * the JSONL telemetry written by scripts/10_dashboard_demo.py (telemetry.py);
#   * pixels of the simulator's X display, streamed as MJPEG (simview.py).
# Optional run control (control.py) starts/stops the launcher as a separate
# process; it is only enabled on a loopback host (reach it via an SSH tunnel).
#
# Time-series panels show the current episode on a fixed 0..episode_len step
# axis; finished episodes are summarised in the table.
#
# Usage:
#   /mnt/disks/data/aerotail-dashboard-venv/bin/python dashboard/app.py [--read-only]
# ---------------------------------------------------------------------------
import argparse
import math
import os
import threading
import time

from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from telemetry import TelemetryReader, DEFAULT_TELEMETRY_PATH
import control
import simview

# --- Palette (dark; reference data-viz palette, dark steps) -----------------
SURFACE = "#161615"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#262624"
AXIS = "#383835"
DRONE = "#3987e5"      # categorical slot 1
TARGET = "#d95926"     # categorical slot 2
ACTION_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]  # slots 1-4, fixed order
CRITICAL = "#d03b3b"
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

ACTION_LABELS = ["vx · forward", "vy · right", "vz · down", "yaw rate"]
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

PANEL_HEIGHTS = {"traj": 400, "track": 330, "reward": 330, "actions": 330, "flight": 330, "lstm": 330}


def style(fig, height, hover="x unified"):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, color=INK_2, size=11),
        margin=dict(l=46, r=14, t=22, b=30), showlegend=False,
        hovermode=hover, uirevision="keep",
        hoverlabel=dict(bgcolor="#222220", bordercolor=AXIS, font=dict(color="#ffffff", family=FONT)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=10),
                     showspikes=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED, size=10),
                     nticks=4)
    for a in fig.layout.annotations:  # subplot titles: small, left-aligned, muted
        a.update(x=0, xanchor="left", font=dict(size=10.5, color=MUTED, family=FONT))
    return fig


def placeholder(height, message=""):
    fig = go.Figure()
    style(fig, height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if message:
        fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=MUTED, size=12, family=FONT))
    return fig


def col(steps, key):
    return [s.get(key) for s in steps]


def line(x, y, name, color, width=1.8, **kw):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width), **kw)


def mark_collisions(fig, steps, rows):
    for s in steps:
        if s.get("collision"):
            for r in range(1, rows + 1):
                fig.add_vline(x=s.get("episode_step"), line=dict(color=CRITICAL, width=2), row=r, col=1)


def step_axis(fig, ep_len, rows):
    fig.update_xaxes(range=[0, ep_len], row=rows, col=1, title_text="episode step",
                     title_font=dict(size=10, color=MUTED))
    for r in range(1, rows):
        fig.update_xaxes(range=[0, ep_len], row=r, col=1)


# --- Figures (current episode only) ------------------------------------------
def fig_trajectory(ep):
    h = PANEL_HEIGHTS["traj"]
    pts = [s for s in ep if s.get("drone_x") is not None and s.get("target_x") is not None]
    if not pts:
        return placeholder(h, "No position data")
    fig = go.Figure()
    # NED frame: x = north (plotted up), y = east (plotted right).
    fig.add_trace(line(col(pts, "target_y"), col(pts, "target_x"), "Target", TARGET, 2,
                       hovertemplate="Target<br>E %{x:.1f} m · N %{y:.1f} m<extra></extra>"))
    fig.add_trace(line(col(pts, "drone_y"), col(pts, "drone_x"), "Drone", DRONE, 2,
                       hovertemplate="Drone<br>E %{x:.1f} m · N %{y:.1f} m<extra></extra>"))
    first, last = pts[0], pts[-1]
    fig.add_trace(go.Scatter(x=[first["drone_y"]], y=[first["drone_x"]], mode="markers", hoverinfo="skip",
                             marker=dict(size=9, color=SURFACE, line=dict(color=DRONE, width=2))))
    fig.add_trace(go.Scatter(x=[last["drone_y"], last["target_y"]], y=[last["drone_x"], last["target_x"]],
                             mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[last["target_y"]], y=[last["target_x"]], mode="markers+text",
                             text=["Target"], textposition="top center", textfont=dict(color=TARGET, size=11),
                             marker=dict(size=13, color=TARGET, symbol="diamond", line=dict(color=SURFACE, width=2)),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[last["drone_y"]], y=[last["drone_x"]], mode="markers+text",
                             text=["Drone"], textposition="bottom center", textfont=dict(color=DRONE, size=11),
                             marker=dict(size=13, color=DRONE, line=dict(color=SURFACE, width=2)),
                             hoverinfo="skip"))
    style(fig, h, hover="closest")
    fig.update_layout(uirevision=f"traj-{last.get('episode')}", margin=dict(l=46, r=14, t=12, b=40))
    fig.update_xaxes(title_text="east (m)", title_font=dict(size=10, color=MUTED), nticks=6)
    fig.update_yaxes(title_text="north (m)", title_font=dict(size=10, color=MUTED),
                     scaleanchor="x", scaleratio=1, nticks=6)
    return fig


def fig_tracking(ep, ep_len, desired):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=(f"Distance error  |d − {desired:g} m|  (m)",
                                        "Centring error  (normalised bearing)"))
    x = col(ep, "episode_step")
    fig.add_trace(line(x, col(ep, "distance_error"), "distance error (m)", DRONE,
                       fill="tozeroy", fillcolor="rgba(57,135,229,0.10)"), row=1, col=1)
    fig.add_trace(line(x, col(ep, "centring_error"), "centring error", DRONE,
                       fill="tozeroy", fillcolor="rgba(57,135,229,0.10)"), row=2, col=1)
    mark_collisions(fig, ep, 2)
    style(fig, PANEL_HEIGHTS["track"])
    fig.update_yaxes(rangemode="tozero")
    step_axis(fig, ep_len, 2)
    return fig


def fig_reward(ep, ep_len):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                        subplot_titles=("Step reward", "Cumulative episode reward"))
    x = col(ep, "episode_step")
    fig.add_trace(line(x, col(ep, "reward"), "step reward", DRONE), row=1, col=1)
    fig.add_trace(line(x, col(ep, "episode_reward"), "episode reward", DRONE), row=2, col=1)
    mark_collisions(fig, ep, 2)
    style(fig, PANEL_HEIGHTS["reward"])
    step_axis(fig, ep_len, 2)
    return fig


def fig_actions(ep, window):
    recent = ep[-window:]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=ACTION_LABELS)
    x = col(recent, "episode_step")
    for i in range(4):
        fig.add_trace(line(x, col(recent, f"action_{i}"), ACTION_LABELS[i], ACTION_COLORS[i], 1.6),
                      row=i + 1, col=1)
        fig.update_yaxes(range=[-1.2, 1.2], tickvals=[-1, 0, 1], row=i + 1, col=1)
    style(fig, PANEL_HEIGHTS["actions"])
    fig.update_layout(margin=dict(l=46, r=14, t=22, b=30))
    fig.update_xaxes(title_text="episode step", title_font=dict(size=10, color=MUTED), row=4, col=1)
    return fig


def fig_flight(ep, ep_len):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        row_heights=[0.4, 0.4, 0.2],
                        subplot_titles=("Altitude (m)", "Speed (m/s)", "Target visible"))
    x = col(ep, "episode_step")
    fig.add_trace(line(x, col(ep, "altitude"), "altitude (m)", DRONE), row=1, col=1)
    fig.add_trace(line(x, col(ep, "drone_speed"), "speed (m/s)", DRONE), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=col(ep, "visibility"), name="visible", mode="lines",
                             line=dict(color=DRONE, width=1.5, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(57,135,229,0.25)"), row=3, col=1)
    fig.update_yaxes(range=[0, 1.15], tickvals=[0, 1], ticktext=["lost", "yes"], row=3, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=2, col=1)
    mark_collisions(fig, ep, 3)
    style(fig, PANEL_HEIGHTS["flight"])
    step_axis(fig, ep_len, 3)
    return fig


def fig_lstm(ep, ep_len):
    h = PANEL_HEIGHTS["lstm"]
    vals = col(ep, "lstm_hidden_norm")
    if all(v is None for v in vals):
        return placeholder(h, "No recurrent state for this algorithm")
    fig = go.Figure(line(col(ep, "episode_step"), vals, "‖h‖", DRONE,
                         fill="tozeroy", fillcolor="rgba(57,135,229,0.10)"))
    style(fig, h)
    fig.update_layout(margin=dict(l=46, r=14, t=12, b=30))
    fig.update_xaxes(range=[0, ep_len], title_text="episode step", title_font=dict(size=10, color=MUTED))
    fig.update_yaxes(rangemode="tozero")
    return fig


# --- Status, cards, table, panels -------------------------------------------
def fmt(v, nd=2, unit=""):
    if v is None:
        return "—"
    try:
        return f"{v:.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def card(label, value, sub=None, tone=None):
    return html.Div(className="card" + (f" tone-{tone}" if tone else ""), children=[
        html.Div(label, className="card-label"),
        html.Div(value, className="card-value"),
        html.Div(sub or " ", className="card-sub"),
    ])


def compute_status(snap, stale_after):
    if not snap["file_exists"] or (snap["meta"] is None and snap["last_step"] is None):
        return "WAITING FOR TELEMETRY", "waiting", "start scripts/10_dashboard_demo.py"
    age = time.time() - snap["last_wall_time"] if snap["last_wall_time"] else None
    if snap["run_end"] is not None:
        sps = snap["run_end"].get("loop_steps_per_s")
        if snap["run_end"].get("interrupted"):
            return "RUN STOPPED", "finished", (f"stopped by user · {snap['run_end'].get('episodes_completed')} "
                                               f"episode(s) completed")
        return "RUN FINISHED", "finished", (f"{snap['run_end'].get('episodes_completed')} episode(s) · "
                                            f"{fmt(sps, 1)} steps/s")
    last_type = (snap["last_record"] or {}).get("type")
    reset_grace = stale_after * 3  # takeoff/settle during reset has no step records
    if age is not None and age > (reset_grace if last_type == "episode_reset" else stale_after):
        return "STALE DATA", "stale", f"no update for {age:.0f} s"
    if last_type == "episode_reset":
        return "RESETTING", "resetting", "episode reset / takeoff"
    if snap["last_step"] is None:
        return "STARTING", "resetting", "waiting for first step"
    return "LIVE", "live", f"updated {age:.1f} s ago" if age is not None else ""


def build_cards(snap, ep):
    meta = snap["meta"] or {}
    s = snap["last_step"] or {}
    ep_len = meta.get("episode_len_steps")
    episodes = meta.get("episodes")
    epi = s.get("episode")
    settings = meta.get("stage_settings") or {}
    stage_sub = (f"noise {settings.get('obs_noise_std')} · dropout {settings.get('detection_dropout_prob')}"
                 if settings else None)
    cards = [
        card("Algorithm", s.get("algorithm") or meta.get("algorithm") or "—", "deterministic inference"),
        card("Curriculum stage", str(s.get("stage", meta.get("stage", "—"))), stage_sub),
        card("Episode", "—" if epi is None else f"{epi + 1}" + (f" / {episodes}" if episodes else ""),
             f"seed {s.get('seed')}" if s.get("seed") is not None else None),
        card("Episode step", "—" if not s else f"{s.get('episode_step')}" + (f" / {ep_len}" if ep_len else ""),
             f"global {s.get('global_step')}" if s else None),
        card("Step reward", fmt(s.get("reward"), 3)),
        card("Episode reward", fmt(s.get("episode_reward"), 1)),
    ]
    if not s:
        cards += [card("Collision", "—"), card("Target visibility", "—")]
        return cards
    if s.get("collision"):
        cards.append(card("Collision", "Collision", s.get("collision_object") or "", "critical"))
    else:
        cards.append(card("Collision", "None", "this episode", "good"))
    ratio = (sum(x.get("visibility") or 0 for x in ep) / len(ep)) if ep else None
    sub = f"{ratio * 100:.0f}% of episode" if ratio is not None else None
    if s.get("visibility"):
        cards.append(card("Target visibility", "Visible", sub, "good"))
    else:
        cards.append(card("Target visibility", "Lost", sub, "warning"))
    return cards


def build_episode_table(episodes):
    if not episodes:
        return html.Div("No completed episodes yet.", className="empty")
    head = ["Ep", "Seed", "Steps", "Reward", "Centring err", "Dist err (m)", "Visibility", "Smoothness", "Collision"]
    rows = []
    for e in list(episodes)[-8:][::-1]:
        rows.append(html.Tr([
            html.Td(e.get("episode", 0) + 1), html.Td(e.get("seed")), html.Td(e.get("steps")),
            html.Td(fmt(e.get("total_reward"), 1)), html.Td(fmt(e.get("mean_centring_error"), 3)),
            html.Td(fmt(e.get("mean_distance_error"), 2)),
            html.Td(fmt((e.get("visibility_ratio") or 0) * 100, 0, "%")),
            html.Td(fmt(e.get("mean_smoothness_penalty"), 3)),
            html.Td(html.Span("Yes" if e.get("collided") else "No",
                              className="tag tag-bad" if e.get("collided") else "tag tag-ok")),
        ]))
    return html.Table(className="ep-table", children=[
        html.Thead(html.Tr([html.Th(h) for h in head])), html.Tbody(rows)])


def panel(title, meta_id, graph_id, span, height):
    return html.Div(className=f"panel {span}", children=[
        html.Div(className="panel-head", children=[
            html.Span(title, className="panel-title"), html.Span(id=meta_id, className="panel-meta")]),
        dcc.Graph(id=graph_id, config=GRAPH_CONFIG, figure=placeholder(height),
                  style={"height": f"{height}px"}),
    ])


def sim_panel(sim):
    if sim is not None and sim.available():
        body = html.Img(src="/sim.mjpg", className="sim-img", alt="simulator view",
                        title="Double-click for fullscreen")
        tools = html.Div(className="sim-tools", children=[
            html.Span(f"display {sim.display}", className="panel-meta sim-meta"),
            # behaviour lives in assets/simview.js (client-side only)
            html.Button("⤢ Enlarge", className="sim-btn sim-btn-expand", title="Enlarge within the dashboard"),
            html.Button("⛶ Fullscreen", className="sim-btn sim-btn-full", title="Fullscreen (Esc to exit)"),
        ])
    else:
        body = html.Div("Simulator view unavailable (no X display to capture).", className="sim-empty")
        tools = html.Span("", className="panel-meta")
    return html.Div(className="panel span-7 sim-panel", children=[
        html.Div(className="panel-head", children=[html.Span("Simulator view", className="panel-title"), tools]),
        html.Div(body, className="sim-wrap", style={"height": f"{PANEL_HEIGHTS['traj']}px"}),
    ])


# --- Run control --------------------------------------------------------------
def control_bar():
    models = control.available_models()
    default_model = models[0][0] if models else None
    return html.Div(className="control-bar", children=[
        html.Div(className="ctl-field ctl-model", children=[
            html.Label("Model", className="ctl-label"),
            dcc.Dropdown(id="ctl-model", options=[{"label": l, "value": v} for v, l in models],
                         value=default_model, clearable=False, searchable=False)]),
        html.Div(className="ctl-field ctl-vehicle", children=[
            html.Label("Vehicle", className="ctl-label"),
            dcc.Dropdown(id="ctl-vehicle", clearable=False, searchable=False, value="Drone1", options=[
                {"label": "Drone1 → Target1 (sim camera)", "value": "Drone1"},
                {"label": "Drone2 → Target2 (eval pair)", "value": "Drone2"}])]),
        html.Div(className="ctl-field ctl-episodes", children=[
            html.Label("Episodes", className="ctl-label"),
            dcc.Input(id="ctl-episodes", type="number", min=1, max=20, step=1, value=2,
                      className="ctl-input")]),
        html.Div(className="ctl-buttons", children=[
            html.Button("Start", id="ctl-start", className="btn btn-start", n_clicks=0),
            html.Button("Stop", id="ctl-stop", className="btn btn-stop", n_clicks=0)]),
        html.Div(className="ctl-info", children=[
            html.Div(id="ctl-state", className="ctl-state"),
            html.Div(id="ctl-msg", className="ctl-msg")]),
    ])


def register_control_callbacks(app, controller):
    @app.callback(Output("ctl-msg", "children"),
                  Input("ctl-start", "n_clicks"), Input("ctl-stop", "n_clicks"),
                  State("ctl-model", "value"), State("ctl-vehicle", "value"), State("ctl-episodes", "value"),
                  prevent_initial_call=True)
    def on_control(_start, _stop, model, vehicle, episodes):
        if ctx.triggered_id == "ctl-start":
            ok, msg = controller.start(model, vehicle, episodes)
        else:
            ok, msg = controller.stop()
        return html.Span(msg, className="ok" if ok else "err")

    @app.callback(Output("ctl-state", "children"), Output("ctl-start", "disabled"),
                  Output("ctl-stop", "disabled"), Input("tick", "n_intervals"))
    def on_tick(_n):
        st = controller.status()
        if st["running"]:
            return html.Span(f"Demo running · pid {st['pid']}", className="run"), True, False
        return html.Span("Idle · no demo running", className="idle"), False, True


# --- App --------------------------------------------------------------------
def create_app(reader, refresh_ms, stale_after, action_window=100, enable_control=False, sim=None,
               sim_hd=None):
    app = Dash(__name__, title="AeroTail Monitor", update_title=None,
               external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap"])
    app.layout = html.Div(className="page", children=[
        html.Header(className="header", children=[
            html.Div([
                html.Div([html.Span("AeroTail", className="brand"),
                          html.Span("UAV Target-Tracking Monitor", className="brand-sub")]),
                html.Div(id="run-info", className="run-info"),
            ]),
            html.Div(id="status", className="status"),
        ]),
        control_bar() if enable_control else html.Div(),
        html.Div(id="cards", className="cards"),
        html.Div(className="grid", children=[
            sim_panel(sim),
            panel("Trajectory · top-down", "m-traj", "g-traj", "span-5", PANEL_HEIGHTS["traj"]),
            panel("Tracking error", "m-track", "g-track", "span-4", PANEL_HEIGHTS["track"]),
            panel("Reward", "m-reward", "g-reward", "span-4", PANEL_HEIGHTS["reward"]),
            panel("Flight state", "m-flight", "g-flight", "span-4", PANEL_HEIGHTS["flight"]),
            panel("Policy actions", "m-actions", "g-actions", "span-6", PANEL_HEIGHTS["actions"]),
            panel("Recurrent state · LSTM ‖h‖", "m-lstm", "g-lstm", "span-6", PANEL_HEIGHTS["lstm"]),
            html.Div(className="panel span-12", children=[
                html.Div(className="panel-head", children=[
                    html.Span("Completed episodes", className="panel-title"),
                    html.Span("metric definitions as in 07_evaluate_all.py", className="panel-meta")]),
                html.Div(id="ep-table", className="table-wrap"),
            ]),
        ]),
        html.Footer(id="footer", className="footer"),
        dcc.Interval(id="tick", interval=refresh_ms, n_intervals=0),
        dcc.Store(id="data-version"),
    ])

    if enable_control:
        register_control_callbacks(app, control.DemoController())
    if sim is not None:
        simview.register(app.server, sim, sim_hd)

    graph_ids = ["g-traj", "g-track", "g-reward", "g-actions", "g-flight", "g-lstm"]
    meta_ids = ["m-traj", "m-track", "m-reward", "m-actions", "m-flight", "m-lstm"]
    # Heavy outputs (cards, figures, table) are rebuilt only when new telemetry
    # has been consumed, and the result is shared by every open browser tab.
    cache = {"version": None, "outputs": None}
    cache_lock = threading.Lock()

    @app.callback(
        Output("status", "children"), Output("run-info", "children"), Output("footer", "children"),
        Input("tick", "n_intervals"),
    )
    def refresh_status(_n):
        reader.poll()
        snap = reader.snapshot()
        label, tone, detail = compute_status(snap, stale_after)
        status = [html.Span(label, className=f"pill pill-{tone}"), html.Span(detail, className="status-detail")]
        meta = snap["meta"] or {}
        if meta:
            run_info = " · ".join(str(p) for p in [
                meta.get("algorithm"), f"Stage {meta.get('stage')}",
                f"{meta.get('vehicle')} → {meta.get('target')}", _short(meta.get("model_path"))] if p)
        else:
            run_info = "No run loaded"
        mode = "Monitor + run control" if enable_control else "Read-only monitor"
        footer = (f"{mode} · source {reader.path} · refresh {refresh_ms} ms"
                  + (f" · {snap['bad_lines']} malformed line(s) skipped" if snap["bad_lines"] else ""))
        return status, run_info, footer

    @app.callback(
        Output("cards", "children"),
        *[Output(g, "figure") for g in graph_ids], *[Output(m, "children") for m in meta_ids],
        Output("ep-table", "children"), Output("data-version", "data"),
        Input("tick", "n_intervals"), State("data-version", "data"),
    )
    def refresh_data(_n, client_version):
        reader.poll()
        snap = reader.snapshot()
        version = snap["version"]
        if version == client_version:
            raise PreventUpdate  # this tab already shows the latest data
        with cache_lock:
            if cache["version"] != version:
                cache["outputs"] = build_data_outputs(snap)
                cache["version"] = version
            outputs = cache["outputs"]
        return (*outputs, version)

    def build_data_outputs(snap):
        meta = snap["meta"] or {}
        last = snap["last_step"]
        ep = [s for s in snap["steps"] if last and s.get("episode") == last.get("episode")]
        ep_len = meta.get("episode_len_steps") or 300
        if not ep:
            _label, tone, _detail = compute_status(snap, stale_after)
            msg = "Waiting for telemetry" if tone == "waiting" else "Waiting for first step"
            figs = [placeholder(PANEL_HEIGHTS[k], msg) for k in PANEL_HEIGHTS]
            metas = [""] * 6
        else:
            figs = [fig_trajectory(ep), fig_tracking(ep, ep_len, meta.get("desired_distance", 8.0)),
                    fig_reward(ep, ep_len), fig_actions(ep, action_window), fig_flight(ep, ep_len),
                    fig_lstm(ep, ep_len)]
            sep = None
            if last.get("drone_x") is not None and last.get("target_x") is not None:
                sep = math.dist((last["drone_x"], last["drone_y"]), (last["target_x"], last["target_y"]))
            metas = [
                f"horizontal separation {fmt(sep, 1, ' m')}",
                f"distance {fmt(last.get('distance_error'), 2, ' m')} · centring {fmt(last.get('centring_error'), 2)}",
                f"step {fmt(last.get('reward'), 2)} · total {fmt(last.get('episode_reward'), 1)}",
                f"last {action_window} steps · normalised [−1, 1]",
                f"{fmt(last.get('altitude'), 1, ' m')} · {fmt(last.get('drone_speed'), 1, ' m/s')}",
                f"‖h‖ {fmt(last.get('lstm_hidden_norm'), 2)}" if last.get("lstm_hidden_norm") is not None else "n/a",
            ]
        return (build_cards(snap, ep), *figs, *metas, build_episode_table(snap["episodes"]))

    return app


def _short(path):
    if not path:
        return None
    return os.sep.join(path.split(os.sep)[-2:])


def main():
    p = argparse.ArgumentParser(description="AeroTail live monitoring dashboard.")
    p.add_argument("--telemetry-path", default=DEFAULT_TELEMETRY_PATH)
    p.add_argument("--host", default="127.0.0.1",
                   help="a non-loopback host (e.g. 0.0.0.0) disables run control")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--history", type=int, default=600,
                   help="recent step records kept in memory (must exceed the episode length)")
    p.add_argument("--refresh-ms", type=int, default=1000)
    p.add_argument("--action-window", type=int, default=100,
                   help="recent steps shown in the policy-actions panel")
    p.add_argument("--stale-after", type=float, default=15.0,
                   help="seconds without new records before showing STALE DATA")
    p.add_argument("--read-only", action="store_true", help="hide the start/stop control bar")
    p.add_argument("--sim-display", default=":99", help="X display the simulator window renders on")
    p.add_argument("--sim-fps", type=int, default=30, help="frame rate of the simulator view")
    p.add_argument("--sim-width", type=int, default=854, help="stream width in pixels")
    p.add_argument("--sim-hd-width", type=int, default=1280,
                   help="stream width used while the view is enlarged or fullscreen (~1 MB/s at 30 fps)")
    p.add_argument("--sim-hd-fps", type=int, default=30, help="frame rate of the enlarged/fullscreen stream")
    p.add_argument("--no-sim-view", action="store_true", help="disable the simulator view panel")
    args = p.parse_args()
    enable_control = not args.read_only
    if enable_control and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"--host {args.host} is not loopback: run control disabled (read-only dashboard).")
        enable_control = False
    reader = TelemetryReader(args.telemetry_path, history=args.history)
    sim = sim_hd = None
    if not args.no_sim_view:
        sim = simview.SimView(args.sim_display, args.sim_fps, args.sim_width)
        sim_hd = simview.SimView(args.sim_display, args.sim_hd_fps, args.sim_hd_width)
    app = create_app(reader, args.refresh_ms, args.stale_after, args.action_window, enable_control, sim, sim_hd)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
