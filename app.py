"""
Market Panic Detector — Streamlit dashboard.

    streamlit run app.py

Layout
  Sidebar  : Time Machine (date scrubber, step, play) · Network filter · Advanced · Guide
  Header   : title + live status badge
  Banner   : collapsible "How to use this dashboard"
  Presets  : one-click jumps to calm / crash episodes
  KPI row  : λ₁, mean correlation, herd share, tight links, drawdown (hover ? for meaning)
  Tabs     : Overview (network + timeline) · Heatmap · Strongest Links · Function & Metric Guide
"""

from __future__ import annotations

import html


import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import panic_detector as pdx

st.set_page_config(page_title="Market Panic Detector", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================== design tokens
PAGE, CARD, RAISED = "#0a0d12", "#11161e", "#171e29"
BORDER = "rgba(255,255,255,0.08)"
INK, INK_2, MUTED = "#f0efec", "#c3c2b7", "#8a8f98"
GRID, AXIS = "#1f2630", "#2e3846"
ACCENT = "#3987e5"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"

GROUP_ORDER = [
    "Tech & Media", "Consumer Discretionary", "Consumer Staples", "Health Care",
    "Financials", "Industrials & Materials", "Energy", "Utilities & Real Estate",
]
# Validated categorical palette (dark-surface steps), assigned in fixed slot order.
GROUP_COLOR = dict(zip(GROUP_ORDER, [
    "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767",
]))

REGIME = {
    "Calm": dict(label="Calm · Dispersed", color=GOOD, icon="●",
                 blurb="Stocks are moving on their own news. Diversification is working."),
    "Elevated": dict(label="Elevated Stress", color=WARN, icon="▲",
                     blurb="A common driver is taking hold. Sector boundaries are blurring."),
    "Panic": dict(label="Panic · High Clustering", color=CRIT, icon="◆",
                  blurb="The market is moving as one herd. Diversification offers little shelter."),
}
LOOKBACKS = {"5 years": 5, "7 years (includes COVID)": 7}
STEP_SIZES = {"1 day": 1, "1 week": 5, "1 month": 21}
SPEEDS = {"Slow": 0.9, "Normal": 0.45, "Fast": 0.15}


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


PLOT_BASE = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
    margin=dict(l=8, r=8, t=12, b=8),
    hoverlabel=dict(bgcolor=RAISED, bordercolor=AXIS, font=dict(color=INK, size=12)),
)

# ===================================================================== CSS
st.markdown(f"""
<style>
.stApp {{ background: {PAGE}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px; }}
[data-testid="stSidebar"] {{ background: #0d1118; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
p, li {{ line-height: 1.55; }}

.eyebrow {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {MUTED}; font-weight: 650; }}
.sb-h {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {MUTED};
         font-weight: 650; margin: 18px 0 6px; display: flex; gap: 8px; align-items: center; }}
.sb-brand {{ font-size: 17px; font-weight: 700; color: {INK}; }}
.sb-brand span {{ color: {MUTED}; font-weight: 500; font-size: 12.5px; display: block; margin-top: 2px; }}
.sb-note {{ font-size: 12.5px; color: {MUTED}; line-height: 1.5; margin-top: 2px; }}

.hdr {{ display: flex; justify-content: space-between; align-items: center; gap: 24px; margin-bottom: 18px; }}
.hdr > div:first-child {{ flex: 1 1 auto; min-width: 0; }}
.hdr > .badge-wrap {{ flex: 0 0 auto; }}
@media (max-width: 900px) {{ .hdr {{ flex-wrap: wrap; }} .badge-wrap {{ text-align: left; }} }}
.hdr h1 {{ font-size: 32px; font-weight: 750; margin: 4px 0 0; padding: 0; color: {INK}; letter-spacing: -0.015em; }}
.hdr .sub {{ color: {INK_2}; font-size: 14.5px; margin-top: 6px; max-width: 760px; }}
.badge-wrap {{ text-align: right; }}
.badge {{ display: inline-flex; align-items: center; gap: 10px; padding: 11px 18px; border-radius: 999px;
          border: 1px solid; font-weight: 700; font-size: 16px; color: {INK}; }}
.badge .ic {{ font-size: 14px; }}
.badge-date {{ font-size: 12.5px; color: {MUTED}; margin-top: 8px; }}

.kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin: 4px 0 14px; }}
@media (max-width: 1150px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px 15px; }}
.kpi-label {{ display: flex; align-items: center; gap: 7px; font-size: 11px; letter-spacing: .12em;
              text-transform: uppercase; color: {MUTED}; font-weight: 650; }}
.kpi-value {{ font-size: 31px; font-weight: 750; color: {INK}; margin-top: 10px; line-height: 1.05;
              font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }}
.kpi-value small {{ font-size: 15px; color: {MUTED}; font-weight: 600; }}
.kpi-sub {{ font-size: 12.5px; color: {INK_2}; margin-top: 8px; line-height: 1.4; }}
.up {{ color: {WARN}; }} .down {{ color: {GOOD}; }} .flat {{ color: {MUTED}; }}

.tip {{ position: relative; display: inline-flex; width: 16px; height: 16px; border-radius: 50%;
        border: 1px solid {AXIS}; color: {MUTED}; font-size: 10px; align-items: center; justify-content: center;
        cursor: help; text-transform: none; letter-spacing: 0; font-weight: 700; }}
.tip:hover {{ border-color: {ACCENT}; color: {INK}; }}
.tip:hover::after {{ content: attr(data-tip); position: absolute; top: 24px; left: -10px; width: 270px;
        background: {RAISED}; color: {INK}; border: 1px solid {AXIS}; padding: 11px 13px; border-radius: 10px;
        font-size: 12.5px; line-height: 1.5; font-weight: 400; z-index: 9999; white-space: normal;
        box-shadow: 0 12px 32px rgba(0,0,0,.5); }}
.tip.r:hover::after {{ left: auto; right: -10px; }}

.narr {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 3px solid; border-radius: 14px;
         padding: 15px 20px; color: {INK_2}; font-size: 14.5px; line-height: 1.6; margin-bottom: 18px; }}
.narr b {{ color: {INK}; }}
.narr .t {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {MUTED}; font-weight: 650; margin-bottom: 4px; }}

[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px 6px; }}
.card-h {{ font-size: 16px; font-weight: 700; color: {INK}; margin: 2px 0 0; }}
.card-s {{ font-size: 12.5px; color: {MUTED}; margin-top: 3px; }}
.legend-note {{ font-size: 12px; color: {MUTED}; padding: 2px 2px 10px; }}

.st-key-presets {{ margin: 2px 0 16px; }}
.st-key-presets button {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; min-height: 46px;
                          color: {INK}; font-weight: 600; transition: border-color .15s, background .15s; }}
.st-key-presets button:hover {{ border-color: {ACCENT}; background: {RAISED}; color: #fff; }}
.st-key-presets button:disabled {{ opacity: .45; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 16px; font-weight: 650; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
[data-testid="stExpander"] summary {{ font-weight: 650; }}

.steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 12px; }}
@media (max-width: 1100px) {{ .steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.step {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; }}
.step .n {{ color: {ACCENT}; font-weight: 750; font-size: 11px; letter-spacing: .14em; }}
.step .h {{ color: {INK}; font-weight: 700; font-size: 14.5px; margin: 6px 0 4px; }}
.step .b {{ color: {INK_2}; font-size: 13px; line-height: 1.5; }}

.guide {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
@media (max-width: 1000px) {{ .guide {{ grid-template-columns: 1fr; }} }}
.g {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 18px 20px; }}
.g h4 {{ margin: 0 0 8px; padding: 0; font-size: 16px; color: {INK}; font-weight: 700; }}
.g p, .g li {{ color: {INK_2}; font-size: 13.5px; line-height: 1.6; margin: 0 0 8px; }}
.g ul {{ margin: 0 0 4px 0; padding-left: 18px; }}
.g b {{ color: {INK}; }}
.g .kicker {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; font-weight: 700; margin-bottom: 6px; }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 650; color: {INK}; border: 1px solid; }}
</style>
""", unsafe_allow_html=True)


def tip(text: str, right: bool = False) -> str:
    return f'<span class="tip{" r" if right else ""}" data-tip="{html.escape(text, quote=True)}">?</span>'


def oneline(s: str) -> str:
    """Collapse whitespace so Markdown never mistakes indented HTML for a code block."""
    return " ".join(s.split())


# ================================================================ cached data
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_prices(years: int, refresh: int) -> pd.DataFrame:
    return pdx.fetch_prices(years=years, refresh=refresh > 0)


@st.cache_resource(show_spinner=False, max_entries=6)
def load_risk(years: int, window: int, refresh: int):
    px = load_prices(years, refresh)
    res = pdx.rolling_risk(pdx.log_returns(px), window=window)
    return res, pdx.equal_weight_drawdown(px)


@st.cache_data(show_spinner=False, max_entries=2048)
def force_positions(years: int, window: int, refresh: int, k: int) -> dict[str, tuple[float, float]]:
    """Kamada-Kawai on correlation distance d = sqrt(2(1-ρ)) — every pair shapes the layout."""
    res, _ = load_risk(years, window, refresh)
    C = res.corr[k].astype(np.float64)
    D = np.sqrt(np.clip(2 * (1 - C), 0, None))
    t = res.tickers
    dist = {a: {b: float(D[i, j]) for j, b in enumerate(t)} for i, a in enumerate(t)}
    pos = nx.kamada_kawai_layout(nx.complete_graph(t), dist=dist, pos=ring_layout(tuple(t)))
    return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}


@st.cache_data(show_spinner=False)
def ring_layout(tickers: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Nodes on a circle grouped by sector, with a gap between groups."""
    grp = lambda t: pdx.SECTOR_GROUP[pdx.TICKER_SECTOR[t]]
    ordered = sorted(tickers, key=lambda t: (GROUP_ORDER.index(grp(t)), t))
    slots, pos_idx, prev = 0.0, {}, None
    for t in ordered:
        if prev is not None and grp(t) != prev:
            slots += 1.5
        pos_idx[t] = slots
        slots += 1
        prev = grp(t)
    total = slots + 1.5
    return {t: np.array([np.cos(2 * np.pi * s / total + np.pi / 2),
                         np.sin(2 * np.pi * s / total + np.pi / 2)]) for t, s in pos_idx.items()}


# ===================================================================== figures
def network_figure(C, tickers, thr, pos, labels_outside: bool) -> tuple[go.Figure, nx.Graph]:
    G = nx.Graph()
    G.add_nodes_from(tickers)
    iu, ju = np.where(np.triu(C, k=1) > thr)
    for i, j in zip(iu, ju):
        G.add_edge(tickers[i], tickers[j], weight=float(C[i, j]))

    fig = go.Figure()
    span = max(1.0 - thr, 1e-6)
    bins = [(thr, thr + span / 3, 0.6, 0.20), (thr + span / 3, thr + 2 * span / 3, 1.1, 0.38),
            (thr + 2 * span / 3, 1.01, 1.8, 0.62)]
    for lo, hi, width, alpha in bins:
        xs, ys = [], []
        for u, v, d in G.edges(data=True):
            if lo < d["weight"] <= hi:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(width=width, color=rgba(INK_2, alpha))))

    idx = {t: i for i, t in enumerate(tickers)}
    for g in GROUP_ORDER:
        nodes = [t for t in tickers if pdx.SECTOR_GROUP[pdx.TICKER_SECTOR[t]] == g]
        if not nodes:
            continue
        deg = [G.degree(t) for t in nodes]
        avg_c = [(C[idx[t]].sum() - 1) / (len(tickers) - 1) for t in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[t][0] for t in nodes], y=[pos[t][1] for t in nodes],
            mode="markers" if labels_outside else "markers+text",
            text=nodes, textposition="top center", textfont=dict(size=9.5, color=INK_2),
            name=g,
            marker=dict(size=[9 + 0.9 * d for d in deg], color=GROUP_COLOR[g], line=dict(width=2, color=CARD)),
            customdata=np.stack([[pdx.TICKER_SECTOR[t] for t in nodes], deg, avg_c], axis=-1),
            hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>"
                          "Strong links: %{customdata[1]}<br>"
                          "Avg. correlation to the other stocks: %{customdata[2]:.2f}<extra></extra>",
        ))
    if labels_outside:
        fig.add_trace(go.Scatter(
            x=[pos[t][0] * 1.13 for t in tickers], y=[pos[t][1] * 1.13 for t in tickers],
            mode="text", text=tickers, textfont=dict(size=9.5, color=INK_2),
            hoverinfo="skip", showlegend=False))
    fig.update_layout(
        **PLOT_BASE, height=560,
        xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="top", y=-0.01, x=0, font=dict(size=11.5, color=INK_2)),
    )
    return fig, G


def panic_spans(regime: pd.Series, level: str):
    flag = (regime == level).to_numpy()
    spans, start, prev = [], None, None
    for d, f in zip(regime.index, flag):
        if f and start is None:
            start = d
        if not f and start is not None:
            spans.append((start, prev))
            start = None
        prev = d
    if start is not None:
        spans.append((start, regime.index[-1]))
    return spans


# Line traces can't be click-selected in Plotly; invisible markers make every day clickable.
CLICKABLE = dict(mode="lines+markers", marker=dict(size=7, opacity=0),
                 selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)))


def timeline_figure(m: pd.DataFrame, dd: pd.Series, sel: pd.Timestamp, n: int,
                    markers: dict[str, pd.Timestamp]) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.4, 0.33, 0.27],
        subplot_titles=(f"Herd strength λ₁  ·  1 = every stock independent, {n} = perfect lockstep",
                        "Average correlation between stocks",
                        "Equal-weight basket: distance below previous high"),
    )
    fig.add_trace(go.Scatter(x=m.index, y=m["lambda1"], name="λ₁", line=dict(color=ACCENT, width=2), **CLICKABLE,
                             customdata=m["lambda1_share"],
                             hovertemplate="λ₁ %{y:.1f}  (herd share %{customdata:.0%})<extra></extra>"),
                  row=1, col=1)
    if markers:
        md = [d for d in markers.values()]
        fig.add_trace(go.Scatter(
            x=md, y=m.loc[md, "lambda1"], mode="markers+text", text=list(markers.keys()),
            textposition="top center", textfont=dict(size=10, color=MUTED),
            marker=dict(symbol="diamond", size=8, color=INK_2, line=dict(width=1.5, color=CARD)),
            hovertemplate="%{text}<extra></extra>", cliponaxis=False,
            unselected=dict(marker=dict(opacity=1), textfont=dict(color=MUTED))), row=1, col=1)
    fig.add_trace(go.Scatter(x=m.index, y=m["mean_corr"], name="Avg ρ", line=dict(color="#d95926", width=2), **CLICKABLE,
                             hovertemplate="Avg correlation %{y:.2f}<extra></extra>"), row=2, col=1)
    for lvl, y in pdx.REGIME_BANDS.items():
        fig.add_hline(y=y, line=dict(color=REGIME[lvl]["color"], width=1, dash="dot"), row=2, col=1,
                      annotation_text=REGIME[lvl]["label"], annotation_position="top right",
                      annotation_font=dict(size=10, color=MUTED))
    d = dd.reindex(m.index)
    fig.add_trace(go.Scatter(x=d.index, y=d, name="Drawdown", fill="tozeroy", **CLICKABLE,
                             line=dict(color=MUTED, width=1.5), fillcolor=rgba(MUTED, 0.16),
                             hovertemplate="%{y:.1%} below high<extra></extra>"), row=3, col=1)
    # Stress-regime shading + selected-date line on all three panels. Shapes are built as plain
    # dicts and set in one assignment: add_vrect(row="all") is ~50x slower and stalls playback.
    axes = [("x", "y"), ("x2", "y2"), ("x3", "y3")]
    shapes = []
    for lvl, a in (("Elevated", 0.06), ("Panic", 0.13)):
        for s, e in panic_spans(m["regime"], lvl):
            shapes += [dict(type="rect", xref=xr, yref=f"{yr} domain", x0=s, x1=e + pd.Timedelta(days=1),
                            y0=0, y1=1, fillcolor=REGIME[lvl]["color"], opacity=a, line_width=0, layer="below")
                       for xr, yr in axes]
    shapes += [dict(type="line", xref=xr, yref=f"{yr} domain", x0=sel, x1=sel, y0=0, y1=1,
                    line=dict(color=INK, width=1.5)) for xr, yr in axes]
    fig.update_layout(shapes=list(fig.layout.shapes) + shapes)
    fig.update_layout(**PLOT_BASE, height=560, showlegend=False, hovermode="x unified", dragmode=False)
    fig.update_layout(margin=dict(l=8, r=8, t=28, b=8))
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED),
                     showspikes=True, spikecolor=AXIS, spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return fig


def heatmap_figure(C, tickers) -> go.Figure:
    grp = lambda t: GROUP_ORDER.index(pdx.SECTOR_GROUP[pdx.TICKER_SECTOR[t]])
    order = sorted(range(len(tickers)), key=lambda i: (grp(tickers[i]), tickers[i]))
    labels = [tickers[i] for i in order]
    fig = go.Figure(go.Heatmap(
        z=C[np.ix_(order, order)], x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, "#1c5cab"], [0.3, "#3f6fa6"], [0.5, "#262c36"], [0.7, "#a8504e"], [1, "#e66767"]],
        colorbar=dict(title=dict(text="ρ", font=dict(color=INK_2)), thickness=12, tickfont=dict(color=MUTED)),
        hovertemplate="%{y} × %{x}: ρ = %{z:.2f}<extra></extra>", xgap=1, ygap=1,
    ))
    fig.update_layout(**PLOT_BASE, height=640)
    fig.update_xaxes(tickfont=dict(size=9, color=MUTED), showgrid=False)
    fig.update_yaxes(tickfont=dict(size=9, color=MUTED), autorange="reversed", showgrid=False)
    return fig


# ============================================================ guide content
GUIDE = [
    ("Network graph", "🕸 Network graph", """
<p>Each <b>dot is one stock</b> (50 in total), coloured by sector. A <b>line between two dots</b> means
those two stocks have been moving up and down together strongly over the past 60 trading days
(their correlation is above the threshold you set). Thicker, brighter lines = tighter links.
Bigger dots = stocks with more strong links.</p>
<ul><li><b>Calm markets look spread out.</b> Few lines, mostly inside a sector: oil stocks move with oil
stocks, banks with banks, because each group reacts to its own news.</li>
<li><b>Crashes pull everything into one tight cluster.</b> When fear hits, investors sell almost everything at
once, so unrelated stocks suddenly move together and lines appear across every sector.</li></ul>
<p>Hover over any dot to see its sector and how many strong links it has.</p>"""),
    ("λ₁ (top eigenvalue)", "📈 Herd strength — top eigenvalue λ₁", """
<p>Imagine one invisible force, “the market mood”, pushing all 50 stocks at once. <b>λ₁ measures how strong
that single shared force is.</b> It is the top eigenvalue of the correlation matrix, which finds the one
pattern that explains the most movement across all stocks.</p>
<ul><li><b>Low (≈ 7–12):</b> stocks are driven mainly by their own stories. Calm, dispersed market.</li>
<li><b>High (≈ 20+):</b> one herd factor dominates; stocks rise and fall together. Typical of sell-offs.</li>
<li>The scale runs from <b>1</b> (every stock fully independent) to <b>50</b> (all stocks identical).</li></ul>
<p><b>Herd share</b> = λ₁ ÷ 50: the share of all price movement explained by that one factor.</p>"""),
    ("Average correlation", "🔗 Average pairwise correlation", """
<p>Take every possible pair of the 50 stocks (1,225 pairs), measure how closely each pair moves together
(−1 = opposite, 0 = unrelated, +1 = perfectly in sync), then average them.</p>
<ul><li><b>Why it matters:</b> diversification only protects you when your holdings <i>don't</i> all fall at
the same time. The higher this number, the less owning many different stocks actually helps.</li>
<li><b>Below 0.25</b> → Calm · Dispersed. <b>0.25–0.40</b> → Elevated Stress. <b>0.40+</b> → Panic · High Clustering.</li></ul>"""),
    ("Status badge", "🚦 Status badge & presets", """
<p>The badge at the top-right reads the average correlation on the selected date and labels it:</p>
<ul><li><span class="pill" style="border-color:#0ca30c">● Calm · Dispersed</span> below 0.25</li>
<li><span class="pill" style="border-color:#fab219">▲ Elevated Stress</span> 0.25 – 0.40</li>
<li><span class="pill" style="border-color:#d03b3b">◆ Panic · High Clustering</span> 0.40 and above</li></ul>
<p><b>Preset buttons</b> jump straight to well-known episodes (each lands on the most-clustered day of that
episode) or to the calmest period in the data. COVID (2020) needs the 7-year lookback; clicking it switches
automatically.</p>"""),
    ("Time controls", "⏱ Time Machine: date scrubber, step & play", """
<p>Every chart shows the market <b>as it looked on the selected date</b>, using the 60 trading days that end
on that date.</p>
<ul><li><b>Drag the date slider</b> to rewind or fast-forward market history.</li>
<li><b>◀ / ▶</b> step back or forward by a day, week or month; <b>⏮ / ⏭</b> jump to the start or the latest date.</li>
<li><b>▶ Play history</b> animates through time so you can watch the network tighten into a crash and relax afterwards.</li>
<li><b>Click any point on the timeline chart</b> to jump to that date.</li></ul>"""),
    ("Threshold", "🎚 Correlation threshold filter", """
<p>Decides how strong a link must be before it's drawn on the network.</p>
<ul><li><b>Lower it (e.g. 0.4)</b> to see weaker, looser relationships. Calm markets show more structure, but crash
dates become a solid hairball.</li>
<li><b>Raise it (e.g. 0.75)</b> to keep only the tightest bonds. Only the most extreme clustering survives,
which makes crash dates stand out sharply.</li></ul>
<p>The threshold only changes the drawing and link counts. λ₁ and average correlation don't depend on it.</p>"""),
    ("Layouts", "🧭 Network layouts", """
<ul><li><b>Sector ring:</b> dots stay in fixed positions grouped by sector, so while you scrub or play only
the lines change. Best for watching links spread across sectors.</li>
<li><b>Cluster map:</b> stocks that move together are pulled close; unrelated ones drift apart. In calm periods
you'll see separate sector islands; in panics they collapse into one blob.</li></ul>"""),
    ("Other views", "🗂 Heatmap, strongest links & drawdown", """
<ul><li><b>Heatmap:</b> the full 50 × 50 correlation grid, sorted by sector. Red = move together, blue = move
opposite. Calm markets show red blocks along the diagonal (sectors); panics turn the whole square red.</li>
<li><b>Strongest links:</b> the most connected stocks and the tightest pairs on the selected date.</li>
<li><b>Drawdown chart:</b> how far an equal-weight basket of the 50 stocks is below its previous high, so
you can line up clustering with actual losses.</li></ul>"""),
    ("Fine print", "⚠️ Things to keep in mind", """
<ul><li><b>It lags a little:</b> each value uses the past 60 trading days, so a crash shows up within days but
keeps the readings high until it rolls out of the window (about 3 months).</li>
<li><b>Today's stocks:</b> the 50 names are current large caps, which slightly flatters the past (survivorship bias).</li>
<li><b>Correlation describes, it doesn't predict.</b> Clustering tells you how fragile diversification is
right now, not where prices go next. This is not investment advice.</li></ul>"""),
]
GUIDE_SHORT = {k: v for k, _, v in GUIDE}


def render_guide():
    cards = "".join(f'<div class="g"><h4>{title}</h4>{oneline(body)}</div>' for _, title, body in GUIDE)
    st.markdown(f'<div class="guide">{cards}</div>', unsafe_allow_html=True)


@st.dialog("📖 Function & Metric Guide", width="large")
def guide_dialog():
    render_guide()


def explain(key: str, label: str = "ⓘ Explain"):
    with st.popover(label):
        st.markdown(oneline(GUIDE_SHORT[key]), unsafe_allow_html=True)


# =============================================================== state
ss = st.session_state
ss.setdefault("playing", False)
ss.setdefault("refresh", 0)
ss.setdefault("lookback", "5 years")
ss.setdefault("tour_open", True)


def stop_play():
    ss.playing = False


def go_to(d):
    ss.sel_date = d
    ss.playing = False


def preset_click(key: str, d):
    ss.playing = False
    if d is None:  # episode not in the loaded history -> extend lookback, resolve after reload
        ss.lookback = "7 years (includes COVID)"
        ss._pending_preset = key
    else:
        ss.sel_date = d


def toggle_play():
    # Pressing Play on the last date restarts from the beginning instead of doing nothing.
    if not ss.playing and "_bounds" in ss and ss.sel_date >= ss._bounds[1]:
        ss.sel_date = ss._bounds[0]
    ss.playing = not ss.playing


# =============================================================== sidebar (part 1)
with st.sidebar:
    st.markdown('<div class="sb-brand">📉 Panic Detector<span>Systemic-risk monitor · 50 S&P 500 stocks</span></div>',
                unsafe_allow_html=True)
    time_box = st.container()
    filter_box = st.container()
    st.markdown('<div class="sb-h">Learn</div>', unsafe_allow_html=True)
    if st.button("📖  Function & Metric Guide", width="stretch", on_click=stop_play):
        guide_dialog()
    if st.button("👋  Show the quick tour", width="stretch"):
        ss.tour_open = True
    with st.expander("⚙  Advanced settings"):
        st.selectbox("History to load", list(LOOKBACKS), key="lookback",
                     help="5 years is the default. 7 years reaches back to the 2020 COVID crash.")
        window = st.slider("Correlation window (trading days)", 20, 120, 60, step=5,
                           help="How many past trading days each correlation snapshot uses. Shorter = reacts "
                                "faster but noisier; longer = smoother but slower. 60 ≈ 3 months.")
        if st.button("↻  Re-download prices", width="stretch"):
            ss.refresh += 1
            st.cache_data.clear()
            st.cache_resource.clear()
    data_note = st.empty()

years = LOOKBACKS[ss.lookback]

# =============================================================== data
try:
    with st.spinner(f"Loading {years} years of daily prices for 50 stocks…"):
        res, dd = load_risk(years, window, ss.refresh)
except Exception as e:
    st.error("Couldn't download market data from Yahoo Finance. Check your internet connection and use "
             "**Advanced settings → Re-download prices** to try again.")
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

m, dates, tickers, N = res.metrics, res.dates, res.tickers, len(res.tickers)
d_min, d_max = dates[0].date(), dates[-1].date()
ss._bounds = (d_min, d_max)
data_note.markdown(f'<div class="sb-note" style="margin-top:14px">Data: Yahoo Finance adjusted closes · '
                   f'{dates[0]:%b %Y} – {dates[-1]:%b %d, %Y} · {N} stocks · {len(dates):,} daily snapshots</div>',
                   unsafe_allow_html=True)


def preset_date(key: str):
    if key == "calm":
        return m["lambda1"].idxmin()
    _, a, b, _ = pdx.EPISODES[key]
    s = m["lambda1"].loc[a:b]
    return None if s.empty else s.idxmax()


PRESETS = {"calm": preset_date("calm"), **{k: preset_date(k) for k in pdx.EPISODES}}

# Apply deferred navigation (must happen before the date slider is created).
if "_pending_preset" in ss:
    d = PRESETS.get(ss.pop("_pending_preset"))
    if d is not None:
        ss.sel_date = d.date()
if "_next" in ss:
    ss.sel_date = ss.pop("_next")
if "sel_date" not in ss or not (d_min <= ss.sel_date <= d_max):
    ss.sel_date = d_max


def idx_of(d) -> int:
    return max(int(dates.searchsorted(pd.Timestamp(d), side="right") - 1), 0)


# =============================================================== sidebar (part 2)
with time_box:
    st.markdown(f'<div class="sb-h">⏱ Time machine {tip("Rewind market history. Every chart shows the market as it looked on this date, using the 60 trading days ending here.")}</div>',
                unsafe_allow_html=True)
    st.slider("Date", min_value=d_min, max_value=d_max, key="sel_date", format="MMM DD, YYYY",
              label_visibility="collapsed", on_change=stop_play,
              help="Drag to scrub through time.")
    step_n = STEP_SIZES[st.segmented_control("Step size", list(STEP_SIZES), default="1 week",
                                             key="step_size", label_visibility="collapsed") or "1 week"]
    k0 = idx_of(ss.sel_date)
    b1, b2, b3, b4 = st.columns(4)
    b1.button("⏮", on_click=go_to, args=(d_min,), width="stretch", help="Jump to the earliest date")
    b2.button("◀", on_click=go_to, args=(dates[max(k0 - step_n, 0)].date(),), width="stretch",
              help="Step back")
    b3.button("▶", on_click=go_to, args=(dates[min(k0 + step_n, len(dates) - 1)].date(),), width="stretch",
              help="Step forward")
    b4.button("⏭", on_click=go_to, args=(d_max,), width="stretch", help="Jump to the latest date")
    st.button("⏸  Pause" if ss.playing else "▶  Play history", on_click=toggle_play, type="primary",
              width="stretch")
    speed = SPEEDS[st.select_slider("Playback speed", list(SPEEDS), value="Normal", key="speed")]

with filter_box:
    st.markdown(f'<div class="sb-h">🎚 Network filter {tip("Only links stronger than this are drawn on the network. It does not change λ₁ or average correlation.")}</div>',
                unsafe_allow_html=True)
    thr = st.slider("Correlation threshold", 0.30, 0.90, 0.60, step=0.05, label_visibility="collapsed",
                    help="Lower = show looser relationships. Higher = keep only the tightest bonds.")
    st.markdown(f'<div class="sb-note">Drawing links where correlation &gt; <b>{thr:.2f}</b> '
                f'(1.00 = moving in perfect sync).</div>', unsafe_allow_html=True)
    layout_mode = st.radio("Layout", ["Sector ring", "Cluster map"], horizontal=True,
                           help="Sector ring keeps dots fixed so you can watch links appear. Cluster map pulls "
                                "stocks that move together close to each other.")

# =============================================================== selected snapshot
k = idx_of(ss.sel_date)
sel = dates[k]
row = m.iloc[k]
prev = m.iloc[max(k - 21, 0)]
C = res.corr[k].astype(np.float64)
n_pairs = N * (N - 1) // 2
n_edges = int((np.triu(C, 1) > thr).sum())
density = n_edges / n_pairs
dd_now = float(dd.reindex([sel]).iloc[0])
reg = REGIME[str(row.regime)]
win_start = sel - pd.tseries.offsets.BDay(res.window - 1)

# =============================================================== header
st.markdown(oneline(f"""
<div class="hdr">
  <div><div class="eyebrow">Systemic risk monitor</div><h1>Market Panic Detector</h1>
  <div class="sub">Are stocks moving on their own, or stampeding together? Scrub through {years} years of market
  history and watch diversification break down during sell-offs.</div></div>
  <div class="badge-wrap"><div class="badge" style="border-color:{reg['color']};background:{rgba(reg['color'], .14)}">
  <span class="ic" style="color:{reg['color']}">{reg['icon']}</span>{reg['label']}</div>
  <div class="badge-date">Snapshot for <b style="color:{INK}">{sel:%a, %b %d, %Y}</b> · window {win_start:%b %d} → {sel:%b %d}</div></div>
</div>"""), unsafe_allow_html=True)

# =============================================================== tour
with st.expander("👋  How to use this dashboard: a 1-minute tour", expanded=ss.tour_open):
    st.markdown(oneline(f"""
<div class="steps">
 <div class="step"><div class="n">STEP 1</div><div class="h">Read the status</div><div class="b">The badge
 top-right says whether stocks are moving independently (<b>Calm</b>), starting to herd (<b>Elevated</b>) or
 stampeding together (<b>Panic</b>) on the selected date.</div></div>
 <div class="step"><div class="n">STEP 2</div><div class="h">Jump to an episode</div><div class="b">Click a preset
 like <b>Calm Market</b> or <b>Rate Shock</b>. Compare the network: calm = scattered sector islands, crash = one
 dense web.</div></div>
 <div class="step"><div class="n">STEP 3</div><div class="h">Travel through time</div><div class="b">Use the
 <b>Time machine</b> in the sidebar: drag the slider, step ◀ ▶, or press <b>Play history</b>. You can also
 click the timeline chart.</div></div>
 <div class="step"><div class="n">STEP 4</div><div class="h">Learn as you go</div><div class="b">Hover any
 <b>?</b> for a plain-English explanation, click <b>ⓘ Explain</b> on a chart, or open the <b>Function &amp; Metric
 Guide</b> tab.</div></div>
</div>"""), unsafe_allow_html=True)
    st.button("Got it, hide the tour", on_click=lambda: ss.update(tour_open=False))

# =============================================================== presets
st.markdown('<div class="eyebrow" style="margin:6px 0 8px">Quick regime presets</div>', unsafe_allow_html=True)
with st.container(key="presets"):
    order = [("calm", "🟢  Calm Market", "The calmest stretch in the loaded history: lowest herd strength.")]
    icons = {"covid": "🦠", "rates": "📈", "banks": "🏦", "yen": "💴", "tariff": "🧾"}
    order += [(key, f"{icons[key]}  {lab}", desc) for key, (lab, _, _, desc) in pdx.EPISODES.items()]
    cols = st.columns(len(order))
    for c, (key, label, desc) in zip(cols, order):
        d = PRESETS[key]
        help_txt = desc + (f" Jumps to {d:%b %d, %Y}." if d is not None else
                           " Not in the 5-year history; clicking switches to the 7-year lookback.")
        c.button(label, key=f"preset_{key}", on_click=preset_click,
                 args=(key, d.date() if d is not None else None), help=help_txt, width="stretch")


# =============================================================== KPI cards
def delta(cur, old, fmt, unit=""):
    dv = cur - old
    eps = 0.005 if fmt == ".2f" else 0.05
    cls, arrow = ("up", "▲") if dv > eps else ("down", "▼") if dv < -eps else ("flat", "■")
    return f'<span class="{cls}">{arrow}</span> {dv:+{fmt}}{unit} vs 1 month ago'


kpis = [
    ("Herd strength λ₁", f"{row.lambda1:.1f}<small> / {N}</small>", delta(row.lambda1, prev.lambda1, ".1f"),
     "Top eigenvalue of the correlation matrix: how strongly one shared 'market herd' force drives all 50 "
     "stocks at once. ~7–12 is calm; 20+ means stocks are moving as a pack."),
    ("Avg. correlation", f"{row.mean_corr:.2f}", delta(row.mean_corr, prev.mean_corr, ".2f"),
     "Average of how closely every pair of stocks moves together (0 = unrelated, 1 = in sync). Higher means "
     "diversification protects you less. Calm < 0.25 · Panic ≥ 0.40."),
    ("Herd share", f"{row.lambda1_share:.0%}", "of all price movement comes from one shared factor",
     "λ₁ divided by the number of stocks: the fraction of total price movement explained by the single "
     "market-wide factor."),
    ("Tight links", f"{n_edges}<small> / {n_pairs:,}</small>",
     f"pairs above {thr:.2f} · network density {density:.0%}",
     "How many stock pairs are linked on the network at the current threshold. Changes with the threshold slider."),
    ("Basket drawdown", f"{dd_now:.0%}", "equal-weight basket vs. its previous high",
     "How far an equal-weight portfolio of the 50 stocks sits below its prior peak. Context for whether "
     "clustering coincided with real losses."),
]
st.markdown('<div class="kpis">' + "".join(
    f'<div class="kpi"><div class="kpi-label">{lab}{tip(t, right=(i >= 3))}</div>'
    f'<div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>'
    for i, (lab, val, sub, t) in enumerate(kpis)) + "</div>", unsafe_allow_html=True)

# =============================================================== narrative
dmc = row.mean_corr - prev.mean_corr
trend = ("Clustering has <b>jumped sharply</b> over the past month." if dmc > 0.08 else
         "Clustering has <b>risen</b> over the past month." if dmc > 0.03 else
         "Clustering has <b>eased sharply</b> over the past month." if dmc < -0.08 else
         "Clustering has <b>eased</b> over the past month." if dmc < -0.03 else
         "Clustering is <b>roughly unchanged</b> from a month ago.")
lead = {
    "Calm": f"Stocks are marching to their own drummers. Only <b>{n_edges}</b> of {n_pairs:,} pairs "
            f"({density:.0%}) move in tight lockstep, and the shared herd factor explains about "
            f"<b>{row.lambda1_share:.0%}</b> of price swings. Diversification is doing its job.",
    "Elevated": f"A common driver is taking hold. <b>{n_edges}</b> pairs ({density:.0%}) are tightly linked and "
                f"the herd factor explains <b>{row.lambda1_share:.0%}</b> of price swings. Sector boundaries "
                f"are starting to blur.",
    "Panic": f"Herd mode. Stocks are moving as one block: <b>{n_edges}</b> pairs ({density:.0%}) are tightly "
             f"linked and one shared factor drives <b>{row.lambda1_share:.0%}</b> of all price swings. Owning "
             f"many different stocks offers little protection right now.",
}[str(row.regime)]
ddtxt = (f"The equal-weight basket is <b>{abs(dd_now):.0%} below</b> its previous high."
         if dd_now < -0.03 else "The equal-weight basket is near its high.")
st.markdown(f'<div class="narr" style="border-left-color:{reg["color"]}"><div class="t">What\'s happening on '
            f'{sel:%b %d, %Y}</div>{lead} {trend} {ddtxt}</div>', unsafe_allow_html=True)

# =============================================================== tabs
tab_over, tab_heat, tab_links, tab_guide = st.tabs(
    ["Overview", "Correlation heatmap", "Strongest links", "📖 Function & Metric Guide"])

with tab_over:
    left, right = st.columns([1, 1.15], gap="medium")
    with left, st.container(key="card_net"):
        h1, h2 = st.columns([3, 1.25])
        h1.markdown(f'<div class="card-h">Correlation network</div><div class="card-s">Dots = stocks · lines = '
                    f'correlation above {thr:.2f}</div>', unsafe_allow_html=True)
        with h2:
            explain("Network graph")
        if layout_mode == "Cluster map":
            pos = force_positions(years, window, ss.refresh, k)
        else:
            pos = ring_layout(tuple(tickers))
        net_fig, G = network_figure(C, tickers, thr, pos, labels_outside=(layout_mode == "Sector ring"))
        st.plotly_chart(net_fig, width="stretch", config={"displayModeBar": False}, key="net")
        st.markdown('<div class="legend-note">Bigger dot = more strong links · brighter, thicker line = '
                    'tighter correlation · hover a dot for details</div>', unsafe_allow_html=True)
    with right, st.container(key="card_time"):
        h1, h2 = st.columns([3, 1.25])
        h1.markdown('<div class="card-h">Risk timeline</div><div class="card-s">Click anywhere on a line to jump '
                    'to that date · shaded = stress regimes</div>', unsafe_allow_html=True)
        with h2:
            explain("λ₁ (top eigenvalue)")
        short = {"calm": "Calm", "covid": "COVID", "rates": "Rate shock", "banks": "Banks",
                 "yen": "Yen", "tariff": "Tariffs"}
        markers = {short[key]: d for key, d in PRESETS.items() if d is not None}
        ev = st.plotly_chart(timeline_figure(m, dd, sel, N, markers), width="stretch",
                             config={"displayModeBar": False}, key="timeline",
                             on_select="rerun", selection_mode="points")
        pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
        if pts:
            x = pts[0].get("x")
            if x is not None and x != ss.get("_last_click"):
                ss._last_click = x
                ss._next = dates[idx_of(pd.Timestamp(x))].date()
                ss.playing = False
                st.rerun()

with tab_heat:
    with st.container(key="card_heat"):
        h1, h2 = st.columns([5, 1])
        h1.markdown('<div class="card-h">Correlation heatmap</div><div class="card-s">All 1,225 pairs, sorted by '
                    'sector · red = move together · blue = move opposite</div>', unsafe_allow_html=True)
        with h2:
            explain("Other views")
        st.plotly_chart(heatmap_figure(C, tickers), width="stretch", config={"displayModeBar": False})

with tab_links:
    with st.container(key="card_links"):
        st.markdown(f'<div class="card-h">Strongest links on {sel:%b %d, %Y}</div><div class="card-s">'
                    f'Which stocks sit at the centre of the herd, and which pairs move most in sync</div>',
                    unsafe_allow_html=True)
        a, b, c = st.columns(3, gap="medium")
        deg = pd.Series(dict(G.degree()), name="Strong links").sort_values(ascending=False).head(10)
        a.markdown("**Most connected stocks**")
        a.dataframe(pd.DataFrame({"Stock": deg.index, "Sector": [pdx.TICKER_SECTOR[t] for t in deg.index],
                                  "Strong links": deg.values}), hide_index=True, width="stretch")
        iu = np.triu_indices(N, 1)
        pairs = pd.DataFrame({"Stock A": np.array(tickers)[iu[0]], "Stock B": np.array(tickers)[iu[1]],
                              "Correlation": C[iu]}).sort_values("Correlation", ascending=False).head(10)
        b.markdown("**Tightest pairs**")
        b.dataframe(pairs.round(2), hide_index=True, width="stretch")
        sec = pd.Series([pdx.TICKER_SECTOR[t] for t in tickers])
        rows = []
        for s in pdx.SECTORS:
            ii = np.where(sec == s)[0]
            if len(ii) > 1:
                sub = C[np.ix_(ii, ii)]
                rows.append((s, sub[np.triu_indices(len(ii), 1)].mean()))
        c.markdown("**Within-sector cohesion**")
        c.dataframe(pd.DataFrame(rows, columns=["Sector", "Avg. correlation"]).sort_values(
            "Avg. correlation", ascending=False).round(2), hide_index=True, width="stretch")
    with st.expander("Download the full metrics history (CSV)"):
        st.dataframe(m.round(4), width="stretch", height=260)
        st.download_button("Download CSV", m.to_csv().encode(), "systemic_risk_metrics.csv", "text/csv")

with tab_guide:
    st.markdown('<div class="card-s" style="margin:6px 0 14px">Everything on this page, explained without the '
                'jargon.</div>', unsafe_allow_html=True)
    render_guide()

# =============================================================== playback loop
# A timer fragment advances the date between full reruns. Nothing sleeps inside the script,
# so Pause and every other control respond immediately while history is playing.
@st.fragment(run_every=speed if ss.playing else None)
def playback_ticker():
    if ss.pop("_tick_skip", False) or not ss.playing:
        return  # the inline call during a full rerun must not advance
    kk = idx_of(ss.sel_date)
    if kk >= len(dates) - 1:
        ss.playing = False
    else:
        ss._next = dates[min(kk + step_n, len(dates) - 1)].date()
    st.rerun(scope="app")


ss._tick_skip = True
playback_ticker()
