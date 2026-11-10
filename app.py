"""Which airline is most likely to leave on time? India, from DGCA reports. Run: streamlit run app.py"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from fetch_weather import AIRPORTS
from train import CATEGORICAL, FEATURES, WX

ROOT = Path(__file__).parent
M = ROOT / "model"
ui.apply("India flight punctuality")
AIRLINE_COLOURS = {"IndiGo": "#1F4A6E", "Air India": "#A33A24", "Akasa Air": "#C26A2A", "SpiceJet": "#6B6B6B",
                   "Alliance Air": "#2F6B3A", "Vistara": "#6B2D4A", "Go First": "#8FA857", "AirAsia India": "#B08D57"}


@st.cache_resource
def load():
    meta = json.loads((M / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(M / "lgb.txt"))
    table = pd.read_parquet(M / "table.parquet")
    clim = pd.read_parquet(M / "climate.parquet").set_index(["airport", "cal_month"])
    return meta, booster, table, clim


meta, booster, T, CLIM = load()
R = meta["results"]
CATS = meta["categories"]
LATEST = T["month"].max()
city = {k: v[0] for k, v in AIRPORTS.items()}


def month_label(m):
    return pd.Timestamp(m + "-01").strftime("%B %Y")


def forecast(airport, target_month):
    """Predict OTP for every airline in the latest report at this airport, for a future month."""
    latest = T[T["month"] == LATEST]
    rows = latest[latest["airport"] == airport]
    if rows.empty:
        return pd.DataFrame()
    t = pd.Timestamp(target_month + "-01")
    since = (t.year - 2021) * 12 + t.month - 1
    a_mean = latest.groupby("airline")["otp"].mean()
    p_mean = latest.groupby("airport")["otp"].mean()
    X = pd.DataFrame({
        "airline": rows["airline"].values, "airport": airport, "cal_month": t.month, "months_since_2021": since,
        "last_otp": rows["otp"].values, "months_since_last": since - rows["months_since_2021"].values,
        "airline_last_mean": rows["airline"].map(a_mean).values, "airport_last_mean": p_mean.get(airport, np.nan)})
    for w in WX:
        X[w] = CLIM.loc[(airport, t.month), w]
    X = X[FEATURES]
    for k in CATEGORICAL:
        X[k] = pd.Categorical(X[k], categories=CATS[k])
    for k in FEATURES:
        if k not in CATEGORICAL:
            X[k] = X[k].astype("float32")
    pred = np.clip(rows["otp"].values + booster.predict(X), 0, 100)
    return pd.DataFrame({"airline": rows["airline"].values, "last": rows["otp"].values, "forecast": pred}) \
        .sort_values("forecast", ascending=False)


# ---------------------------------------------------------------- header
st.markdown(f'<div class="kicker">DGCA monthly traffic reports, {len(meta["months"])} months from '
            f'{month_label(meta["months"][0])} to {month_label(LATEST)}</div>', unsafe_allow_html=True)
st.markdown("# Which airline is most likely to leave on time?")
st.markdown('<p class="lede">India\'s aviation regulator publishes, every month, the share of each airline\'s flights that '
            'left on time at the country\'s major airports. This app learns from those official figures and the weather '
            'at each airport, and forecasts the coming months.</p>', unsafe_allow_html=True)

tab_pick, tab_record, tab_why, tab_acc = st.tabs(["Choose an airline", "Track record", "Why flights run late",
                                                  "How accurate it is"])

with tab_pick:
    latest_airports = sorted(T.loc[T["month"] == LATEST, "airport"].unique(), key=lambda a: city.get(a, a))
    nxt = pd.date_range(pd.Timestamp.today().normalize().replace(day=1), periods=12, freq="MS").strftime("%Y-%m").tolist()
    c1, c2 = st.columns(2)
    with c1:
        ap = st.selectbox("Airport", latest_airports, index=latest_airports.index("DEL") if "DEL" in latest_airports else 0,
                          format_func=lambda a: f"{city.get(a, a)} ({a})")
    with c2:
        tm = st.selectbox("Month you fly", nxt, index=min(1, len(nxt) - 1), format_func=month_label)
    fc = forecast(ap, tm)
    if fc.empty:
        st.info("No airline data for this airport in the latest report.")
    else:
        best = fc.iloc[0]
        cl = CLIM.loc[(ap, pd.Timestamp(tm + "-01").month)]
        ui.stats([(best["airline"], f"Most likely on time at {city[ap]} in {month_label(tm)}"),
                  (f"{best['forecast']:.0f}%", "Its forecast share of flights leaving on time"),
                  (f"{fc['forecast'].mean():.0f}%", "Average forecast across airlines"),
                  (f"{cl['fog_risk_hours']:.0f} h, {cl['rain_mm']:.0f} mm", "Typical early-morning fog-risk hours and rain that month")])
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown(f"### Forecast on-time share at {city[ap]}, {month_label(tm)}")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=fc["airline"], x=fc["forecast"], orientation="h",
                                 marker_color=[ui.ACCENT if i == 0 else "#B9B0A0" for i in range(len(fc))],
                                 text=[f"{v:.0f}%" for v in fc["forecast"]], textposition="inside", insidetextanchor="start",
                                 textfont=dict(color=["#FFFFFF" if i == 0 else ui.INK for i in range(len(fc))]), name="Forecast"))
            fig.add_trace(go.Scatter(y=fc["airline"], x=fc["last"], mode="markers", name=f"{month_label(LATEST)} actual",
                                     marker=dict(color=ui.INK, size=10, symbol="line-ns-open", line=dict(width=3))))
            ui.plotly_layout(fig, height=300 + 20 * len(fc))
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, 108], ticksuffix="%")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown(f'<p class="note">Bars: forecast. Marks: the latest official figure ({month_label(LATEST)}). '
                        f'Typical forecast error is about {R["LightGBM"]["mae"]:.0f} percentage points, so airlines within a few '
                        'points of each other are effectively level.</p>', unsafe_allow_html=True)
        with right:
            st.markdown(f"### {city[ap]} through the year")
            cm = CLIM.loc[ap].reset_index()
            mon = [pd.Timestamp(2025, int(m), 1).strftime("%b") for m in cm["cal_month"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mon, y=cm["rain_mm"], name="Rain, mm", marker_color="#8FA5B8"))
            fig.add_trace(go.Scatter(x=mon, y=cm["fog_risk_hours"], name="Fog-risk hours", yaxis="y2",
                                     mode="lines+markers", line=dict(color=ui.INK, width=2.5)))
            ui.plotly_layout(fig, height=340)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0),
                              yaxis=dict(title="Rain, mm"),
                              yaxis2=dict(title="Fog-risk hours", overlaying="y", side="right", showgrid=False,
                                          rangemode="tozero", tickformat=".0f", tickfont=dict(color=ui.MUTED)))
            fig.update_yaxes(rangemode="tozero", selector=dict(overlaying=None))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown('<p class="note">Fog-risk hours: hours between midnight and 10 AM with humidity of 97% or more and '
                        'light wind, averaged over 2021 to 2026.</p>', unsafe_allow_html=True)

with tab_record:
    st.markdown(f"### Latest official figures, {month_label(LATEST)}")
    lt = T[T["month"] == LATEST].pivot(index="airline", columns="airport", values="otp")
    lt = lt.loc[lt.mean(axis=1).sort_values(ascending=False).index, sorted(lt.columns, key=lambda a: city.get(a, a))]
    head = "".join(f"<th class='num'>{city.get(a, a)}</th>" for a in lt.columns)

    def cell(v):
        if pd.isna(v):
            return "<td class='num'></td>"
        cls = "pos" if v >= 85 else ("neg" if v < 60 else "")
        return f"<td class='num {cls}'>{v:.0f}%</td>"
    body = "".join(f"<tr><td>{a}</td>{''.join(cell(v) for v in r.values)}</tr>" for a, r in lt.iterrows())
    st.markdown(f"<div style='overflow-x:auto'><table class='plain'><tr><th>Airline</th>{head}</tr>{body}</table></div>"
                "<p class='note'>Share of departures on time. Green 85% or more, red under 60%.</p>", unsafe_allow_html=True)

    st.markdown("### One airline over time")
    airlines = sorted(T["airline"].unique())
    al = st.selectbox("Airline", airlines, index=airlines.index("IndiGo") if "IndiGo" in airlines else 0)
    d = T[T["airline"] == al]
    fig = go.Figure()
    all_months = pd.date_range(pd.Timestamp(T["month"].min() + "-01"), pd.Timestamp(LATEST + "-01"), freq="MS")
    for a in sorted(d["airport"].unique(), key=lambda x: city.get(x, x)):
        g = d[d["airport"] == a].set_index(pd.to_datetime(d.loc[d["airport"] == a, "month"]))["otp"]
        g = g.reindex(all_months)          # months without a report stay empty, so the line breaks there
        fig.add_trace(go.Scatter(x=g.index, y=g.values, mode="lines+markers", name=city.get(a, a),
                                 line=dict(width=2), connectgaps=False))
    ui.plotly_layout(fig, height=380)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.12, x=0),
                      colorway=["#1F4A6E", "#A33A24", "#2F6B3A", "#C26A2A", "#6B2D4A", "#4A4F59", "#8FA857", "#B08D57", "#3D7A8C", "#16181D"])
    fig.update_yaxes(title="On time, %", range=[0, 102], ticksuffix="%")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<p class="note">DGCA reports are missing on the public site for October 2023 to April 2025, apart from a '
                'few months, so the lines have a gap. 2021 figures are unusually high because traffic was low during COVID.</p>',
                unsafe_allow_html=True)

with tab_why:
    rs = pd.DataFrame(meta["reasons"])
    lr = rs[rs["month"] == rs["month"].max()].sort_values("share", ascending=False)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown(f"### Causes of delay, {month_label(rs['month'].max())}")
        fig = go.Figure(go.Bar(y=lr["reason"], x=lr["share"], orientation="h", marker_color=ui.ACCENT,
                               text=[f"{v:.0f}%" for v in lr["share"]], textposition="outside"))
        ui.plotly_layout(fig, height=380)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(ticksuffix="%", range=[0, lr["share"].max() * 1.25])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Late incoming aircraft, share of all delays")
        re_ = rs[rs["reason"].str.startswith("Reactionary")].set_index("month")["share"]
        re_.index = pd.to_datetime(re_.index)
        re_ = re_.reindex(pd.date_range(re_.index.min(), re_.index.max(), freq="MS"))   # gaps stay gaps
        fig = go.Figure(go.Scatter(x=re_.index, y=re_.values, mode="lines+markers", connectgaps=False,
                                   line=dict(color=ui.INK, width=2.5)))
        ui.plotly_layout(fig, height=380)
        fig.update_yaxes(ticksuffix="%", range=[0, 100])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<p class="note">"Reactionary" means the aircraft arrived late from its previous flight, so one early delay '
                'ripples through the day. That is why airline-wide punctuality moves more than any single airport\'s weather.</p>',
                unsafe_allow_html=True)

with tab_acc:
    L, P, A = R["LightGBM"], R["Last reported value"], R["Airline average"]
    ui.stats([(f"{L['mae']:.1f} pts", "Average error on months the model never saw"),
              (f"{P['mae']:.1f} pts", "Repeating the last reported figure"),
              (f"{A['mae']:.1f} pts", "Using the airline's average"),
              (f"{L['within5']:.0%}", "Forecasts within 5 points")])
    bp = meta["by_period"]
    body = "".join(f"<tr><td>{k}</td>{''.join(f'<td class=num>{v[m]:.1f}</td>' for m in ['LightGBM', 'Last reported value', 'Airline average'])}</tr>"
                   for k, v in bp.items())
    st.markdown("<table class='plain'><tr><th>Test period</th><th class='num'>Model</th><th class='num'>Last reported</th>"
                f"<th class='num'>Airline average</th></tr>{body}</table><p class='note'>Average error in percentage points.</p>",
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="panel">
<h3>Data</h3>
<p>{meta['n_rows']:,} official figures: the on-time share of each airline at each major airport, read from {len(meta['months'])}
DGCA monthly domestic traffic reports (March 2021 to September 2023, May, October and November 2025, and April 2026). DGCA
covered four metro airports until 2023, six in 2025 and ten from 2026. Reports for most months from October 2023 to April 2025
are not on DGCA's public file server. Weather is monthly rain, heavy-rain, thunder and gusty days, temperature and early-morning
fog-risk hours at each airport, from the Open-Meteo archive.</p>
<h3>Model</h3>
<p>LightGBM with very small trees ({meta['trees']} trees of 4 leaves), predicting the change from the airline's last reported
figure at that airport. Inputs: airline, airport, calendar month, time since 2021, the last figure and how old it is, the airline's
and the airport's averages in the previous report, and the weather.</p>
<h3>Test</h3>
<p>The model learned from March 2021 to March 2023 only. Every later report ({meta['n_test']} figures) is the test.</p>
<h3>Limits</h3>
<p>This is monthly and airline-wide, not your individual flight: it ranks airlines, it cannot say whether flight 6E 123 will be
late. With so few months, the model only modestly beats repeating the last figure, and the long gap in public reports makes
2025 and 2026 harder to forecast. Fog-risk hours are an estimate from humidity and wind, not observed visibility.</p>
</div>""", unsafe_allow_html=True)
