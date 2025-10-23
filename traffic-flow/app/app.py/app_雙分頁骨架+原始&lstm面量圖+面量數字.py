# app/app.py — 整合與增強版
# =========================================
# Tabs:
#   1) 21-day Compare (OBS vs ARIMA vs TensorTS vs LSTM) — period level
#   2) Full 252-day (OBS Hourly)
# Features:
#   - Reds color scale
#   - Value labels at zone centroids
#   - Global center/zoom controls (apply to all 4 maps)
#   - Aggregate modes: Origin sum / Destination sum / OD sum
# =========================================

import json
from pathlib import Path
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from branca.colormap import linear

# ---------- Paths ----------
ROOT = Path(r"C:\traffic-flow")   # ← 改成你的專案根目錄
GEO  = ROOT / "data" / "geo" / "midtown6.geojson"
DATA = ROOT / "data" / "processed"

st.set_page_config(page_title="Midtown Taxi Flow", layout="wide")
st.title("Midtown Manhattan – Taxi Flow Viewer")

# ---------- Load GeoJSON ----------
with open(GEO, "r", encoding="utf-8") as f:
    gj = json.load(f)

# ---------- Helpers: geometry & centroids ----------
def guess_center(geojson):
    try:
        from shapely.geometry import shape
        feats = geojson.get("features", [])
        if feats:
            c = shape(feats[0]["geometry"]).centroid
            return float(c.y), float(c.x)
    except Exception:
        pass
    return 40.758, -73.985  # Times Square

def compute_bounds(geojson):
    try:
        from shapely.geometry import shape
        import numpy as np
        minx = miny =  1e9
        maxx = maxy = -1e9
        for ft in geojson["features"]:
            b = shape(ft["geometry"]).bounds  # (minx, miny, maxx, maxy)
            minx = min(minx, b[0]); miny = min(miny, b[1])
            maxx = max(maxx, b[2]); maxy = max(maxy, b[3])
        return (miny, minx, maxy, maxx)
    except Exception:
        # fallback for Manhattan-ish area
        return (40.74, -74.01, 40.78, -73.95)

def compute_centroids(geojson):
    """回傳 dict: zone_id -> (lat, lng)，使用 representative_point() 確保在面內。"""
    centers = {}
    try:
        from shapely.geometry import shape
        for ft in geojson["features"]:
            props = ft.get("properties", {})
            zid = props.get("zone_id")
            if zid is None:
                continue
            geom = shape(ft["geometry"])
            p = geom.representative_point()   # 比 centroid 更適合標註
            centers[zid] = (float(p.y), float(p.x))
    except Exception:
        pass
    return centers

GLOBAL_CENTER = guess_center(gj)
GLOBAL_BOUNDS = compute_bounds(gj)
ZONE_CENTERS  = compute_centroids(gj)

# ---------- Common map builder ----------
def make_map_with_values(title, gj, values, vmin=None, vmax=None, center=None, zoom=13, show_labels=True):
    center = center or GLOBAL_CENTER
    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")

    # Color scale — Reds
    if values is not None and len(values) > 0:
        vmin = float(values.min()) if vmin is None else float(vmin)
        vmax = float(values.max()) if vmax is None else float(vmax)
        if vmax == vmin: vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = linear.Reds_09.scale(vmin, vmax)  # ← 改為紅色漸層
    base_style = {"weight": 1.2, "color": "#333333", "fillOpacity": 0.85}
    highlight  = {"weight": 2, "color": "#B30000", "fillOpacity": 0.95}
    tooltip    = folium.features.GeoJsonTooltip(fields=["zone_name", "zone_id"],
                                                aliases=["Zone", "ID"], sticky=True)

    def style_fn(feat):
        zid = feat["properties"].get("zone_id")
        fill = "#cccccc"
        if values is not None and zid in values.index:
            fill = cmap(values.loc[zid])
        s = base_style.copy()
        s["fillColor"] = fill
        return s

    folium.GeoJson(gj, name=title, style_function=style_fn,
                   highlight_function=lambda x: highlight, tooltip=tooltip).add_to(m)

    # Value labels (DivIcon at polygon representative point, centered + halo)
    if show_labels and values is not None and len(values) > 0:
        for zid, val in values.items():
            if zid in ZONE_CENTERS:
                lat, lng = ZONE_CENTERS[zid]
                label = f"{int(round(val)):,}"  # 整數＋千分位

                # 無白底的清晰標示：
                # - transform: translate(-50%,-50%) 讓文字以座標為中心
                # - text-shadow 做「光暈/描邊」以提升對比
                # - 可視需求調整 font-size / shadow 強度
                html = f"""
                <div style="
                    position: relative;
                    left: 50%; top: 50%;
                    transform: translate(-50%, -50%);
                    font-size: 12px; font-weight: 700;
                    color: #000;                /* 黑色文字 */
                    white-space: nowrap;
                    text-shadow:
                        -1px -1px 1px rgba(255,255,255,0.9),
                         1px -1px 1px rgba(255,255,255,0.9),
                        -1px  1px 1px rgba(255,255,255,0.9),
                         1px  1px 1px rgba(255,255,255,0.9),
                         0px  0px 2px rgba(0,0,0,0.35);  /* 輕微暗暈 */
                ">{label}</div>
                """

                folium.map.Marker(
                    [lat, lng],
                    icon=folium.DivIcon(
                        html=html,
                        # 讓 Leaflet 不要對齊到左上角，而是由我們的 CSS 置中處理：
                        icon_size=(0, 0),       # 交給 CSS 控制
                        icon_anchor=(0, 0)
                    )
                ).add_to(m)

    cmap.caption = title
    cmap.add_to(m)
    return m

# ---------- Loaders ----------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)

def get_period_file(model: str, period: str) -> Path | None:
    if model == "lstm":
        return {
            "morning":   DATA / "lstm_morning_canonical_period.csv",
            "afternoon": DATA / "lstm_afternoon_canonical_period.csv",
            "night":     DATA / "lstm_night_canonical_period.csv",
        }[period]
    if model == "obs":
        return DATA / "obs_period_canonical.csv"
    # 待你提供後再串：
    # if model == "arima":   return DATA / f"arima_{period}_canonical_period.csv"
    # if model == "tensors": return DATA / f"tensors_{period}_canonical_period.csv"
    return None

def period_zone_values(model: str, day_abs: int, period: str, agg_mode: str) -> pd.Series | None:
    path = get_period_file(model, period)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)

    val_col = next((c for c in ["y_pred", "value", "y_sum"] if c in df.columns), None)
    if val_col is None:
        return None

    df = df[(df["day_abs"] == day_abs) & (df["period"] == period)]
    if df.empty:
        return None

    if agg_mode == "Origin sum":
        g = df.groupby("origin_zone_id", dropna=False)[val_col].sum()
    elif agg_mode == "Destination sum":
        g = df.groupby("dest_zone_id", dropna=False)[val_col].sum()
    elif agg_mode == "OD sum":
        g1 = df.groupby("origin_zone_id")[val_col].sum().rename("o")
        g2 = df.groupby("dest_zone_id")[val_col].sum().rename("d")
        g = pd.concat([g1, g2], axis=1).fillna(0.0)
        g["total"] = g["o"] + g["d"]
        g = g["total"]
    else:
        return None

    g.index.name = "zone_id"
    return g

# ---------- Sidebar: global view controls ----------
st.sidebar.header("View Controls")
use_fit = st.sidebar.checkbox("Fit to Midtown bounds", value=False)
global_zoom = st.sidebar.slider("Zoom", min_value=10, max_value=16, value=13, step=1)
show_labels = st.sidebar.checkbox("Show labels (values)", value=True)

if use_fit:
    # 目前 Folium 只能在初次畫圖時 fit；這裡用固定 center/zoom 模擬，
    # 你也可以改寫 make_map_with_values 讓它使用 fit_bounds。
    center = ((GLOBAL_BOUNDS[0] + GLOBAL_BOUNDS[2]) / 2.0,
              (GLOBAL_BOUNDS[1] + GLOBAL_BOUNDS[3]) / 2.0)
else:
    center = GLOBAL_CENTER

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["21-day Compare (OBS vs Models)", "Full 252-day (OBS)"])

# ===== Tab 1: 21-day compare =====
with tab1:
    st.subheader("Compare last 21 days (Day 232–252) by Period")

    colc1, colc2, colc3, colc4 = st.columns([1,1,1,1])
    with colc1:
        day_21 = st.number_input("Day (232–252)", min_value=232, max_value=252, value=232, step=1)
    with colc2:
        period = st.radio("Period", ["morning","afternoon","night"], index=0, horizontal=True)
    with colc3:
        agg_mode = st.selectbox("Aggregate to zone", ["Origin sum","Destination sum","OD sum"], index=0)
    with colc4:
        st.write("")  # 占位

    # 四組值（目前 ARIMA/TensorTS 可能尚未接好 → None）
    v_obs    = period_zone_values("obs",     day_21, period, agg_mode)
    v_arima  = period_zone_values("arima",   day_21, period, agg_mode)     # 待接
    v_tensor = period_zone_values("tensors", day_21, period, agg_mode)     # 待接
    v_lstm   = period_zone_values("lstm",    day_21, period, agg_mode)

    # 共享色域
    vals_for_scale = [s for s in [v_obs, v_arima, v_tensor, v_lstm] if s is not None and len(s) > 0]
    if vals_for_scale:
        vmin = float(min(s.min() for s in vals_for_scale))
        vmax = float(max(s.max() for s in vals_for_scale))
    else:
        vmin, vmax = 0.0, 1.0

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        st.markdown("**OBS (Ground Truth)**")
        m1 = make_map_with_values("OBS", gj, v_obs, vmin, vmax, center=center, zoom=global_zoom, show_labels=show_labels)
        st_folium(m1, width=None, height=360)

    with c2:
        st.markdown("**ARIMA** *(待接資料)*")
        m2 = make_map_with_values("ARIMA", gj, v_arima, vmin, vmax, center=center, zoom=global_zoom, show_labels=show_labels)
        st_folium(m2, width=None, height=360)

    with c3:
        st.markdown("**TensorTS** *(待接資料)*")
        m3 = make_map_with_values("TensorTS", gj, v_tensor, vmin, vmax, center=center, zoom=global_zoom, show_labels=show_labels)
        st_folium(m3, width=None, height=360)

    with c4:
        st.markdown("**LSTM**")
        m4 = make_map_with_values("LSTM", gj, v_lstm, vmin, vmax, center=center, zoom=global_zoom, show_labels=show_labels)
        st_folium(m4, width=None, height=360)

# ===== Tab 2: Full 252-day (OBS hourly) =====
with tab2:
    st.subheader("OBS – Full 252 days (Hourly)")
    df_obs_hourly = load_csv("obs_hourly_canonical.csv")

    colh1, colh2 = st.columns([1,1])
    with colh1:
        d_full  = st.number_input("Day (1–252)", min_value=1, max_value=252, value=1, step=1)
    with colh2:
        hr_full = st.number_input("Hour (1–24)", min_value=1, max_value=24, value=1, step=1)

    df_sel = df_obs_hourly[(df_obs_hourly["day_abs"] == d_full) & (df_obs_hourly["hour_abs"] == hr_full)]
    v_hour = df_sel.groupby("origin_zone_id")["count"].sum()
    v_hour.index.name = "zone_id"

    m = make_map_with_values(f"OBS – Day {d_full} Hour {hr_full}", gj, v_hour,
                             center=center, zoom=global_zoom, show_labels=show_labels)
    st_folium(m, width=None, height=520)

    st.caption("之後這頁會加：自訂時間區間聚合、趨勢折線、與動畫播放。")
