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
#   - NEW: Label rounding = half-up（Sum: 0 位; Mean: 1 位）
# =========================================

import json
from pathlib import Path
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium
from branca.colormap import linear

# ---------- NEW: half-up 格式化用 ----------
# 用於地圖「數值標籤」的四捨五入顯示（不影響計算與色階）
from decimal import Decimal, ROUND_HALF_UP
def format_label_half_up(val: float, digits: int) -> str:
    """
    以「四捨五入（half-up）」格式化數字字串：
    - digits=0 → 整數
    - digits=1 → 一位小數
    - 顯示含千分位
    """
    q = Decimal(str(val)).quantize(Decimal("1") if digits == 0 else Decimal(f"1.{'0'*digits}"),
                                   rounding=ROUND_HALF_UP)
    return f"{q:,.{digits}f}"

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
# NEW: 增加 label_digits 參數，控制標籤顯示的小數位（Sum=0, Mean=1）
def make_map_with_values(title, gj, values, vmin=None, vmax=None,
                         center=None, zoom=13, show_labels=True, label_digits=0):
    center = center or GLOBAL_CENTER
    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")

    # Color scale — Reds
    if values is not None and len(values) > 0:
        vmin = float(values.min()) if vmin is None else float(vmin)
        vmax = float(values.max()) if vmax is None else float(vmax)
        if vmax == vmin: vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = linear.Reds_09.scale(vmin, vmax)  # ← 紅色漸層
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

    # Value labels（置中＋光暈；僅改「顯示」的 rounding 規則為 half-up）
    if show_labels and values is not None and len(values) > 0:
        for zid, val in values.items():
            if zid in ZONE_CENTERS:
                lat, lng = ZONE_CENTERS[zid]
                # NEW: 用 half-up，依 label_digits 控制 0/1 位小數
                label = format_label_half_up(float(val), digits=label_digits)

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
                         0px  0px 2px rgba(0,0,0,0.35);
                ">{label}</div>
                """

                folium.map.Marker(
                    [lat, lng],
                    icon=folium.DivIcon(
                        html=html,
                        icon_size=(0, 0),  # 交給 CSS 控制
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
    """
    回傳各模型在「period 等級」的檔案路徑。
    已串接：
      - LSTM: lstm_<period>_canonical_period.csv
      - ARIMA: arima_<period>_canonical.csv
      - TensorTS: tensor_<period>_canonical.csv
      - OBS: obs_period_canonical.csv
    """
    # LSTM（你已有 *_canonical_period.csv）
    if model.lower() == "lstm":
        return {
            "morning":   DATA / "lstm_morning_canonical_period.csv",
            "afternoon": DATA / "lstm_afternoon_canonical_period.csv",
            "night":     DATA / "lstm_night_canonical_period.csv",
        }.get(period)

    # ARIMA（你上傳的是 arima_<period>_canonical.csv）
    if model.lower() == "arima":
        return {
            "morning":   DATA / "arima_morning_canonical.csv",
            "afternoon": DATA / "arima_afternoon_canonical.csv",
            "night":     DATA / "arima_night_canonical.csv",
        }.get(period)

    # TensorTS（你上傳的是 tensor_<period>_canonical.csv）
    # 注意：我這裡用「tensor」當模型代碼（不是 tensors）
    if model.lower() in ["tensor", "tensors", "tensorts"]:
        return {
            "morning":   DATA / "tensor_morning_canonical.csv",
            "afternoon": DATA / "tensor_afternoon_canonical.csv",
            "night":     DATA / "tensor_night_canonical.csv",
        }.get(period)

    # OBS（原始實測的 period 聚合）
    if model.lower() == "obs":
        return DATA / "obs_period_canonical.csv"

    return None

# 每個 period 的小時數（用於 Mean）
HOURS_PER_PERIOD = {"morning": 6, "afternoon": 6, "night": 5}

def period_zone_values(model: str, day_abs: int, period: str, agg_mode: str, stat: str = "Sum") -> pd.Series | None:
    """
    回傳 index=zone_id 的 Series。
    - LSTM 檔是「每小時平均」：Mean -> 原值；Sum -> 乘上時段小時數
    - 其他模型（OBS/ARIMA/TensorTS）是「時段合計」：Sum -> 原值；Mean -> 除以時段小時數
    """
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

    # 先把 period 等級數值聚合到「每區」
    if agg_mode == "Origin sum (flow-out)":
        g = df.groupby("origin_zone_id", dropna=False)[val_col].sum()
    elif agg_mode == "Destination sum (flow-in)":
        g = df.groupby("dest_zone_id", dropna=False)[val_col].sum()
    elif agg_mode == "OD sum":
        g1 = df.groupby("origin_zone_id")[val_col].sum().rename("o")
        g2 = df.groupby("dest_zone_id")[val_col].sum().rename("d")
        g = (pd.concat([g1, g2], axis=1).fillna(0.0).sum(axis=1))
    else:
        return None

    # 依模型/統計量做單位對齊
    hrs = HOURS_PER_PERIOD.get(period, 0)

    m = model.lower()
    if m == "lstm":
        # LSTM: period 檔是「每小時平均」
        if stat == "Sum" and hrs > 0:
            g = g * float(hrs)   # 轉成時段合計
        # stat == "Mean" -> 保持原值
    else:
        # 其他模型: period 檔是「時段合計」
        if stat == "Mean" and hrs > 0:
            g = g / float(hrs)   # 轉成每小時平均

    g.index.name = "zone_id"
    return g


# ---- 將 OBS 逐小時依「任意小時清單」聚合成每區數值 ----
def hourly_zone_values(df_obs_hourly: pd.DataFrame, day_abs: int, hours: list[int],
                       agg_mode: str, stat: str = "Sum") -> pd.Series | None:
    """
    df_obs_hourly: obs_hourly_canonical.csv 讀進來的 DataFrame
    day_abs      : 第幾天 (1..252)
    hours        : 小時清單（例如 [1,2,3] 或 range 轉成 list）
    agg_mode     : "Origin sum (flow-out)" / "Destination sum (flow-in)" / "OD sum"
    stat         : "Sum" or "Mean"（對「小時」做加總或平均）
    回傳：index=zone_id 的 Series
    """
    sub = df_obs_hourly[(df_obs_hourly["day_abs"] == day_abs) &
                        (df_obs_hourly["hour_abs"].isin(hours))]
    if sub.empty:
        return None

    # 先把「小時」層級匯整到 OD（對選定小時做 sum 或 mean）
    val = sub.groupby(["origin_zone_id", "dest_zone_id"], dropna=False)["count"]
    od = (val.sum() if stat == "Sum" else val.mean()).reset_index(name="value")

    # 再把 OD → 每區（依 agg_mode 匯整到區域）
    if agg_mode == "Origin sum (flow-out)":
        g = od.groupby("origin_zone_id", dropna=False)["value"].sum()
        g.index.name = "zone_id"
        return g
    elif agg_mode == "Destination sum (flow-in)":
        g = od.groupby("dest_zone_id", dropna=False)["value"].sum()
        g.index.name = "zone_id"
        return g
    elif agg_mode == "OD sum":
        g1 = od.groupby("origin_zone_id")["value"].sum().rename("o")
        g2 = od.groupby("dest_zone_id")["value"].sum().rename("d")
        g = pd.concat([g1, g2], axis=1).fillna(0.0)
        g["total"] = g["o"] + g["d"]
        g = g["total"]
        g.index.name = "zone_id"
        return g
    return None

# ---- 找檔：優先 data/processed，再嘗試 data/raw ----
def _find_csv_in_data_dirs(filename: str) -> Path | None:
    cand1 = DATA / filename
    if cand1.exists(): return cand1
    cand2 = ROOT / "data" / "raw" / filename
    if cand2.exists(): return cand2
    return None

# 0..5 → 真實 zone_id 的對應（依你專案）
LSTM_ZONE_MAP = {0:186, 1:100, 2:230, 3:161, 4:162, 5:163}

# ---- 讀原始 LSTM（每小時 per-hour 值），直接聚合到 zone（目前未在 Tab 使用，保留備用） ----
def lstm_raw_zone_values(day_abs: int, period: str, agg_mode: str) -> pd.Series | None:
    fname = {
        "morning":   "lstm_morning.csv",
        "afternoon": "lstm_afternoon.csv",
        "night":     "lstm_night.csv",
    }.get(period)
    if fname is None:
        return None

    path = _find_csv_in_data_dirs(fname)
    if path is None or not path.exists():
        return None

    df = pd.read_csv(path)
    need = {"day","from","to","value"}
    if not need.issubset(set(df.columns)):
        return None

    sub = df[df["day"] == day_abs].copy()
    if sub.empty:
        return None

    sub["origin_zone_id"] = sub["from"].map(LSTM_ZONE_MAP)
    sub["dest_zone_id"]   = sub["to"].map(LSTM_ZONE_MAP)

    if agg_mode == "Origin sum (flow-out)":
        g = sub.groupby("origin_zone_id", dropna=False)["value"].sum()
    elif agg_mode == "Destination sum (flow-in)":
        g = sub.groupby("dest_zone_id", dropna=False)["value"].sum()
    elif agg_mode == "OD sum":
        g1 = sub.groupby("origin_zone_id")["value"].sum().rename("o")
        g2 = sub.groupby("dest_zone_id")["value"].sum().rename("d")
        g = (pd.concat([g1, g2], axis=1).fillna(0.0).sum(axis=1))
    else:
        return None

    g.index.name = "zone_id"
    return g

# ---------- Sidebar: global view controls ----------
st.sidebar.header("View Controls")
use_fit = st.sidebar.checkbox("Fit to Midtown bounds", value=False, key="sb_fit")
global_zoom = st.sidebar.slider("Zoom", min_value=10, max_value=16, value=13, step=1, key="sb_zoom")
show_labels = st.sidebar.checkbox("Show labels (values)", value=True, key="sb_labels")

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
        day_21 = st.number_input("Day (232–252)", min_value=232, max_value=252, value=232, step=1, key="t1_day")
    with colc2:
        period = st.radio("Period", ["morning","afternoon","night"], index=0, horizontal=True, key="t1_period")
    with colc3:
        agg_mode = st.selectbox("Aggregate to zone", ["Origin sum (flow-out)","Destination sum (flow-in)","OD sum"], index=0, key="t1_agg")
    with colc4:
        stat1 = st.radio("Period agg", ["Sum", "Mean"], index=0, horizontal=True, key="t1_stat")

    # OBS / ARIMA / TensorTS / LSTM 依 stat1 統一轉換單位（內部不捨入）
    v_obs    = period_zone_values("obs",    day_21, period, agg_mode, stat=stat1)
    v_arima  = period_zone_values("arima",  day_21, period, agg_mode, stat=stat1)
    v_tensor = period_zone_values("tensor", day_21, period, agg_mode, stat=stat1)
    v_lstm   = period_zone_values("lstm",   day_21, period, agg_mode, stat=stat1)

    # 共享色域
    vals_for_scale = [s for s in [v_obs, v_arima, v_tensor, v_lstm] if s is not None and len(s) > 0]
    if vals_for_scale:
        vmin = float(min(s.min() for s in vals_for_scale))
        vmax = float(max(s.max() for s in vals_for_scale))
    else:
        vmin, vmax = 0.0, 1.0

    # NEW: Sum → 0 位；Mean → 1 位（僅影響標籤顯示）
    label_digits_tab1 = 0 if stat1 == "Sum" else 1

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        st.markdown(f"**OBS (Ground Truth) – {stat1}**")
        m1 = make_map_with_values("OBS", gj, v_obs, vmin, vmax,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=label_digits_tab1)
        st_folium(m1, width=None, height=360)

    with c2:
        st.markdown(f"**ARIMA – {stat1}**")
        m2 = make_map_with_values("ARIMA", gj, v_arima, vmin, vmax,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=label_digits_tab1)
        st_folium(m2, width=None, height=360)

    with c3:
        st.markdown(f"**TensorTS – {stat1}**")
        m3 = make_map_with_values("TensorTS", gj, v_tensor, vmin, vmax,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=label_digits_tab1)
        st_folium(m3, width=None, height=360)

    with c4:
        st.markdown(f"**LSTM – {stat1}**")
        m4 = make_map_with_values("LSTM", gj, v_lstm, vmin, vmax,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=label_digits_tab1)
        st_folium(m4, width=None, height=360)

# ===== Tab 2: Full 252-day (OBS hourly) - 任意時間區間聚合 + 動畫（雙 state 版） =====
with tab2:
    st.subheader("OBS – Full 252 days (Hourly / Range & Animation)")

    df_obs_hourly = load_csv("obs_hourly_canonical.csv")

    # --- 初始化 session_state ---
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = False
    if "anim_hour" not in st.session_state:
        st.session_state.anim_hour = 1      # 內部動畫目前小時
    if "anim_hour_widget" not in st.session_state:
        st.session_state.anim_hour_widget = 1  # 滑桿自己的值
    if "prev_day" not in st.session_state:
        st.session_state.prev_day = 1
    if "prev_range" not in st.session_state:
        st.session_state.prev_range = (1, 6)

    # 當滑桿被用戶改動時，把它同步到內部動畫小時
    def _sync_anim_from_widget():
        st.session_state.anim_hour = st.session_state.anim_hour_widget

    # ---- 控制列 ----
    colh1, colh2, colh3 = st.columns([1, 2, 1])
    with colh1:
        d_full  = st.number_input("Day (1–252)", min_value=1, max_value=252, value=1, step=1)
        agg_mode2 = st.selectbox("Aggregate to zone",
                                 ["Origin sum (flow-out)", "Destination sum (flow-in)", "OD sum"],
                                 index=0)
        stat2 = st.radio("Hour agg", ["Sum", "Mean"], horizontal=True, index=0)
    with colh2:
        h_start, h_end = st.slider("Hour range (1–24)", min_value=1, max_value=24, value=(1, 6), step=1)
        hours_range = list(range(h_start, h_end + 1))
        st.caption(f"區間小時：{hours_range}")
    with colh3:
        animate = st.checkbox("Animate hours", value=False)
        speed = st.slider("Speed (sec/frame)", 0.05, 1.0, 0.25, 0.05)

        cplay1, cplay2 = st.columns(2)
        with cplay1:
            if st.button("▶ Play"):
                st.session_state.anim_running = True
        with cplay2:
            if st.button("⏸ Pause"):
                st.session_state.anim_running = False

        # 「目前小時」滑桿（key 與內部 state 分離；on_change 做同步）
        st.slider("Current hour", 1, 24,
                  value=st.session_state.anim_hour, step=1,
                  key="anim_hour_widget", on_change=_sync_anim_from_widget)

    # Day 或 範圍改變時，把內部動畫小時 & 滑桿都重設到區間起點
    if (d_full != st.session_state.prev_day) or ((h_start, h_end) != st.session_state.prev_range):
        st.session_state.prev_day = d_full
        st.session_state.prev_range = (h_start, h_end)
        st.session_state.anim_hour = h_start
        st.session_state.anim_hour_widget = h_start

    # ---- 畫面：左「區間聚合」、右「單一小時（動畫）」 ----
    left, right = st.columns(2)

    # 左圖：區間聚合（Sum/Mean over hours_range）
    with left:
        v_range = hourly_zone_values(df_obs_hourly, d_full, hours_range, agg_mode2, stat2)
        title_l = f"OBS – Day {d_full}, Hours {h_start}-{h_end} ({stat2})"
        # NEW: Sum → 0 位；Mean → 1 位（僅顯示）
        label_digits_left = 0 if stat2 == "Sum" else 1
        mL = make_map_with_values(title_l, gj, v_range,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=label_digits_left)
        st_folium(mL, width=None, height=520,
                  key=f"left_map_{d_full}_{h_start}_{h_end}_{agg_mode2}_{stat2}")

    # 右圖：單一小時（逐幀變化；用「動態 key」強制重繪）
    with right:
        cur_h = st.session_state.anim_hour
        df_sel = df_obs_hourly[(df_obs_hourly["day_abs"] == d_full) &
                               (df_obs_hourly["hour_abs"] == cur_h)]
        if agg_mode2 == "Origin sum (flow-out)":
            v_single = df_sel.groupby("origin_zone_id")["count"].sum()
        elif agg_mode2 == "Destination sum (flow-in)":
            v_single = df_sel.groupby("dest_zone_id")["count"].sum()
        else:  # OD sum
            g1 = df_sel.groupby("origin_zone_id")["count"].sum().rename("o")
            g2 = df_sel.groupby("dest_zone_id")["count"].sum().rename("d")
            v_single = (pd.concat([g1, g2], axis=1).fillna(0.0).sum(axis=1))
        v_single.index.name = "zone_id"

        title_r = f"OBS – Day {d_full}, Hour {cur_h}"
        # NEW: 單一小時本來就是整數 → 標籤 0 位
        mR = make_map_with_values(title_r, gj, v_single,
                                  center=center, zoom=global_zoom, show_labels=show_labels,
                                  label_digits=0)
        st_folium(mR, width=None, height=520,
                  key=f"right_map_{d_full}_{cur_h}_{agg_mode2}")

        # 顯示目前影格（可當作簡易進度指標）
        st.markdown(f"**Frame hour:** {cur_h}")

    st.caption("左：任意時間區間聚合（Sum/Mean）。右：單一小時，支援播放動畫。")

    # ---- 渲染完才推進到下一幀並 rerun ----
    if animate and st.session_state.anim_running:
        import time
        time.sleep(speed)

        nxt = st.session_state.anim_hour + 1
        if nxt > h_end or nxt < h_start:
            nxt = h_start
        # 只更新內部狀態（不是 widget 的 key）
        st.session_state.anim_hour = nxt

        st.rerun()
