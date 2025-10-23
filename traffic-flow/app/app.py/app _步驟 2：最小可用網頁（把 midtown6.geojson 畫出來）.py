# -*- coding: utf-8 -*-

# 1) 匯入套件
import json
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

# 2) 路徑設定
ROOT = Path(r"C:\traffic-flow")             # ← 改成你的專案根目錄
GEO  = ROOT / "data" / "geo" / "midtown6.geojson"

# 3) Streamlit 頁面外觀
st.set_page_config(page_title="Midtown Taxi Flow (Base Map)", layout="wide")
st.title("Midtown Manhattan – Taxi Zones (Base Map)")
st.caption("這是最小可用版本：先把六個區塊畫在地圖上，後續再接資料上色與動畫。")

# 4) 讀 GeoJSON 檔
with open(GEO, "r", encoding="utf-8") as f:
    gj = json.load(f)

# 5) 估地圖中心座標（小工具）
def guess_center(geojson):
    try:
        import shapely.geometry as geom
        from shapely.geometry import shape
        feats = geojson["features"]
        if len(feats) > 0:
            centroid = shape(feats[0]["geometry"]).centroid
            return centroid.y, centroid.x
    except Exception:
        pass
    return 40.758, -73.985  # Times Square

center_lat, center_lng = guess_center(gj)

# 6) 建立 Folium 地圖
m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="CartoDB positron")

# 7) 設定圖層樣式與 hover 提示
style = {"fillColor": "#ffffff", "color": "#333333", "weight": 1.2, "fillOpacity": 0.15}
highlight = {"weight": 2, "color": "#2C7FB8", "fillOpacity": 0.35}

tooltip = folium.features.GeoJsonTooltip(
    fields=["zone_name", "zone_id"],
    aliases=["Zone", "ID"],
    sticky=True
)

# 8) 把六區圖層加到地圖
folium.GeoJson(
    data=gj,
    name="Midtown 6 Zones",
    style_function=lambda x: style,
    highlight_function=lambda x: highlight,
    tooltip=tooltip
).add_to(m)

folium.LayerControl().add_to(m)

# 9) 在 Streamlit 中顯示地圖
st_folium(m, width=None, height=650)











