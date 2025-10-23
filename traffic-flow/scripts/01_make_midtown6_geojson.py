# -*- coding: utf-8 -*-


# scripts/01_make_midtown6_geojson.py
from pathlib import Path
import geopandas as gpd

DATA_DIR = Path(r"C:\traffic-flow")     # ← 只要改這行成你的專案根目錄
SHP_DIR  = DATA_DIR / "data" / "geo"
OUT_DIR  = SHP_DIR

shp_path = SHP_DIR / "taxi_zones.shp"

gdf = gpd.read_file(shp_path)

target_ids = [186, 100, 230, 161, 162, 163]
midtown6 = gdf[gdf["LocationID"].isin(target_ids)].copy()

name_map = {
    186: "Penn Station/Madison Sq West",
    100: "Garment District",
    230: "Times Sq/Theatre District",
    161: "Midtown Center",
    162: "Midtown East",
    163: "Midtown North",
}
midtown6["zone_id"] = midtown6["LocationID"]
midtown6["zone_name"] = midtown6["zone_id"].map(name_map)

# CRS 統一成 WGS84（給網頁地圖用）
if midtown6.crs is None:
    midtown6.set_crs(epsg=2263, inplace=True)  # TLC 常見座標
midtown6 = midtown6.to_crs(epsg=4326)

keep = [c for c in ["zone_id","zone_name","borough","zone","geometry"] if c in midtown6.columns]
midtown6 = midtown6[keep]

out_geojson = OUT_DIR / "midtown6.geojson"
midtown6.to_file(out_geojson, driver="GeoJSON")

print("saved:", out_geojson)
