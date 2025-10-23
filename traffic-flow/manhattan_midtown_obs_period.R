
library(tidyverse)
library(janitor)

# hourly
## 0) 基本設定（專案根目錄與輸出資料夾）
setwd("C:/traffic-flow")  # ← 改成你的專案根路徑
out_dir <- "data/processed"

## 1) 載入 RData
load("data/raw/manhattan_midtown_origin.RData")

## 2) y.midtown 維度與順序說明：
##    依你回傳：num [1:252, 1:6, 1:6, 1:24]
##    我們採用：day × origin × dest × hour 的順序（因為 24 在最後一維）
stopifnot(length(dim(y.midtown)) == 4)

## 3) 看看這個 RData 裡面有哪些物件
ls()

## 4) 逐一檢查物件的型別與概況（挑看起來像「252天×24小時×6×6」或長表的那個）
str(y.midtown)
class(y.midtown)
if (is.array(y.midtown)) dim(y.midtown)

## 5) 若沒有 dimnames，先補上（避免 as.data.frame.table 變成無意義的 1,2,3）
if (is.null(dimnames(y.midtown))) {
  dimnames(y.midtown) <- list(
    day   = as.character(1:252),
    orig  = as.character(1:6),
    dest  = as.character(1:6),
    hour  = as.character(1:24)
  )
}

## 6) 攤平成長表
df <- as.data.frame.table(
  y.midtown,
  responseName = "count",
  stringsAsFactors = FALSE
)
## 這會產生欄位：day, orig, dest, hour, count

## 7) 轉整數 + 只留我們要的欄位與順序
df <- df %>%
  clean_names() %>%
  mutate(
    day_abs = as.integer(day),     # 1..252
    hour_abs = as.integer(hour),   # 1..24
    origin_pos = as.integer(orig), # 1..6
    dest_pos   = as.integer(dest), # 1..6
    count = as.numeric(count)
  ) %>%
  select(day_abs, hour_abs, origin_pos, dest_pos, count)

## 8) 把 1..6 對應到真實的 zone_id
zone_ids <- c(186, 100, 230, 161, 162, 163)  # 依你指定的順序
df <- df %>%
  mutate(
    origin_zone_id = zone_ids[origin_pos],
    dest_zone_id   = zone_ids[dest_pos]
  ) %>%
  select(day_abs, hour_abs, origin_zone_id, dest_zone_id, count)

## 9) 加上 period 與 slot_in_period（之後做彙總會用得到）
df <- df %>%
  mutate(
    period = case_when(
      hour_abs %in% 1:6   ~ "morning",
      hour_abs %in% 14:19 ~ "afternoon",
      hour_abs %in% 20:24 ~ "night",
      TRUE ~ NA_character_
    ),
    slot_in_period = case_when(
      period == "morning"   ~ hour_abs - 0,   # 1..6  → 1..6
      period == "afternoon" ~ hour_abs - 13,  # 14..19 → 1..6
      period == "night"     ~ hour_abs - 19,  # 20..24 → 1..5
      TRUE ~ NA_real_
    ),
    model = "obs",          # 之後合併模型時好分辨
    day_order = day_abs     # 全 252 天；之後要 232–252 再篩
  ) %>%
  relocate(model, .before = day_abs)

## 10) 輸出 CSV（逐小時原始實測的 canonical 表）
#out_csv <- file.path(out_dir, "obs_hourly_canonical.csv")
#readr::write_csv(df, out_csv)

#cat("✅ 已輸出：", out_csv, "\n")
## 快速檢查
#cat("rows:", nrow(df), "\n")           # 應為 252*24*36 = 217,728
#print(head(df, 3))

# Period
library(dplyr)
library(readr)

# df 目前是「逐小時 canonical」：含 model="obs", day_abs/hour_abs, period/slot_in_period, origin_zone_id, dest_zone_id, count

# 1) 僅保留有落在三大時段內的資料
df_period_src <- df %>%
  filter(!is.na(period))

# 2) 聚合：同日×同 period × 同 OD，把 01–06 或 14–19 或 20–24 的小時合併
#    預設使用「加總」（Sum）。如果你想用平均，把 sum(count) 換成 mean(count) 即可。
obs_period_canonical <- df_period_src %>%
  group_by(day_abs, day_order, period, origin_zone_id, dest_zone_id) %>%
  summarise(
    y_sum = sum(count),   # ← 若想看平均：mean(count)
    .groups = "drop"
  ) %>%
  mutate(model = "obs") %>%
  select(model, day_abs, day_order, period, origin_zone_id, dest_zone_id, y_sum) %>%
  arrange(day_abs, period, origin_zone_id, dest_zone_id)

# 3) 輸出
out_csv_period <- file.path(out_dir, "obs_period_canonical.csv")
write_csv(obs_period_canonical, out_csv_period)

cat("✅ 已輸出：", out_csv_period, "\n")

# 4) 快速檢查
#    每天三個 period（morning/afternoon/night），每個 period 有 6x6=36 筆 OD
#    → 總列數應為 252 天 × 3 × 36 = 27,216 列
n_rows <- nrow(obs_period_canonical)
cat("rows:", n_rows, "\n")  # 期待 27216

#    三個 period 的分布檢查（每個應該是 252*36 = 9,072 列）
print(
  obs_period_canonical %>%
    count(period) %>%
    arrange(period)
)

#    抽樣看前幾列
print(head(obs_period_canonical, 6))

