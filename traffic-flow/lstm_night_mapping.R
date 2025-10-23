
setwd("C:/traffic-flow")

zone_map    <- readr::read_csv("zone_map.csv")

zone_map <- zone_map %>%
  mutate(
    tensor_zone = as.character(tensor_zone),
    arima_zone  = as.character(arima_zone),
    lstm_zone   = as.character(lstm_zone)
  )

day_map     <- readr::read_csv("day_map.csv")

lstm_night <- readr::read_csv("lstm_night.csv")  

library(tidyverse)  # 會載入 dplyr（含 %>%）、readr 等
library(janitor)    # 提供 clean_names()

# 統一欄位命名（依你的實際欄位調整）
lstm_night_std <- lstm_night %>%
  clean_names() %>%
  rename(
    lstm_o   = from,
    lstm_d   = to,   
    lstm_day = day,    
    y_pred   = value   
  ) %>%
  mutate(
    model  = "LSTM",
    period = "night"
  )

# 鍵值型別先對齊（很重要）
day_map <- day_map %>%
  mutate(
    lstm_day   = as.character(lstm_day),
    tensor_day = as.character(tensor_day),
    arima_day  = as.character(arima_day)
  )

# 對齊「天數」（232–252 → day_abs / day_order）
lstm_night_std <- lstm_night_std %>%
  mutate(lstm_day = as.character(lstm_day)) %>%
  left_join(day_map %>% select(lstm_day, day_abs, day_order),
            by = "lstm_day")

# 對齊「起訖區域」（0–5 → 真實 zone_id）
lstm_night_std <- lstm_night_std %>%
  mutate(
    lstm_o = as.character(lstm_o),
    lstm_d = as.character(lstm_d)
  ) %>%
  # 起點
  left_join(
    zone_map %>%
      select(lstm_zone, zone_id) %>%
      rename(origin_zone_id = zone_id),
    by = c("lstm_o" = "lstm_zone")
  ) %>%
  # 終點
  left_join(
    zone_map %>%
      select(lstm_zone, zone_id) %>%
      rename(dest_zone_id = zone_id),
    by = c("lstm_d" = "lstm_zone")
  )

# 做成「period 等級」的 canonical 長表
# LSTM 本來就是整段時間的彙總，因此沒有每小時；我們把 slot_in_period/hour_abs 設成 NA。

lstm_night_canonical <- lstm_night_std %>%
  transmute(
    model,
    day_abs, day_order,
    period,                      # "morning"
    slot_in_period = NA_integer_,
    hour_abs       = NA_integer_,
    origin_zone_id, dest_zone_id,
    y_pred
  ) %>%
  arrange(day_order, origin_zone_id, dest_zone_id)

readr::write_csv(lstm_night_canonical, "lstm_night_canonical_period.csv")
