setwd("C:\\traffic-flow")

zone_map <- readr::read_csv("zone_map.csv")

zone_map <- zone_map %>%
  mutate(
    arima_zone  = as.character(arima_zone),
    tensor_zone = as.character(tensor_zone),
    lstm_zone   = as.character(lstm_zone)
  )

day_map <- readr::read_csv("day_map.csv")

period_hour_map <- readr::read_csv("period_hour_map.csv")

arima_afternoon <- readr::read_csv("arima_afternoon.csv")

# 統一命名
arima_afternoon_std <- arima_afternoon %>%
  janitor::clean_names() %>%  # 讓欄位名變小寫、底線
  rename(
    arima_day  = idx3,   # ←若原本就叫 arima_day 就刪這行
    arima_o    = orig,   # ←若叫 o/d/slot/value，請照實改
    arima_d    = dest,
    arima_slot = idx4,
    y_pred     = value
  ) %>%
  mutate(model = "ARIMA", period = "afternoon")


## 關鍵修正：把 day_map 的鍵欄位變成 character
day_map <- day_map %>%
  mutate(
    arima_day  = as.character(arima_day),
    lstm_day   = as.character(lstm_day),
    tensor_day = as.character(tensor_day)
  )

# 對齊「天數」
arima_afternoon_std <- arima_afternoon_std %>%
  mutate(arima_day = as.character(arima_day)) %>%
  left_join(
    day_map %>% select(arima_day, day_abs, day_order),
    by = "arima_day"
  )

# 對齊「起迄點區域」
arima_afternoon_std <- arima_afternoon_std %>%
  mutate(
    arima_o = as.character(arima_o),
    arima_d = as.character(arima_d)
  ) %>%                                  # 起點
  left_join(
    zone_map %>%
      select(arima_zone, zone_id) %>%
      rename(origin_zone_id = zone_id),
    by = c("arima_o" = "arima_zone")
  ) %>%                                  # 終點
  left_join(
    zone_map %>%
      select(arima_zone, zone_id) %>%
      rename(dest_zone_id = zone_id),
    by = c("arima_d" = "arima_zone")
  )

# 對齊「時段槽位 → 小時」
arima_afternoon_std <- arima_afternoon_std %>%
  mutate(arima_slot = as.character(arima_slot)) %>%
  left_join(
    period_hour_map %>%
      filter(period == "afternoon") %>%
      mutate(arima_slot = as.character(arima_slot)) %>%  # <-- 關鍵：轉成字串
      select(period, arima_slot, slot_in_period, hour_abs),
    by = c("period", "arima_slot")
  )

# 取出共同欄位，得到「標準化的 ARIMA‐清晨」長表
arima_afternoon_canonical <- arima_afternoon_std %>%
  select(
    model, day_abs, day_order, period, slot_in_period, hour_abs,
    origin_zone_id, dest_zone_id, y_pred
  ) %>%
  arrange(day_order, slot_in_period, origin_zone_id, dest_zone_id)

readr::write_csv(arima_afternoon_canonical, "arima_afternoon_canonical.csv")
