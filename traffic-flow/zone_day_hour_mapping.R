
setwd("C:\\traffic-flow")

# 區域對齊
zone_map <- tibble(
  pos          = 1:6,
  tensor_zone  = LETTERS[1:6],            # A-F (TensorTS)
  arima_zone   = as.character(1:6),       # 1-6 (ARIMA)
  lstm_zone    = as.character(0:5),       # 0-5 (LSTM)
  zone_id      = c(186, 100, 230, 161, 162, 163)
)

zone_map

readr::write_csv(zone_map, "zone_map.csv")

# 天數對齊
day_map <- tibble(
  day_order  = 1:21,
  tensor_day = LETTERS[1:21],                 # A-U (TensorTS)
  arima_day  = as.character(1:21),            # 1-21 (ARIMA)
  lstm_day   = as.character(232:252),         # 232-252 (LSTM)
  day_abs    = 232:252
)

day_map
readr::write_csv(day_map, "day_map.csv")

# 時段/小時對齊
## 小工具：產生一段時段表
build_period_tbl <- function(period_name, hours, tensor_labels, arima_labels) {
  tibble(
    period         = period_name,
    slot_in_period = seq_along(hours),
    hour_abs       = hours,
    hour_label     = sprintf("%02d:00", hours),
    tensor_slot    = tensor_labels[seq_along(hours)],
    arima_slot     = arima_labels[seq_along(hours)]
  )
}

morning_tbl <- build_period_tbl(
  period_name   = "morning",
  hours         = 1:6,                  # 01..06
  tensor_labels = strsplit("A B C D E F"," ")[[1]],
  arima_labels  = as.character(1:6)
)

afternoon_tbl <- build_period_tbl(
  period_name   = "afternoon",
  hours         = 14:19,                # 14..19
  tensor_labels = strsplit("A B C D E F"," ")[[1]],
  arima_labels  = as.character(1:6)
)

night_tbl <- build_period_tbl(
  period_name   = "night",
  hours         = 20:24,                # 20..24（5 小時）
  tensor_labels = strsplit("A B C D E"," ")[[1]],  # 夜間 5 槽
  arima_labels  = as.character(1:5)
)

period_hour_map <- bind_rows(morning_tbl, afternoon_tbl, night_tbl)

period_hour_map
readr::write_csv(period_hour_map, "period_hour_map.csv")

