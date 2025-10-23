library(tidyverse)
library(janitor)

# 讀檔與統一命名
## 讀對齊字典（上一階段做好的）
zone_map        <- readr::read_csv("zone_map.csv")
day_map         <- readr::read_csv("day_map.csv")
period_hour_map <- readr::read_csv("period_hour_map.csv")

## 讀 TensorTS 清晨原始長表
tensor_afternoon <- readr::read_csv("tensor_afternoon.csv")


## 統一命名（請按你的實際欄位名調整以下 rename）
tensor_afternoon_std <- tensor_afternoon %>%
  janitor::clean_names() %>%   # 轉小寫＋底線
  rename(
    tensor_day  = var1,   # ← 若你的天數欄不是 idx3，請改成實際欄名
    tensor_o    = var2,   # ← 起點欄
    tensor_d    = var3,   # ← 終點欄
    tensor_slot = var4,   # ← 清晨的時段槽位（A–F）
    y_pred      = value   # ← 預測值欄
  ) %>%
  mutate(
    model  = "TensorTS",
    period = "afternoon"
  )

# 對齊「天數」（A–U → 232–252 / 1–21）
## 先把 day_map 的 key 欄位轉字串（保險）
day_map <- day_map %>%
  mutate(
    tensor_day = as.character(tensor_day),
    arima_day  = as.character(arima_day),
    lstm_day   = as.character(lstm_day)
  )

## join 成統一的 day_abs / day_order
tensor_afternoon_std <- tensor_afternoon_std %>%
  mutate(tensor_day = as.character(tensor_day)) %>%
  left_join(day_map %>% select(tensor_day, day_abs, day_order),
            by = "tensor_day")

# 對齊「起訖區域」（A–F → 186/100/230/161/162/163）
tensor_afternoon_std <- tensor_afternoon_std %>%
  mutate(
    tensor_o = as.character(tensor_o),
    tensor_d = as.character(tensor_d)
  ) %>%
  # 起點
  left_join(
    zone_map %>% select(tensor_zone, zone_id) %>%
      rename(origin_zone_id = zone_id),
    by = c("tensor_o" = "tensor_zone")
  ) %>%
  # 終點
  left_join(
    zone_map %>% select(tensor_zone, zone_id) %>%
      rename(dest_zone_id = zone_id),
    by = c("tensor_d" = "tensor_zone")
  )

# 對齊「清晨槽位 → 小時」（A–F → 01–06）
tensor_afternoon_std <- tensor_afternoon_std %>%
  mutate(tensor_slot = as.character(tensor_slot)) %>%
  left_join(
    period_hour_map %>%
      filter(period == "afternoon") %>%
      mutate(tensor_slot = as.character(tensor_slot)) %>%
      select(period, tensor_slot, slot_in_period, hour_abs),
    by = c("period", "tensor_slot")
  )

# 取共同欄位，輸出 canonical 長表
tensor_afternoon_canonical <- tensor_afternoon_std %>%
  select(
    model, day_abs, day_order, period,
    slot_in_period, hour_abs,
    origin_zone_id, dest_zone_id,
    y_pred
  ) %>%
  arrange(day_order, slot_in_period, origin_zone_id, dest_zone_id)

readr::write_csv(tensor_afternoon_canonical, "tensor_afternoon_canonical.csv")





