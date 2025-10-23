
library(dplyr)
obs <- readr::read_csv("C:/traffic-flow/data/processed/obs_period_canonical.csv") %>%
  filter(day_abs == 232, period == "morning") %>%
  group_by(origin_zone_id) %>%
  summarise(obs_sum = sum(y_sum), .groups = "drop")  # 你叫 y_sum

lstm <- readr::read_csv("C:/traffic-flow/data/processed/lstm_morning_canonical_period.csv") %>%
  filter(day_abs == 232, period == "morning") %>%
  group_by(origin_zone_id) %>%
  summarise(lstm_val = sum(y_pred), .groups = "drop")  # 這是看起來像每小時平均的值加總

left_join(obs, lstm, by="origin_zone_id") %>%
  mutate(ratio = obs_sum / lstm_val) %>%
  print(n=Inf)
