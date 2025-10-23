f <- "C:\\traffic-flow\\四時段預測結果.RData"

# 1) 先載入到「獨立環境」避免汙染工作區
e <- new.env()

# 回傳被載入的物件名稱向量
objs <- load(f, envir = e)   

# 2) 看看有哪些物件
objs
ls(e)

# 把需要的那個物件取出
pred_list <- get("plots", envir = e)

# 看資料維度＝幾列、幾欄
dim(pred_list)

# 列出欄位名稱（變數名）
names(pred_list) 

# 看前幾列資料（預設 6 列）
utils::head(pred_list)

# 逐一輸出長表：
invisible(lapply(names(pred_list), function(nm) {
  x  <- pred_list[[nm]]
  df <- as.data.frame.table(x, responseName = "value")  # 含各維度索引 + 數值
  write.csv(df, file = paste0(nm, "_long.csv"), row.names = FALSE, fileEncoding = "UTF-8")
}))