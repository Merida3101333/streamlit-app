f <- "C:\\traffic-flow\\y_pred_morning.RData"

# 1) 先載入到「獨立環境」避免汙染工作區
e <- new.env()

# 回傳被載入的物件名稱向量
objs <- load(f, envir = e)   

# 2) 看看有哪些物件
objs
ls(e)

# 3) 快速查看每個物件的結構與前幾列
lapply(objs, function(nm) {
  cat("\n==", nm, "==\n")
  print(utils::head(e[[nm]]))
  utils::str(e[[nm]])
})

# 1) 取出物件
x <- e[["y_pred"]]

# 2) 若無 dimnames，先補上索引名稱（避免輸出成 1,2,3... 不知所指）
if (is.null(dimnames(x))) {
  dimnames(x) <- list(
    orig = seq_len(dim(x)[1]),
    dest = seq_len(dim(x)[2]),
    idx3 = seq_len(dim(x)[3]),  # 你可事後改列名
    idx4 = seq_len(dim(x)[4])
  )
}

# 3) 攤平為長表（含所有維度索引 + 數值）
df <- as.data.frame.table(x, responseName = "value", stringsAsFactors = FALSE)

# 4)（可選）改易懂欄名
#names(df)[1:4] <- c("orig","dest","k","m")

# 5) 匯出 CSV
write.csv(df, file = "morning_long.csv", row.names = FALSE, fileEncoding = "UTF-8")