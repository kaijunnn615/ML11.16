
################################################
## Step 4: Prepare XGBoost Data
################################################

# 合併訓練與測試資料，統一類別編碼
all_features <- rbind(train_data, test_data)

# 將類別變數轉換成數值矩陣
# 排除 Id 與預測目標 Logsalary
x.all <- model.matrix(
  Logsalary ~ . - Id,
  data = all_features
)

# 移除截距欄位
x.all <- x.all[
  , colnames(x.all) != "(Intercept)",
  drop = FALSE
]

# 依照原本的資料筆數切分
n.train <- nrow(train_data)

x.train <- x.all[
  1:n.train, ,
  drop = FALSE
]

x.test <- x.all[
  (n.train + 1):nrow(x.all), ,
  drop = FALSE
]

# 建立目標變數
y.train <- train_data$Logsalary
y.test <- test_data$Logsalary

# 檢查結果
dim(x.train)
dim(x.test)

# 確認兩組矩陣欄位一致
identical(colnames(x.train), colnames(x.test))


################################################
## Step 5: Create XGBoost DMatrix
################################################

# 載入 XGBoost
library(xgboost)

# 建立訓練資料
dtrain <- xgb.DMatrix(
  data = x.train,
  label = y.train
)

# 建立測試資料
dtest <- xgb.DMatrix(
  data = x.test,
  label = y.test
)

# 確認資料筆數
length(getinfo(dtrain, "label"))
length(getinfo(dtest, "label"))


################################################
## Step 6: Train XGBoost Model
################################################

# 設定模型參數
params.xgb <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist"
)

# 固定亂數種子
set.seed(123)

# 訓練 XGBoost 模型
xgb.fit <- xgb.train(
  params = params.xgb,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain),
  verbose = 0
)

# 查看模型
print(xgb.fit)

# 確認模型存在
exists("xgb.fit")


################################################
## Step 7: XGBoost Prediction
################################################

# 預測測試集
pred.xgb <- predict(
  xgb.fit,
  newdata = dtest
)

# 查看前 6 筆預測值
head(pred.xgb)

# 查看預測筆數
length(pred.xgb)

# 確認預測結果存在
exists("pred.xgb")


################################################
## Step 8: XGBoost Evaluation
################################################

# 計算 MSE
mse.xgb <- mean(
  (y.test - pred.xgb)^2
)

# 計算 RMSE
rmse.xgb <- sqrt(mse.xgb)

# 顯示結果
cat("XGBoost MSE =", mse.xgb, "\n")
cat("XGBoost RMSE =", rmse.xgb, "\n")

################################################
## Step 9: XGBoost Actual vs Predicted
################################################

# 繪製 XGBoost 預測散佈圖
plot(
  y.test,
  pred.xgb,
  main = "XGBoost: Actual vs Predicted Logsalary",
  xlab = "Actual Logsalary",
  ylab = "Predicted Logsalary",
  col = "darkorange",
  pch = 19
)

# 加入理想預測線
abline(
  a = 0,
  b = 1,
  col = "red",
  lty = 2,
  lwd = 2
)


################################################
## Step 10: XGBoost Variable Importance
################################################

# 計算變數重要性
importance.xgb <- xgb.importance(
  feature_names = colnames(x.train),
  model = xgb.fit
)

# 查看前 10 個重要變數
head(importance.xgb, 10)

# 繪製變數重要性圖
xgb.plot.importance(
  importance_matrix = importance.xgb,
  top_n = 10,
  main = "XGBoost Top 10 Important Features"
)

################################################
## Step 11: Three Models Comparison
################################################

# 建立三種模型比較表
model.comparison <- data.frame(
  Model = c("CART", "RF", "XGBoost"),
  MSE = c(
    0.1904637,
    0.1141903,
    0.1248529
  ),
  RMSE = c(
    0.4364215,
    0.3379205,
    0.3533453
  )
)

# 顯示比較表
print(model.comparison)

# 繪製 RMSE 比較圖
barplot(
  model.comparison$RMSE,
  names.arg = model.comparison$Model,
  col = c("steelblue", "darkgreen", "darkorange"),
  main = "CART vs RF vs XGBoost: RMSE",
  xlab = "Machine Learning Model",
  ylab = "RMSE",
  ylim = c(0, 0.55)
)

################################################
## Step 12: MSE Comparison
################################################

barplot(
  model.comparison$MSE,
  names.arg = model.comparison$Model,
  col = c("steelblue", "darkgreen", "darkorange"),
  main = "CART vs RF vs XGBoost: MSE",
  xlab = "Machine Learning Model",
  ylab = "MSE",
  ylim = c(0, 0.25)
)

################################################
## Step 13: Save XGBoost Plot
################################################

png(
  "XGBoost_Actual_vs_Predicted.png",
  width = 1000,
  height = 700,
  res = 120
)

plot(
  y.test,
  pred.xgb,
  main = "XGBoost: Actual vs Predicted Logsalary",
  xlab = "Actual Logsalary",
  ylab = "Predicted Logsalary",
  col = "darkorange",
  pch = 19
)

abline(0, 1, col = "red", lty = 2, lwd = 2)

dev.off()
